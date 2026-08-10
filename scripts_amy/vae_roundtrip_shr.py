#!/usr/bin/env python3
"""Does vocal grit survive the AudioVAE?

The frontier question from storm-reports/prompt-conditioned-svs-english. Three
solid premises that nobody has ever joined with an experiment:

  1. Vocal roughness -- rasp, growl, grit -- is carried by SUBHARMONICS, i.e.
     period-doubling that puts spectral energy at half-integer multiples of F0.
     Voices can be perceptually rough with entirely normal jitter and shimmer
     (Journal of Voice S0892199797800227, S0892199701000182).
  2. Singing vocoders model the waveform as periodic + aperiodic. A subharmonic
     is neither -- it is periodic at F0/2 (Period Singer, arXiv 2406.09894).
  3. Audio VAEs lose high-frequency spectral fidelity in reconstruction
     (Stable Audio Open analysis, OpenReview 5PBKxl7o49).

If grit dies in a bare encode/decode with no generation involved, then no
amount of training, LoRA rank, or prompt engineering can put it back, and
per-artist conversion has to happen AFTER the VAE. If it survives, the loss is
in generation rather than representation and is worth attacking directly.

This script isolates the VAE. No LM, no DiT, no sampling -- encode, decode,
compare.

What it measures
----------------
Subharmonic-to-harmonic ratio (SHR), per voiced frame, on the original and on
the round-tripped signal. For each frame we take the F0, then sum spectral
magnitude in narrow bands around the integer harmonics (n*F0) and around the
half-integer subharmonics ((n-0.5)*F0). SHR = subharmonic energy / harmonic
energy. A growled note has visible energy between the harmonics; a clean note
does not.

A large negative delta in SHR after the round trip means the VAE is discarding
the acoustic carrier of grit.

The band-energy check is a cross-check on the same question from the mixing
engineer's side: rasp is conventionally located around 2.5 kHz with harmonic
partners at 5 and 10 kHz, and the 10 kHz partner is what lossy representations
kill first.

Usage
-----
    .venv/Scripts/python.exe scripts_amy/vae_roundtrip_shr.py \
      --ckpt_dir pretrained_models/VocalRender \
      --audio "C:/path/to/growled_vocal.wav" \
      --out_dir outputs/vae_roundtrip

Point --audio at a genuinely rasped/growled phrase. A clean sung note has
little subharmonic energy to begin with, so it cannot show a loss. Pass
several files to get a distribution rather than one number.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


# ---------------------------------------------------------------------------
# VAE loading
# ---------------------------------------------------------------------------

def load_audio_vae(ckpt_dir: str, device: str):
    """Load just the AudioVAE out of a VocalRender checkpoint.

    Mirrors the resolution order in scripts/infer_vocalrender_svs_single.py so
    this measures the same VAE inference actually uses.
    """
    ckpt_path = Path(ckpt_dir)
    if (ckpt_path / "latest").exists():
        ckpt_path = ckpt_path / "latest"

    with open(ckpt_path / "config.json") as f:
        config_dict = json.load(f)
    architecture = str(config_dict.get("architecture", "voxcpm")).lower()

    if architecture == "voxcpm2":
        from vocalrender.model.voxcpm2 import VoxCPMConfig as ConfigClass
        from vocalrender.modules.audiovae import AudioVAEV2 as AudioVAEClass
    else:
        from vocalrender.model.voxcpm import VoxCPMConfig as ConfigClass
        from vocalrender.modules.audiovae import AudioVAE as AudioVAEClass

    config = ConfigClass(**config_dict)
    audio_vae_config = getattr(config, "audio_vae_config", None)
    audio_vae = AudioVAEClass(config=audio_vae_config) if audio_vae_config else AudioVAEClass()

    vae_sf_path = ckpt_path / "audiovae.safetensors"
    vae_pt_path = ckpt_path / "audiovae.pth"
    if vae_sf_path.exists():
        from safetensors.torch import load_file
        vae_sd = load_file(str(vae_sf_path))
    elif vae_pt_path.exists():
        blob = torch.load(str(vae_pt_path), map_location="cpu", weights_only=True)
        vae_sd = blob.get("state_dict", blob)
    else:
        raise FileNotFoundError(f"AudioVAE weights not found in {ckpt_path}")

    audio_vae.load_state_dict(vae_sd)
    # The VAE is kept in float32 during training too -- see runners/svs.py.
    return audio_vae.to(torch.float32).to(device).eval(), architecture


@torch.no_grad()
def roundtrip(audio_vae, wav: np.ndarray, sr: int, device: str) -> tuple[np.ndarray, int]:
    """Encode then decode, with no generative step in between.

    ``encode`` returns the posterior mean (``mu``) rather than a sample, so
    this is the deterministic path: we are measuring what the representation
    can carry, not adding sampling noise on top.
    """
    # V2 is an ASYMMETRIC VAE: it encodes at in_sample_rate (16 kHz on this
    # checkpoint) and decodes at out_sample_rate (48 kHz). Everything above the
    # 8 kHz encode-side Nyquist is therefore absent from the latent by
    # construction, and whatever appears above it in the output is the
    # decoder's invention. The encoder asserts on a sample-rate mismatch, so
    # the resample here is mandatory, not a convenience.
    in_sr = int(getattr(audio_vae, "in_sample_rate", sr) or sr)
    if sr != in_sr:
        import librosa

        enc_wav = librosa.resample(wav, orig_sr=sr, target_sr=in_sr)
    else:
        enc_wav = wav

    x = torch.from_numpy(np.ascontiguousarray(enc_wav)).to(torch.float32).to(device)
    if x.ndim == 1:
        x = x[None, None, :]
    elif x.ndim == 2:
        x = x[None, :, :]

    latent = audio_vae.encode(x, in_sr)
    if isinstance(latent, dict):
        latent = latent.get("mu", next(iter(latent.values())))
    elif isinstance(latent, (tuple, list)):
        latent = latent[0]

    decoded = audio_vae.decode(latent)
    if isinstance(decoded, dict):
        decoded = decoded.get("audio", next(iter(decoded.values())))
    elif isinstance(decoded, (tuple, list)):
        decoded = decoded[0]

    out_sr = int(getattr(audio_vae, "out_sample_rate", sr) or sr)
    return decoded.squeeze().detach().cpu().float().numpy(), out_sr


# ---------------------------------------------------------------------------
# Subharmonic measurement
# ---------------------------------------------------------------------------

def subharmonic_ratio(
    wav: np.ndarray,
    sr: int,
    f0: np.ndarray,
    voiced: np.ndarray,
    hop: int,
    n_fft: int = 2048,
    n_harm: int = 12,
    band_frac: float = 0.12,
) -> np.ndarray:
    """Per-frame subharmonic-to-harmonic ratio.

    For each voiced frame, sum magnitude in +/- ``band_frac * F0`` bands around
    the integer harmonics and around the half-integer subharmonics, and return
    their ratio. Frames with no measurable harmonic energy are dropped.

    ``band_frac`` at 0.12 keeps the harmonic and subharmonic bands disjoint
    (they are 0.5*F0 apart) while staying wide enough to tolerate vibrato
    within a frame.
    """
    import librosa

    spec = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop, center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    n_frames = min(spec.shape[1], len(f0))

    out = np.full(n_frames, np.nan, dtype=np.float64)
    nyquist = sr / 2.0

    for i in range(n_frames):
        if not voiced[i] or not np.isfinite(f0[i]) or f0[i] <= 0:
            continue
        base = float(f0[i])
        half_bw = band_frac * base

        harm_energy = 0.0
        sub_energy = 0.0
        col = spec[:, i]

        for n in range(1, n_harm + 1):
            hf = n * base
            if hf >= nyquist:
                break
            m = (freqs >= hf - half_bw) & (freqs <= hf + half_bw)
            if m.any():
                harm_energy += float(col[m].sum())

            sf_ = (n - 0.5) * base
            if sf_ <= 0 or sf_ >= nyquist:
                continue
            m = (freqs >= sf_ - half_bw) & (freqs <= sf_ + half_bw)
            if m.any():
                sub_energy += float(col[m].sum())

        if harm_energy > 1e-9:
            out[i] = sub_energy / harm_energy

    return out


def band_energy_db(wav: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Total magnitude in a frequency band, in dB. Cross-check on SHR."""
    import librosa

    spec = np.abs(librosa.stft(wav, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    m = (freqs >= lo) & (freqs < hi)
    e = float(spec[m].sum())
    return 20.0 * np.log10(e + 1e-12)


def analyse(path: Path, audio_vae, device: str, sr_target: int, out_dir: Path) -> dict:
    import librosa

    wav, sr = sf.read(str(path), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != sr_target:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=sr_target)
        sr = sr_target

    recon, out_sr = roundtrip(audio_vae, wav, sr, device)
    if out_sr != sr:
        # Compare like with like: the SHR bands are defined in Hz against an
        # F0 track measured on the original.
        recon = librosa.resample(recon, orig_sr=out_sr, target_sr=sr)
    n = min(len(wav), len(recon))
    wav, recon = wav[:n], recon[:n]

    # F0 from the ORIGINAL only, and reused for both signals. Estimating F0
    # separately on the reconstruction would confound "the grit is gone" with
    # "the tracker locked onto a different octave", which is exactly the
    # failure mode subharmonics cause in pitch trackers.
    hop = 256
    f0, voiced, _ = librosa.pyin(
        wav, sr=sr, fmin=65.0, fmax=1200.0, hop_length=hop, fill_na=np.nan,
    )
    voiced = np.nan_to_num(voiced, nan=0.0).astype(bool)

    shr_orig = subharmonic_ratio(wav, sr, f0, voiced, hop)
    shr_recon = subharmonic_ratio(recon, sr, f0, voiced, hop)

    both = np.isfinite(shr_orig) & np.isfinite(shr_recon)
    n_frames = int(both.sum())
    if n_frames == 0:
        return {"file": path.name, "error": "no voiced frames with harmonic energy"}

    o, r = shr_orig[both], shr_recon[both]
    rel = float(np.median((r - o) / np.maximum(o, 1e-9)) * 100.0)

    bands = {}
    for name, (lo, hi) in {
        "2.0-3.0kHz": (2000, 3000),
        "4.5-5.5kHz": (4500, 5500),
        "9-11kHz": (9000, 11000),
        "12-16kHz": (12000, 16000),
    }.items():
        bands[name] = round(
            band_energy_db(recon, sr, lo, hi) - band_energy_db(wav, sr, lo, hi), 2
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / f"{path.stem}__vae_roundtrip.wav"), recon, sr)

    return {
        "file": path.name,
        "voiced_frames": n_frames,
        "shr_original_median": round(float(np.median(o)), 5),
        "shr_roundtrip_median": round(float(np.median(r)), 5),
        "shr_change_pct": round(rel, 2),
        "shr_original_p90": round(float(np.percentile(o, 90)), 5),
        "shr_roundtrip_p90": round(float(np.percentile(r, 90)), 5),
        "band_delta_db": bands,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure whether subharmonic (grit) energy survives an AudioVAE round trip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt_dir", required=True, help="VocalRender checkpoint dir")
    ap.add_argument("--audio", required=True, nargs="+",
                    help="One or more vocal wavs. Use genuinely rasped/growled material.")
    ap.add_argument("--out_dir", default="outputs/vae_roundtrip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sample_rate", type=int, default=48000)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device != args.device:
        print("[VAE-RT] CUDA unavailable, falling back to CPU", file=sys.stderr)

    audio_vae, arch = load_audio_vae(args.ckpt_dir, device)
    print(f"[VAE-RT] Loaded AudioVAE ({arch}) on {device}", file=sys.stderr)

    out_dir = Path(args.out_dir)
    results = []
    for spec in args.audio:
        # Absolute paths cannot go through Path().glob (pathlib rejects
        # non-relative patterns), so only treat a spec as a pattern when it
        # actually contains wildcard characters.
        if any(ch in spec for ch in "*?["):
            base = Path(spec).anchor or "."
            pattern = spec[len(base):] if base != "." else spec
            matches = sorted(Path(base).glob(pattern))
        else:
            matches = [Path(spec)]
        for path in matches:
            if not path.is_file():
                print(f"[VAE-RT] skip (not found): {path}", file=sys.stderr)
                continue
            print(f"[VAE-RT] {path.name}", file=sys.stderr)
            try:
                results.append(analyse(path, audio_vae, device, args.sample_rate, out_dir))
            except Exception as e:  # keep going across a batch
                print(f"[VAE-RT]   failed: {e}", file=sys.stderr)
                results.append({"file": path.name, "error": str(e)})

    ok = [r for r in results if "error" not in r]
    print("\n=== Subharmonic-to-harmonic ratio, original -> VAE round trip ===")
    for r in ok:
        print(
            f"{r['file'][:44]:<44} "
            f"{r['shr_original_median']:.4f} -> {r['shr_roundtrip_median']:.4f}  "
            f"({r['shr_change_pct']:+.1f}%)  [{r['voiced_frames']} frames]"
        )
    if ok:
        agg = float(np.median([r["shr_change_pct"] for r in ok]))
        print(f"\nMedian SHR change across {len(ok)} file(s): {agg:+.1f}%")
        print(
            "\nReading it: a large negative number means the VAE is discarding the\n"
            "acoustic carrier of rasp and growl, and no amount of training on this\n"
            "architecture can restore it -- per-artist conversion must move after\n"
            "the VAE. A number near zero means the representation is not the\n"
            "bottleneck and the loss is in generation."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "shr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_dir / 'shr_results.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
