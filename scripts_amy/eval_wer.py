#!/usr/bin/env python3
"""
Measure English intelligibility of a checkpoint by round-tripping lyrics.

Synthesize each held-out score, transcribe the result with whisper, and
compare against the lyrics that went in. The resulting WER is the same metric
the VocalRender paper reports (4.44 on Opencpop, Mandarin), so a number from
this script is directly comparable to the published Chinese figure -- which is
the point: it tells us how far English is behind, in units someone else has
already calibrated.

This exists because judging "does it sound less Chinese now?" by ear does not
scale across a data-scale sweep, and because no published ablation says how
much English singing data a cross-lingual finetune needs. We have to measure
our own curve.

Prompt audio is mandatory (the checkpoints were trained with
prompt_audio_prob=1.0), and it is drawn from a *different segment of the same
song* -- matching how training pairs prompts, without letting the model hear
the exact target it is being asked to sing.

Usage:
    python scripts_amy/eval_wer.py \\
        --ckpt_dir pretrained_models/VocalRender \\
        --json_file data/gtsinger_en/annotations_val.json \\
        --audio_root C:/gts \\
        --out_dir outputs/wer_baseline --limit 40
"""

import argparse
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

WORD_RE = re.compile(r"[^a-z0-9']+")


def load_infer_module():
    """Import the single-sample inference script as a module.

    It is a script, not a package, but its model loading and prompt building
    are exactly what we need -- reimplementing them here would let the eval
    drift away from the real inference path, which would make the WER number
    describe something other than what we ship.
    """
    path = PROJECT_ROOT / "scripts" / "infer_vocalrender_svs_single.py"
    spec = importlib.util.spec_from_file_location("infer_single", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["infer_single"] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize(text: str):
    """Lowercase, strip punctuation, drop empties -> comparable word list."""
    return [w for w in WORD_RE.sub(" ", text.lower()).split() if w]


def wer(ref, hyp):
    """Levenshtein word error rate. Returns (errors, ref_len)."""
    if not ref:
        return (0, 0) if not hyp else (len(hyp), 0)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + (r != h)))  # substitution
        prev = cur
    return prev[-1], len(ref)


def reference_words(entry):
    """Lyrics as written in the score, minus breath/silence slots."""
    return normalize(" ".join(w for w in entry["word"] if w not in ("AP", "SP")))


def pick_prompts(entries, separator, group_indices):
    """Map each item to a prompt from a different segment of the same song."""
    groups = defaultdict(list)
    for e in entries:
        parts = e["item_name"].split(separator)
        key = separator.join(parts[i] for i in group_indices if i < len(parts))
        groups[key].append(e)

    prompts = {}
    for key, members in groups.items():
        for i, e in enumerate(members):
            other = members[(i + 1) % len(members)] if len(members) > 1 else e
            prompts[e["item_name"]] = other["wav_fn"]
    return prompts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--lora_dir", default="", help="LoRA adapter dir to apply over --ckpt_dir")
    p.add_argument("--json_file", required=True)
    p.add_argument("--audio_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--separator", default="#")
    p.add_argument("--group_indices", type=int, nargs="+", default=[0, 1])
    p.add_argument("--prompt_audio", default="",
                   help="Fixed prompt wav; default is same-song, different segment")
    p.add_argument("--prompt_max_frames", type=int, default=50)
    p.add_argument("--cfg_value", type=float, default=2.0)
    p.add_argument("--inference_timesteps", type=int, default=10)
    p.add_argument("--max_len", type=int, default=2000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--fsq_temperature", type=float, default=0.0)
    p.add_argument("--whisper_model", default="large-v3")
    p.add_argument("--compute_type", default="float16")
    p.add_argument("--keep_audio", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_root = Path(args.audio_root)

    entries = json.load(open(args.json_file, encoding="utf-8"))
    if args.limit > 0:
        entries = entries[: args.limit]
    print(f"[wer] {len(entries)} held-out scores from {args.json_file}", flush=True)

    prompts = pick_prompts(entries, args.separator, args.group_indices)

    infer = load_infer_module()
    from vocalrender.evaluation.audio_utils import normalize_audio
    from vocalrender.model.utils import get_out_sample_rate
    import soundfile as sf

    model = infer.load_svs_model(args.ckpt_dir, device=args.device,
                                 lora_dir=args.lora_dir or None)
    sample_rate = get_out_sample_rate(model)

    from faster_whisper import WhisperModel
    asr = WhisperModel(args.whisper_model, device=args.device,
                       compute_type=args.compute_type)

    rows, tot_err, tot_ref = [], 0, 0

    for n, entry in enumerate(entries, 1):
        name = entry["item_name"]
        prompt_wav = (Path(args.prompt_audio) if args.prompt_audio
                      else audio_root / prompts[name])
        if not prompt_wav.is_file():
            print(f"[{n}/{len(entries)}] SKIP {name}: no prompt audio", flush=True)
            continue

        svs_prompt = infer.build_svs_prompt_from_entry(entry, model)
        feats = infer.encode_prompt_audio(str(prompt_wav), model,
                                          max_frames=args.prompt_max_frames)
        if feats is None or feats.numel() == 0:
            print(f"[{n}/{len(entries)}] SKIP {name}: prompt encode failed", flush=True)
            continue

        with torch.no_grad():
            audio = model.generate_batch(
                target_texts=[svs_prompt],
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                max_len=args.max_len,
                verbose=False,
                temperature=args.temperature,
                fsq_temperature=args.fsq_temperature,
                prompt_audio_feats=[feats],
            )
        if not audio or audio[0] is None or audio[0].numel() == 0:
            print(f"[{n}/{len(entries)}] SKIP {name}: generation failed", flush=True)
            continue

        wav_out = out_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', name)}.wav"
        sf.write(str(wav_out), normalize_audio(audio[0].float().numpy().flatten()),
                 sample_rate)

        segs, _ = asr.transcribe(str(wav_out), language="en", beam_size=5)
        hyp = normalize(" ".join(s.text for s in segs))
        ref = reference_words(entry)
        err, n_ref = wer(ref, hyp)
        tot_err += err
        tot_ref += n_ref

        rows.append({"item_name": name, "ref": " ".join(ref), "hyp": " ".join(hyp),
                     "errors": err, "ref_words": n_ref,
                     "wer": round(err / n_ref, 4) if n_ref else None})
        print(f"[{n}/{len(entries)}] WER {err}/{n_ref} "
              f"({100*err/max(1,n_ref):5.1f}%)  ref: {' '.join(ref)[:45]}"
              f"  |  hyp: {' '.join(hyp)[:45]}", flush=True)

        if not args.keep_audio:
            wav_out.unlink(missing_ok=True)

    overall = tot_err / tot_ref if tot_ref else float("nan")
    report = {
        "ckpt_dir": args.ckpt_dir,
        "json_file": args.json_file,
        "scored": len(rows),
        "total_errors": tot_err,
        "total_ref_words": tot_ref,
        "wer": round(overall, 4),
        "wer_percent": round(100 * overall, 2),
        "samples": rows,
    }
    with open(out_dir / "wer_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[wer] {len(rows)} scored | {tot_err}/{tot_ref} words wrong "
          f"| WER {100*overall:.2f}%")
    print(f"[wer] paper reports 4.44 WER on Opencpop (Mandarin) for reference")
    print(f"[wer] -> {out_dir / 'wer_report.json'}")


if __name__ == "__main__":
    main()
