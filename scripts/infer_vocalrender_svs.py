#!/usr/bin/env python3
"""
SVS (Singing Voice Synthesis) Inference Script

This script performs inference using a VoxCPM model fine-tuned for SVS tasks.
It loads the fine-tuned checkpoint (with extended SVS tokenizer) and runs 
inference on preprocessed validation data.

Usage:
    python scripts/infer_vocalrender_svs.py --config_path conf/svs_infer.yaml

Environment:
    This script requires GPU for inference. Run on a compute node with GPU access.
"""

import json
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

from tqdm import tqdm

import argbind
import numpy as np
import soundfile as sf
import torch

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vocalrender.evaluation.inference import run_inference_batch
from vocalrender.evaluation.svs_metrics import (
    SVSEvaluator,
    items_from_infer_results,
)


def _resolve_checkpoint_path(ckpt_dir: str) -> Path:
    path = Path(ckpt_dir)
    if (path / "latest").exists():
        path = path / "latest"
    return path.resolve()


from vocalrender.inference.backends import TTSRequest, build_tts_backend
from vocalrender.model.utils import get_out_sample_rate


def _apply_infer_legacy_aliases(scalars: dict) -> dict:
    """Normalize evaluator scalar keys for the metrics_summary.json schema."""
    out = {}
    for key, val in scalars.items():
        out[key.lower() if key.startswith("aes_") else key] = val
    return out


def _load_standalone_audio_vae(ckpt_dir: str, device: str):
    from vocalrender.training.vae_loader import load_audio_vae_for_eval
    ckpt_path = Path(ckpt_dir)
    if (ckpt_path / "latest").exists():
        ckpt_path = ckpt_path / "latest"
    return load_audio_vae_for_eval(str(ckpt_path)).to(device)


def _resolve_ckpt_path(ckpt_dir: str) -> Path:
    ckpt_path = Path(ckpt_dir)
    if (ckpt_path / "latest").exists():
        ckpt_path = ckpt_path / "latest"
    return ckpt_path


def _detect_svs_architecture(ckpt_dir: str) -> str:
    with (_resolve_ckpt_path(ckpt_dir) / "config.json").open() as f:
        return str(json.load(f).get("architecture", "voxcpm")).lower()


def _detect_svs_sample_rate(ckpt_dir: str) -> int:
    architecture = _detect_svs_architecture(ckpt_dir)
    return 48000 if architecture == "voxcpm2" else 44100


def _resolve_inference_devices(backend_cfg) -> list[str]:
    """Resolve ``inference_backend.devices`` to a normalized device list.

    Accepts:
        * ``"auto"`` / unset / ``None``  → all visible CUDA GPUs, or
          ``["cpu"]`` when CUDA is unavailable.
        * ``int`` (e.g. ``0``)            → ``["cuda:0"]``.
        * ``str`` (e.g. ``"cuda:1"`` /  ``"cpu"``) → singleton list.
        * ``list[int|str]``               → coerced to ``cuda:N`` / ``cpu``.

    Returns a non-empty list. ``devices[0]`` is the canonical "primary"
    device used for driver-side model loading and VAE decoding;
    ``len(devices) > 1`` triggers the multi_gpu backend path.
    """
    cfg = backend_cfg.get("devices") if isinstance(backend_cfg, dict) else None

    def _coerce(item) -> str:
        if isinstance(item, int):
            return f"cuda:{item}"
        if isinstance(item, str):
            s = item.strip()
            if s.lower() == "cpu":
                return "cpu"
            if s.startswith("cuda"):
                return s
            if s.isdigit():
                return f"cuda:{s}"
            return s
        raise ValueError(f"Unrecognised device entry: {item!r}")

    if cfg is None or (isinstance(cfg, str) and cfg.strip().lower() == "auto"):
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        return ["cpu"]

    if isinstance(cfg, (int, str)):
        return [_coerce(cfg)]

    if isinstance(cfg, list):
        if not cfg:
            raise ValueError("inference_backend.devices is an empty list")
        return [_coerce(d) for d in cfg]

    raise ValueError(
        f"inference_backend.devices must be 'auto' / int / str / list, "
        f"got {type(cfg).__name__}"
    )


def _resolve_prompt_audio_feats(
    *,
    prompt_audio_config: dict | None,
    samples: list[dict],
    val_ds,
) -> list | None:
    """Resolve per-sample prompt-audio latents from ``prompt_audio_config``.

    Returns a list of ``Optional[np.ndarray]`` aligned with ``samples`` order,
    or ``None`` when no prompt audio is configured. Mirrors the three branches
    used by the multi_gpu backend: pre-extracted (offline cache),
    static-ref (one or per-dataset fixed clip), and same-song (random
    matching-song slice). Shared between multi_gpu and nano_vllm dispatch
    paths so both backends apply identical resolution semantics.
    """
    def _require_all(feats: list | None) -> list:
        if feats is None:
            raise RuntimeError(
                "Prompt-conditioned inference requires prompt audio for every sample."
            )
        missing = [
            samples[pos]["index"]
            for pos, feat in enumerate(feats)
            if feat is None
        ]
        if missing:
            preview = ", ".join(str(idx) for idx in missing[:10])
            suffix = "..." if len(missing) > 10 else ""
            raise RuntimeError(
                "No prompt audio could be resolved for sample indices "
                f"{preview}{suffix}. Use a same-song pool with another segment "
                "for every sample, or set prompt_audio_mode=static_ref."
            )
        return feats

    if prompt_audio_config is None:
        return _require_all(None)
    from vocalrender.evaluation.inference import _extract_batch_prompt_audio

    sorted_indices = [s["index"] for s in samples]
    if "pre_extracted" in prompt_audio_config:
        pre_extracted = prompt_audio_config["pre_extracted"]
        return _require_all([
            pre_extracted[idx] if idx < len(pre_extracted) else None
            for idx in sorted_indices
        ])
    if "static_by_dataset" in prompt_audio_config or "static_default" in prompt_audio_config:
        by_dataset = prompt_audio_config.get("static_by_dataset") or {}
        default = prompt_audio_config.get("static_default")
        out = []
        for idx in sorted_indices:
            ds_name = val_ds[idx].get("dataset_name", "")
            out.append(by_dataset.get(ds_name, default))
        return _require_all(out)
    if "dataset" in prompt_audio_config and "song_index" in prompt_audio_config:
        return _require_all(_extract_batch_prompt_audio(
            sorted_indices,
            prompt_audio_config["dataset"],
            prompt_audio_config["song_index"],
            {
                global_idx: val_ds[global_idx]["song_name"]
                for global_idx in sorted_indices
                if val_ds[global_idx].get("song_name")
            },
            prompt_audio_config.get("max_frames", 50),
            __import__("random").Random(prompt_audio_config.get("seed", 42)),
            index_map=prompt_audio_config.get("index_map"),
        ))
    return _require_all(None)


def _select_sample_indices(num_samples: int, n_total: int, seed: int = 42) -> list[int]:
    """Pick which dataset rows to infer on.

    ``num_samples <= 0`` (or ``>= n_total``) keeps the whole split in natural
    order. Otherwise draws a deterministic random subset (fixed ``seed`` so
    eval runs are reproducible/comparable). Returns *sorted* original val_ds
    indices — sorted only for tidy shard access; the caller still re-sorts the
    sample list by duration for batching, so eval ordering is unaffected.

    Previously this was ``range(num_samples)``, i.e. always the first N rows.
    Because the preprocessed Arrow keeps rows in shard order, the first N were
    a length-biased cluster (e.g. the first 100 muse rows are all ~2 s), which
    made the eval set unrepresentative.
    """
    if num_samples <= 0 or num_samples >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), num_samples))


def _run_inference_nano_vllm(
    *,
    backend,
    val_ds,
    ckpt_dir: str,
    output_dir: Path,
    num_samples: int,
    cfg_value: float,
    inference_timesteps: int,
    max_len: int,
    save_audio: bool,
    request_audio: bool,
    return_audio_arrays: bool,
    save_reference: bool,
    save_score: bool,
    save_audio_max: int,
    sample_rate: int,
    vae_device: str,
    verbose: bool = True,
    prompt_audio_config: dict | None = None,
):
    """Run SVS inference via the nano-vllm backend and produce result dicts
    whose schema matches :func:`run_inference_batch`'s.

    Reference-audio decoding is deferred until after backend shutdown so the
    driver-side VAE doesn't fight the nano-vllm workers for VRAM.

    ``save_audio_max`` caps the number of generated/reference wav pairs
    written to disk. ``-1`` (default) writes every sample. Eval metrics
    still consume all samples — only the on-disk wav count is bounded.
    """
    from vocalrender.evaluation.audio_utils import decode_reference_audio, normalize_audio

    if save_audio:
        output_dir.mkdir(parents=True, exist_ok=True)

    chosen = _select_sample_indices(num_samples, len(val_ds))
    col_names = val_ds.column_names if hasattr(val_ds, "column_names") else []
    has_est_dur = "estimated_duration" in col_names
    prep_cols = ["svs_prompt", "item_name"] + (["estimated_duration"] if has_est_dur else [])
    prep = val_ds.select(chosen).select_columns(prep_cols)

    samples = []
    for pos, idx in enumerate(chosen):
        row = prep[pos]
        svs = row.get("svs_prompt") or ""
        if not svs:
            continue
        item_name = row.get("item_name", f"sample_{idx:05d}")
        if isinstance(item_name, bytes):
            item_name = item_name.decode("utf-8")
        est = float(row["estimated_duration"]) if has_est_dur else 10.0
        samples.append({"index": idx, "svs_prompt": svs, "item_name": item_name,
                        "estimated_duration": est})
    samples.sort(key=lambda s: s["estimated_duration"])
    if not samples:
        return []

    prompt_audio_feats = _resolve_prompt_audio_feats(
        prompt_audio_config=prompt_audio_config,
        samples=samples,
        val_ds=val_ds,
    )
    requests = []
    for pos, s in enumerate(samples):
        req_kwargs = {}
        if prompt_audio_feats is not None:
            req_kwargs["prompt_audio_feats"] = prompt_audio_feats[pos]
        requests.append(
            TTSRequest(idx=s["index"], target_text=s["svs_prompt"], **req_kwargs)
        )

    print(
        f"[SVS Inference] [nano_vllm] dispatching {len(requests)} requests",
        file=sys.stderr,
    )
    t0 = time.time()
    raw = backend.generate(
        requests,
        return_latents=False,
        return_audio=request_audio,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        max_gen_len=max_len,
    )
    elapsed = time.time() - t0
    print(f"[SVS Inference] [nano_vllm] generate took {elapsed:.2f}s "
          f"({elapsed/max(len(raw),1):.2f}s/sample)", file=sys.stderr)

    by_idx = {r.idx: r for r in raw}

    saved_ids: set = set()
    score_tasks = []

    def _can_save() -> bool:
        return save_audio_max < 0 or len(saved_ids) < save_audio_max

    results = []
    for s in samples:
        idx = s["index"]
        item_name = s["item_name"]
        r = by_idx.get(idx)
        if r is None or r.error is not None:
            if verbose:
                print(f"[Sample {idx}] Skipped: {getattr(r, 'error', 'missing result')}", file=sys.stderr)
            continue
        result = {
            "sample_id": idx,
            "item_name": item_name,
            "svs_prompt": s["svs_prompt"][:200],
            "duration": 0.0,
            "estimated_duration": s["estimated_duration"],
        }
        if r.audio is not None and r.audio.size > 0:
            audio_np = normalize_audio(r.audio.astype(np.float32).reshape(-1))
            result["duration"] = len(audio_np) / sample_rate
            if return_audio_arrays:
                result["audio_np"] = audio_np
            if save_audio and _can_save():
                p = output_dir / f"{item_name}_generated.wav"
                # item_name may contain '/' (e.g. muse "song/segment"),
                # nesting a subdir that output_dir.mkdir didn't create.
                p.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(p), audio_np, sample_rate)
                result["generated_path"] = str(p)
                saved_ids.add(idx)
        # Score render is paired with the generated wav — only schedule it when
        # the wav made the save_audio_max cut. The GT notation comes straight
        # from the dataset row (bpm/word/pitch/note), so it's backend-agnostic;
        # ``val_ds`` is the same full-column dataset the reference decode uses.
        if save_score and idx in saved_ids:
            full_sample_for_score = val_ds[idx]
            score_tasks.append((
                int(full_sample_for_score.get("bpm", 120)),
                [str(w) for w in full_sample_for_score.get("word", [])],
                [int(pp) for pp in full_sample_for_score.get("pitch", [])],
                [str(n) for n in full_sample_for_score.get("note", [])],
                [int(pp) for pp in full_sample_for_score.get("pitch2word", [])],
                str(output_dir / f"{item_name}_score"),
            ))
        results.append(result)

    if save_audio and save_audio_max >= 0:
        print(
            f"[SVS Inference] [nano_vllm] saved {len(saved_ids)}/{len(results)} "
            f"generated wav(s) to disk (save_audio_max={save_audio_max})",
            file=sys.stderr,
        )

    # Reference decode runs after backend shutdown — driver-side VAE would
    # otherwise fight nano-vllm workers for VRAM at gpu_memory_utilization≥0.9.
    if save_reference or return_audio_arrays:
        print("[SVS Inference] [nano_vllm] shutting down backend before ref-audio decode", file=sys.stderr)
        backend.shutdown()
        try:
            vae = _load_standalone_audio_vae(ckpt_dir, vae_device)
        except Exception as e:
            print(f"[SVS Inference] [nano_vllm] reference decode unavailable: {e}", file=sys.stderr)
            vae = None
        if vae is not None:
            for result in results:
                sample = val_ds[result["sample_id"]]
                ref = decode_reference_audio(sample, vae, vae_device, sample_rate)
                if ref is None:
                    continue
                ref = normalize_audio(ref)
                if return_audio_arrays:
                    result["ref_audio_np"] = ref
                # Pair reference saves with generated saves so the cap
                # produces a coherent (gen, ref) set on disk.
                if save_reference and save_audio and result["sample_id"] in saved_ids:
                    p = output_dir / f"{result['item_name']}_reference.wav"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(p), ref, sample_rate)
                    result["reference_path"] = str(p)
                    result["reference_duration"] = len(ref) / sample_rate
            del vae

    if save_score and score_tasks:
        from vocalrender.utils.score_rendering import render_scores_parallel

        score_results = render_scores_parallel(score_tasks)
        for result in results:
            score_key = str(output_dir / f"{result['item_name']}_score")
            if score_key in score_results and score_results[score_key]:
                result["score_path"] = score_results[score_key]

    results.sort(key=lambda r: r["sample_id"])
    return results


def _apply_lyrics_only_to_dataset(val_ds, tokenizer):
    from vocalrender.preprocessing import rebuild_svs_prompt, create_lightweight_preprocessor

    _tokenizer = tokenizer
    if hasattr(_tokenizer, "tokenizer"):
        _tokenizer = _tokenizer.tokenizer
    _preprocessor = create_lightweight_preprocessor(_tokenizer)

    words_col = val_ds["word"]
    pitch_col = val_ds["pitch"]
    note_col = val_ds["note"]
    p2w_col = val_ds["pitch2word"]
    bpm_col = val_ds["bpm"]
    hs_col = val_ds["has_score"]

    new_prompts = []
    for i in range(len(val_ds)):
        sample = {
            "word": words_col[i], "pitch": pitch_col[i],
            "note": note_col[i], "pitch2word": p2w_col[i],
            "bpm": bpm_col[i], "has_score": hs_col[i],
        }
        new_prompts.append(rebuild_svs_prompt(
            sample, _tokenizer, _preprocessor, force_lyrics_only=True,
        ))
    return val_ds.remove_columns(["svs_prompt"]).add_column("svs_prompt", new_prompts)


def _build_prompt_audio_config(train_ds, val_ds, prompt_max_frames: int):
    from datasets import concatenate_datasets

    train_song_index = {}
    if train_ds is not None and len(train_ds) > 0:
        for idx, sn in enumerate(train_ds["song_name"]):
            if sn:
                train_song_index.setdefault(sn, []).append(idx)

    val_song_index = {}
    for idx, sn in enumerate(val_ds["song_name"]):
        if sn:
            val_song_index.setdefault(sn, []).append(idx)

    if train_ds is not None and len(train_ds) > 0:
        prompt_pool_ds = concatenate_datasets([train_ds, val_ds])
    else:
        prompt_pool_ds = val_ds

    prompt_pool_song_index = {sn: list(indices) for sn, indices in train_song_index.items()}
    val_offset = len(train_ds) if train_ds is not None else 0
    for idx, sn in enumerate(val_ds["song_name"]):
        if sn:
            prompt_pool_song_index.setdefault(sn, []).append(val_offset + idx)

    prompt_audio_config = {
        "dataset": prompt_pool_ds,
        "song_index": prompt_pool_song_index,
        "max_frames": prompt_max_frames,
        "seed": 42,
        "index_map": {idx: val_offset + idx for idx in range(len(val_ds))},
    }
    return prompt_audio_config, val_song_index, prompt_pool_song_index, val_offset


def _resolve_repo_path(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else Path(project_root) / p


def _build_static_ref_prompt_audio_config(
    *,
    backend,
    reference_wavs: dict | None,
    reference_wav_path: str | None,
    dataset_ref_key_map: dict | None,
):
    """Encode static ref-audio latents for inference-time prompt_audio_feats.

    Mirrors the training runner's static ref-audio path: each dataset maps to
    a named reference wav key, and that reference is prepended as
    [ref_audio_start, ref_audio, ref_audio_end] by model.generate_batch.
    """
    by_key = {}
    if reference_wav_path:
        resolved = _resolve_repo_path(reference_wav_path)
        by_key["_default"] = backend.encode_prompt_wav(str(resolved), padding_mode="right")

    for key, wav_path in (reference_wavs or {}).items():
        resolved = _resolve_repo_path(str(wav_path))
        by_key[str(key)] = backend.encode_prompt_wav(str(resolved), padding_mode="right")

    if not by_key:
        raise ValueError(
            "prompt_audio_mode=static_ref requires reference_wavs or reference_wav_path"
        )

    by_dataset = {}
    for ds_name, ref_key in (dataset_ref_key_map or {}).items():
        ref_key = str(ref_key)
        if ref_key not in by_key:
            raise KeyError(
                f"dataset_ref_key_map[{ds_name!r}]={ref_key!r}, "
                f"but available reference keys are {sorted(by_key)}"
            )
        by_dataset[str(ds_name)] = by_key[ref_key]

    default = by_key.get("_default")
    if default is None and len(by_key) == 1:
        default = next(iter(by_key.values()))

    return {
        "static_by_dataset": by_dataset,
        "static_default": default,
    }


def _run_inference_multi_gpu_backend(
    *,
    backend,
    val_ds,
    ckpt_dir: str,
    output_dir: Path,
    num_samples: int,
    cfg_value: float,
    inference_timesteps: int,
    max_len: int,
    save_audio: bool,
    request_audio: bool,
    return_audio_arrays: bool,
    save_reference: bool,
    save_score: bool,
    save_audio_max: int,
    sample_rate: int,
    vae_device: str,
    prompt_audio_config: dict | None = None,
    verbose: bool = True,
):
    """``save_audio_max`` caps the number of generated/reference/score
    artefacts written to disk. ``-1`` (default) writes every sample. Eval
    metrics still see all samples — only the on-disk count is bounded.
    """
    from vocalrender.evaluation.audio_utils import decode_reference_audio, normalize_audio

    if save_audio:
        output_dir.mkdir(parents=True, exist_ok=True)

    chosen = _select_sample_indices(num_samples, len(val_ds))
    col_names = val_ds.column_names if hasattr(val_ds, "column_names") else []
    has_est_dur = "estimated_duration" in col_names
    prep_cols = ["svs_prompt", "item_name"] + (["estimated_duration"] if has_est_dur else [])
    prep = val_ds.select(chosen).select_columns(prep_cols)

    samples = []
    for pos, idx in enumerate(chosen):
        row = prep[pos]
        svs = row.get("svs_prompt") or ""
        if not svs:
            continue
        item_name = row.get("item_name", f"sample_{idx:05d}")
        if isinstance(item_name, bytes):
            item_name = item_name.decode("utf-8")
        est = float(row["estimated_duration"]) if has_est_dur else 10.0
        samples.append({
            "index": idx,
            "svs_prompt": svs,
            "item_name": item_name,
            "estimated_duration": est,
        })
    samples.sort(key=lambda s: s["estimated_duration"])
    if not samples:
        return []

    prompt_audio_feats = _resolve_prompt_audio_feats(
        prompt_audio_config=prompt_audio_config,
        samples=samples,
        val_ds=val_ds,
    )

    requests = []
    for pos, sample in enumerate(samples):
        req_kwargs = {}
        if prompt_audio_feats is not None:
            req_kwargs["prompt_audio_feats"] = prompt_audio_feats[pos]
        requests.append(TTSRequest(
            idx=sample["index"],
            target_text=sample["svs_prompt"],
            **req_kwargs,
        ))

    print(
        f"[SVS Inference] [multi_gpu] dispatching {len(requests)} requests",
        file=sys.stderr,
    )
    t0 = time.time()
    raw = backend.generate(
        requests,
        return_latents=False,
        return_audio=request_audio,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        max_gen_len=max_len,
    )
    elapsed = time.time() - t0
    print(f"[SVS Inference] [multi_gpu] generate took {elapsed:.2f}s "
          f"({elapsed/max(len(raw),1):.2f}s/sample)", file=sys.stderr)

    by_idx = {r.idx: r for r in raw}
    results = []
    score_tasks = []
    saved_ids: set = set()

    def _can_save() -> bool:
        return save_audio_max < 0 or len(saved_ids) < save_audio_max

    for sample in samples:
        idx = sample["index"]
        item_name = sample["item_name"]
        r = by_idx.get(idx)
        if r is None or r.error is not None:
            if verbose:
                print(f"[Sample {idx}] Skipped: {getattr(r, 'error', 'missing result')}", file=sys.stderr)
            continue
        result = {
            "sample_id": idx,
            "item_name": item_name,
            "svs_prompt": sample["svs_prompt"][:200],
            "duration": 0.0,
            "estimated_duration": sample["estimated_duration"],
        }
        if r.audio is not None and r.audio.size > 0:
            audio_np = normalize_audio(r.audio.astype(np.float32).reshape(-1))
            result["duration"] = len(audio_np) / sample_rate
            if return_audio_arrays:
                result["audio_np"] = audio_np
            if save_audio and _can_save():
                gen_path = output_dir / f"{item_name}_generated.wav"
                # item_name may contain '/' (e.g. muse "song/segment"),
                # nesting a subdir that output_dir.mkdir didn't create.
                gen_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(gen_path), audio_np, sample_rate)
                result["generated_path"] = str(gen_path)
                saved_ids.add(idx)
        # Score render is paired with the generated wav — only schedule it
        # when the wav itself made the save_audio_max cut.
        if save_score and idx in saved_ids:
            full_sample_for_score = val_ds[idx]
            score_tasks.append((
                int(full_sample_for_score.get("bpm", 120)),
                [str(w) for w in full_sample_for_score.get("word", [])],
                [int(p) for p in full_sample_for_score.get("pitch", [])],
                [str(n) for n in full_sample_for_score.get("note", [])],
                [int(p) for p in full_sample_for_score.get("pitch2word", [])],
                str(output_dir / f"{item_name}_score"),
            ))
        results.append(result)

    if save_audio and save_audio_max >= 0:
        print(
            f"[SVS Inference] [multi_gpu] saved {len(saved_ids)}/{len(results)} "
            f"generated wav(s) to disk (save_audio_max={save_audio_max})",
            file=sys.stderr,
        )

    if save_reference or return_audio_arrays:
        print("[SVS Inference] [multi_gpu] shutting down backend before ref-audio decode", file=sys.stderr)
        backend.shutdown()
        try:
            vae = _load_standalone_audio_vae(ckpt_dir, vae_device)
        except Exception as e:
            print(f"[SVS Inference] [multi_gpu] reference decode unavailable: {e}", file=sys.stderr)
            vae = None
        if vae is not None:
            for result in results:
                sample = val_ds[result["sample_id"]]
                ref = decode_reference_audio(sample, vae, vae_device, sample_rate)
                if ref is None:
                    continue
                ref = normalize_audio(ref)
                if return_audio_arrays:
                    result["ref_audio_np"] = ref
                # Pair reference saves with generated saves so the cap
                # produces a coherent (gen, ref) set on disk.
                if save_reference and save_audio and result["sample_id"] in saved_ids:
                    ref_path = output_dir / f"{result['item_name']}_reference.wav"
                    ref_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(ref_path), ref, sample_rate)
                    result["reference_path"] = str(ref_path)
                    result["reference_duration"] = len(ref) / sample_rate
            del vae

    if save_score and score_tasks:
        from vocalrender.utils.score_rendering import render_scores_parallel

        score_results = render_scores_parallel(score_tasks)
        for result in results:
            score_key = str(output_dir / f"{result['item_name']}_score")
            if score_key in score_results and score_results[score_key]:
                result["score_path"] = score_results[score_key]

    results.sort(key=lambda r: r["sample_id"])
    return results


def load_svs_model(
    ckpt_dir: str,
    device: str = "cuda",
):
    """
    Load SVS fine-tuned model from checkpoint directory.
    
    The checkpoint contains:
    - model.safetensors or pytorch_model.bin: Model weights (with extended embeddings for SVS tokens)
    - config.json: Model configuration
    - audiovae.pth: Audio VAE weights
    - tokenizer.json, tokenizer_config.json, etc.: Extended tokenizer with SVS tokens
    """
    from vocalrender.model.voxcpm import VoxCPMModel, VoxCPMConfig
    from vocalrender.model.voxcpm2 import VoxCPM2Model, VoxCPMConfig as VoxCPMConfigV2
    from vocalrender.modules.audiovae.audio_vae import AudioVAE, AudioVAEConfig
    from transformers import LlamaTokenizerFast
    
    ckpt_path = Path(ckpt_dir)
    
    # Handle 'latest' symlink
    if (ckpt_path / "latest").exists():
        ckpt_path = ckpt_path / "latest"
    
    print(f"[SVS Inference] Loading model from: {ckpt_path}", file=sys.stderr)
    
    # Load config
    config_path = ckpt_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    

    with open(config_path) as f:
        config_dict = json.load(f)
    
    # Auto-detect architecture
    architecture = config_dict.get("architecture", "voxcpm").lower()
    is_v2 = (architecture == "voxcpm2")
    print(f"[SVS Inference] Detected architecture: {architecture}", file=sys.stderr)
    
    if is_v2:
        config = VoxCPMConfigV2(**config_dict)
    else:
        config = VoxCPMConfig(**config_dict)
    
    # Load tokenizer (with SVS tokens)
    print(f"[SVS Inference] Loading SVS tokenizer...", file=sys.stderr)
    tokenizer = LlamaTokenizerFast.from_pretrained(str(ckpt_path))
    print(f"[SVS Inference] Tokenizer vocab size: {len(tokenizer)}", file=sys.stderr)
    
    # Update config to use SVS-extended vocab size
    original_vocab_size = config.lm_config.vocab_size
    new_vocab_size = len(tokenizer)
    if new_vocab_size != original_vocab_size:
        print(f"[SVS Inference] Updating vocab_size in config: {original_vocab_size} -> {new_vocab_size}", file=sys.stderr)
        config.lm_config.vocab_size = new_vocab_size
    
    # Load AudioVAE (V1 or V2)
    print(f"[SVS Inference] Loading AudioVAE ({'V2' if is_v2 else 'V1'})...", file=sys.stderr)
    if is_v2:
        from vocalrender.modules.audiovae.audio_vae_v2 import AudioVAE as AudioVAEV2, AudioVAEConfig as AudioVAEConfigV2
        audio_vae_config = getattr(config, 'audio_vae_config', None)
        audio_vae = AudioVAEV2(config=audio_vae_config) if audio_vae_config else AudioVAEV2()
    else:
        audio_vae_config = getattr(config, 'audio_vae_config', None)
        audio_vae = AudioVAE(config=audio_vae_config) if audio_vae_config else AudioVAE()
    
    # Load VAE weights (support both safetensors and pytorch formats)
    vae_safetensors_path = ckpt_path / "audiovae.safetensors"
    vae_pth_path = ckpt_path / "audiovae.pth"
    
    if vae_safetensors_path.exists():
        try:
            from safetensors.torch import load_file as _load_sf
            vae_state_dict = _load_sf(str(vae_safetensors_path), device="cpu")
        except ImportError:
            vae_state_dict = torch.load(str(vae_pth_path), map_location="cpu", weights_only=True)["state_dict"]
    elif vae_pth_path.exists():
        checkpoint = torch.load(str(vae_pth_path), map_location="cpu", weights_only=True)
        vae_state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        raise FileNotFoundError(f"AudioVAE weights not found at {ckpt_path}")
    
    audio_vae.load_state_dict(vae_state_dict)
    
    # Create model with correct class
    print(f"[SVS Inference] Creating {'VoxCPM2Model' if is_v2 else 'VoxCPMModel'}...", file=sys.stderr)
    ModelClass = VoxCPM2Model if is_v2 else VoxCPMModel
    model = ModelClass(config, tokenizer, audio_vae, lora_config=None)

    # Load model weights
    safetensors_path = ckpt_path / "model.safetensors"
    pytorch_model_path = ckpt_path / "pytorch_model.bin"
    
    try:
        from safetensors.torch import load_file as load_safetensors
        SAFETENSORS_AVAILABLE = True
    except ImportError:
        SAFETENSORS_AVAILABLE = False
    
    if safetensors_path.exists() and SAFETENSORS_AVAILABLE:
        print(f"[SVS Inference] Loading weights from safetensors...", file=sys.stderr)
        model_state_dict = load_safetensors(str(safetensors_path))
    elif pytorch_model_path.exists():
        print(f"[SVS Inference] Loading weights from pytorch_model.bin...", file=sys.stderr)
        checkpoint = torch.load(str(pytorch_model_path), map_location="cpu", weights_only=True)
        model_state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        raise FileNotFoundError(
            f"Model weights not found. Expected {safetensors_path} or {pytorch_model_path}"
        )
    
    # Add VAE weights to state dict
    for kw, val in vae_state_dict.items():
        model_state_dict[f"audio_vae.{kw}"] = val
    
    # Load weights (strict=False for potential minor mismatches)
    model.load_state_dict(model_state_dict, strict=False)
    
    # Convert model to correct dtype for inference (matches from_local behavior)
    from vocalrender.model.utils import get_dtype
    lm_dtype = get_dtype(config.dtype)
    print(f"[SVS Inference] Converting model to dtype: {lm_dtype}", file=sys.stderr)
    model = model.to(lm_dtype)
    
    # Move to device and set to eval mode
    model = model.to(device).eval()
    model.device = device  # sync so _generate_batch creates tensors on correct device
    # Keep AudioVAE in float32 for decode precision
    model.audio_vae = model.audio_vae.to(torch.float32)
    
    print(f"[SVS Inference] Model loaded successfully on {device}", file=sys.stderr)
    return model


def load_preprocessed_split_data(
    preprocessed_data_path: str,
    val_datasets: list = None,
    val_songs: list = None,
    val_samples: int = 0,
):
    """Load preprocessed data and extract train/validation split."""
    from vocalrender.training.svs_data import load_preprocessed_svs_datasets

    train_ds, val_ds = load_preprocessed_svs_datasets(
        preprocessed_data_path,
        val_datasets=val_datasets,
        val_songs=val_songs,
        val_samples=val_samples,
    )
    if val_ds is None:
        raise ValueError(f"No validation data found. Specify val_datasets, val_songs, or val_samples.")
    
    print(f"[SVS Inference] Loaded {len(val_ds)} validation samples", file=sys.stderr)
    return train_ds, val_ds


@argbind.bind(without_prefix=True)
def infer(
    # Model configuration
    ckpt_dir: str = "",
    # Data configuration
    preprocessed_data_path: str = "",
    # Validation split (which samples to infer on)
    val_datasets: list = None,
    val_songs: list = None,
    val_samples: int = 0,
    # Inference parameters
    num_samples: int = -1,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    max_len: int = 2000,
    # Batch inference parameters
    use_batch_inference: bool = True,
    batch_size: int = 4,
    # Output configuration
    output_dir: str = "outputs/svs_inference",
    save_audio: bool = True,
    save_reference: bool = True,
    save_score: bool = False,
    # Cap on the number of generated/reference/score artefacts written to
    # disk. ``-1`` writes one set per inferred sample (the legacy
    # default). Eval metrics still consume every sample — only the
    # on-disk wav/png count is bounded. Useful for "infer 1000 samples
    # for SingMOS / RPA, but only keep 50 wavs for listening".
    save_audio_max: int = -1,
    # Config path (for argbind)
    config_path: str = "",
    # Evaluation metrics (nested dict, see YAML config for structure)
    eval_metrics: dict = None,
    # Prompt audio configuration
    prompt_max_frames: int = 50,
    prompt_audio_mode: str = "same_song",
    reference_wavs: dict = None,
    reference_wav_path: str = "",
    dataset_ref_key_map: dict = None,
    # Lyrics-only mode: force all prompts to use <SVS_MASK> for BPM/pitch/note
    lyrics_only: bool = False,
    # Inference backend dispatch (see conf/**/infer_*.yaml for the schema).
    # Unset / None keeps the default multi_gpu backend behaviour.
    inference_backend: dict = None,
):
    """
    SVS Batch Inference main function.
    
    Runs inference on preprocessed validation data and computes evaluation metrics.
    
    Args:
        ckpt_dir: Checkpoint directory containing fine-tuned SVS model
        preprocessed_data_path: Path to preprocessed SVS data (Arrow format)
        num_samples: Number of samples to generate (-1 for all)
        cfg_value: CFG (Classifier-Free Guidance) scale
        inference_timesteps: Number of diffusion inference steps
        max_len: Maximum generation length (audio frames)
        use_batch_inference: Whether to use batch inference for faster processing
        batch_size: Batch size for batch inference (higher = faster but more VRAM)
        output_dir: Output directory for batch inference
        save_reference: Whether to save reference audio
        save_score: Whether to render score notation images
        config_path: Path to YAML config file
        inference_backend: Dispatch dict, e.g.
            ``{type: multi_gpu, devices: [cuda:0, cuda:1]}``. ``devices``
            is the single source of truth for device placement (``"auto"``
            picks every visible GPU, falling back to CPU). Driver-side
            model loading uses ``devices[0]``; ``len(devices) > 1``
            triggers the multi_gpu backend automatically.
        eval_metrics: Dict of metric configs (singmos, aes, alignment), each with enabled flag and params
        prompt_max_frames: Max VAE latent frames for prompt audio
        save_audio_max: Cap on the number of generated/reference/score
            artefacts written to disk. ``-1`` (default) writes every
            inferred sample. Eval metrics still see all samples.
    """
    _ = config_path  # Used by argbind, not needed here
    # Released checkpoints were trained with prompt_audio_prob=1.0. Public
    # inference therefore exposes prompt-conditioned generation only.
    use_prompt_audio = True
    
    # Validate arguments
    if not ckpt_dir:
        print("Error: ckpt_dir is required", file=sys.stderr)
        sys.exit(1)
    
    if not preprocessed_data_path:
        print("Error: preprocessed_data_path is required", file=sys.stderr)
        sys.exit(1)

    # Single source of truth for device placement: ``inference_backend.devices``.
    # Resolves to a non-empty list (``"auto"`` → all visible CUDA GPUs, or
    # ``["cpu"]``); ``primary_device`` drives any driver-side model load,
    # ``is_multi_gpu`` triggers the multi_gpu backend path.
    devices = _resolve_inference_devices(inference_backend or {})
    primary_device = devices[0]
    is_multi_gpu = len(devices) > 1

    print(f"[SVS Inference] Configuration:", file=sys.stderr)
    print(f"  ckpt_dir: {ckpt_dir}", file=sys.stderr)
    print(f"  preprocessed_data_path: {preprocessed_data_path}", file=sys.stderr)
    print(f"  num_samples: {num_samples}", file=sys.stderr)
    print(f"  cfg_value: {cfg_value}", file=sys.stderr)
    print(f"  inference_timesteps: {inference_timesteps}", file=sys.stderr)
    print(f"  max_len: {max_len}", file=sys.stderr)
    print(f"  use_batch_inference: {use_batch_inference}", file=sys.stderr)
    print(f"  batch_size: {batch_size}", file=sys.stderr)
    print(f"  output_dir: {output_dir}", file=sys.stderr)
    print(f"  save_audio: {save_audio}"
          + (f" (max={save_audio_max})" if save_audio and save_audio_max >= 0 else ""),
          file=sys.stderr)
    print(f"  save_reference: {save_reference}", file=sys.stderr)
    print(f"  inference devices: {devices}", file=sys.stderr)
    print(f"  use_prompt_audio: {use_prompt_audio}", file=sys.stderr)
    print(f"  prompt_audio_mode: {prompt_audio_mode}", file=sys.stderr)
    print(f"  lyrics_only: {lyrics_only}", file=sys.stderr)

    # Driver-side model is needed for: lyrics_only token-rebuild path, and
    # for the legacy ``run_inference_batch`` fallback (single-device, no
    # backend abstraction). The two backend-dispatch paths (multi_gpu /
    # nano_vllm) load weights inside their workers and don't need it.
    need_driver_model = lyrics_only or not is_multi_gpu
    model = (
        load_svs_model(ckpt_dir, device=primary_device)
        if need_driver_model
        else None
    )
    sample_rate = get_out_sample_rate(model) if model is not None else _detect_svs_sample_rate(ckpt_dir)
    
    # Build the shared SVS evaluator — parses `eval_metrics`, resolves checkpoint
    # paths against `project_root`, and lazy-loads only the backends whose
    # sub-metrics are enabled.
    evaluator = SVSEvaluator(
        eval_metrics,
        project_root=Path(project_root),
        sample_rate=sample_rate,
        log_fn=lambda msg: print(msg, file=sys.stderr),
    )
    need_audio_arrays = evaluator.needs_audio_arrays()
    active_metrics = evaluator.active_metrics_label()
    if active_metrics:
        print(f"[Eval] Active metrics: {active_metrics}", file=sys.stderr)
        if evaluator.aes_axes:
            print(f"[Eval] AES axes: {evaluator.aes_axes}", file=sys.stderr)

    # ---- Batch inference on validation set ----
    train_ds, val_ds = load_preprocessed_split_data(
        preprocessed_data_path,
        val_datasets=val_datasets,
        val_songs=val_songs,
        val_samples=val_samples,
    )

    # ---- Lyrics-only: rebuild svs_prompt column from metadata ----
    if lyrics_only:
        if model is None:
            raise RuntimeError("lyrics_only requires a driver-side model/tokenizer.")
        val_ds = _apply_lyrics_only_to_dataset(val_ds, model.text_tokenizer)
        print(f"[Lyrics-Only] Rebuilt {len(val_ds)} prompts", file=sys.stderr)

    output_dir_path = Path(output_dir)

    # ---- Build prompt audio config if enabled ----
    prompt_audio_config = None
    if use_prompt_audio and prompt_audio_mode != "static_ref":
        prompt_audio_config, val_song_index, prompt_pool_song_index, _ = _build_prompt_audio_config(
            train_ds, val_ds, prompt_max_frames,
        )
        val_multi = {k: v for k, v in val_song_index.items() if len(v) >= 2}
        pool_multi = {k: v for k, v in prompt_pool_song_index.items() if len(v) >= 2}
        print(f"[SVS Inference] Prompt audio enabled (validation-style)", file=sys.stderr)
        print(f"[SVS Inference] Val prompt audio: {len(val_song_index)} songs, "
              f"{len(val_multi)} with ≥2 segments", file=sys.stderr)
        print(f"[SVS Inference] Combined prompt pool: {len(prompt_pool_song_index)} songs, "
              f"{len(pool_multi)} with ≥2 segments "
              f"({sum(len(v) for v in pool_multi.values())} eligible), "
              f"max_frames={prompt_max_frames}, seed=42", file=sys.stderr)
    elif use_prompt_audio:
        print("[SVS Inference] Prompt audio enabled (static ref-audio mode)", file=sys.stderr)

    backend_cfg = dict(inference_backend) if inference_backend else {}
    backend_type = backend_cfg.pop("type", "multi_gpu")
    model_arch = _detect_svs_architecture(ckpt_dir)

    if backend_type == "nano_vllm":
        # nano-vllm supports prompt_audio_feats (wrapper translates to the
        # server's ref_audio_latents byte path) and save_score (GT notation
        # rendered driver-side from the dataset row — backend-agnostic, see
        # _run_inference_nano_vllm).
        request_audio = bool(need_audio_arrays or save_audio)
        if request_audio:
            backend_cfg["load_audio_vae"] = True
        else:
            backend_cfg.setdefault("load_audio_vae", False)
        # nano_vllm honours ``inference_backend.devices``; if the user
        # didn't override, pass our resolved list down so the worker pool
        # uses the same set we surface in the launch banner.
        backend_cfg.setdefault("devices", devices)
        vae_device = primary_device
        # We already loaded a full SVS model on ``primary_device`` above to
        # hand to the evaluator constructors; release it before spawning
        # nano-vllm workers so we don't double-book VRAM on cuda:0.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[SVS Inference] Using NANO_VLLM backend ({backend_cfg})", file=sys.stderr)
        backend = build_tts_backend(
            dict(backend_cfg, type="nano_vllm"),
            pretrained_path=ckpt_dir,
            log_fn=lambda m: print(f"  {m}", file=sys.stderr),
        )
        try:
            if use_prompt_audio and prompt_audio_mode == "static_ref":
                # NanoVLLMBackend.encode_prompt_wav is a CPU-side standalone VAE
                # encode (see nano_vllm.py:encode_prompt_wav), so it doesn't fight
                # the just-spawned nano-vllm workers for VRAM.
                prompt_audio_config = _build_static_ref_prompt_audio_config(
                    backend=backend,
                    reference_wavs=reference_wavs,
                    reference_wav_path=reference_wav_path,
                    dataset_ref_key_map=dataset_ref_key_map,
                )
            results = _run_inference_nano_vllm(
                backend=backend,
                val_ds=val_ds,
                ckpt_dir=ckpt_dir,
                output_dir=output_dir_path,
                num_samples=num_samples,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                max_len=max_len,
                save_audio=save_audio,
                request_audio=request_audio,
                return_audio_arrays=need_audio_arrays,
                save_reference=save_reference,
                save_score=save_score,
                save_audio_max=save_audio_max,
                sample_rate=sample_rate,
                vae_device=vae_device,
                verbose=True,
                prompt_audio_config=prompt_audio_config,
            )
        finally:
            try:
                backend.shutdown()
            except Exception:
                pass
    elif backend_type == "multi_gpu" and is_multi_gpu:
        backend_devices = list(devices)
        print(
            f"[SVS Inference] Using MULTI_GPU BACKEND mode "
            f"({len(backend_devices)} worker(s): {backend_devices})",
            file=sys.stderr,
        )
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        request_audio = bool(need_audio_arrays or save_audio)
        mgpu_cfg = dict(backend_cfg, type="multi_gpu")
        mgpu_cfg.setdefault("devices", backend_devices)
        mgpu_cfg.setdefault("batch_size", batch_size)
        vae_device = primary_device
        backend = build_tts_backend(
            mgpu_cfg,
            pretrained_path=ckpt_dir,
            log_fn=lambda m: print(f"  {m}", file=sys.stderr),
        )
        if use_prompt_audio and prompt_audio_mode == "static_ref":
            prompt_audio_config = _build_static_ref_prompt_audio_config(
                backend=backend,
                reference_wavs=reference_wavs,
                reference_wav_path=reference_wav_path,
                dataset_ref_key_map=dataset_ref_key_map,
            )
        try:
            results = _run_inference_multi_gpu_backend(
                backend=backend,
                val_ds=val_ds,
                ckpt_dir=ckpt_dir,
                output_dir=output_dir_path,
                num_samples=num_samples,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                max_len=max_len,
                save_audio=save_audio,
                request_audio=request_audio,
                return_audio_arrays=need_audio_arrays,
                save_reference=save_reference,
                save_score=save_score,
                save_audio_max=save_audio_max,
                sample_rate=sample_rate,
                vae_device=vae_device,
                prompt_audio_config=prompt_audio_config,
                verbose=True,
            )
        finally:
            try:
                backend.shutdown()
            except Exception:
                pass
    elif use_batch_inference:
        print(f"[SVS Inference] Using BATCH inference mode", file=sys.stderr)
        results = run_inference_batch(
            model=model,
            val_ds=val_ds,
            output_dir=output_dir_path,
            num_samples=num_samples,
            batch_size=batch_size,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
            save_audio=save_audio,
            return_audio=need_audio_arrays,  # need audio arrays for metrics
            save_reference=save_reference,
            save_score=save_score,
            sample_rate=sample_rate,
            device=primary_device,
            verbose=True,
            prompt_audio_config=prompt_audio_config,
        )
    else:
        print(f"[SVS Inference] Using SEQUENTIAL inference mode", file=sys.stderr)
        # Sequential fallback: use batch inference with batch_size=1
        results = run_inference_batch(
            model=model,
            val_ds=val_ds,
            output_dir=output_dir_path,
            num_samples=num_samples,
            batch_size=1,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            max_len=max_len,
            save_audio=save_audio,
            return_audio=need_audio_arrays,
            save_reference=save_reference,
            save_score=save_score,
            save_audio_max=save_audio_max,
            sample_rate=sample_rate,
            device=primary_device,
            verbose=True,
            prompt_audio_config=prompt_audio_config,
        )
    
    # ---- Unified metrics summary (collects averages from all metrics) ----
    metrics_summary = {
        "ckpt_dir": ckpt_dir,
        "num_samples": len(results) if results else 0,
        "cfg_value": cfg_value,
        "inference_timesteps": inference_timesteps,
    }
    
    # ---- Run unified evaluator over all results ----
    if results:
        evaluator.load_models(devices)
        eval_items = items_from_infer_results(results, val_ds)
        eval_result = evaluator.evaluate(eval_items)
        metrics_summary.update(_apply_infer_legacy_aliases(eval_result.scalars))

    # ---- Save unified metrics summary ----
    output_dir_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir_path / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
    print(f"[Summary] Metrics summary saved to: {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    from vocalrender.training.config import load_yaml_config
    
    args = argbind.parse_args()
    config_file = args.get("config_path")
    if config_file:
        yaml_args = load_yaml_config(config_file)
        infer(**yaml_args)
    else:
        with argbind.scope(args):
            infer()
