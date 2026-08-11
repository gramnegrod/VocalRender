#!/usr/bin/env python3
"""WER for SoulX-Singer on the same held-out set eval_wer.py uses.

Deliberately reuses ``normalize``, ``wer`` and ``reference_words`` from
scripts_amy/eval_wer.py rather than reimplementing them, so the number this
prints is comparable to the VocalRender numbers by construction. Same items,
same references, same normalisation, same ASR.

The comparison is still only as good as the eval itself, which
experiment-02-wer-variance.md showed is very noisy: 141 reference words gives a
binomial standard error of roughly 4 points per run before clustering, so a
single run of this script settles nothing on its own. Use --runs.

SoulX generation is a subprocess call into the SoulX-Singer checkout, because
it needs its own virtualenv (torch 2.11 + transformers 4.41).

Usage:
    python scripts_amy/eval_wer_soulx.py \
        --soulx_dir ../SoulX-Singer \
        --json_file data/amy/annotations_val.json \
        --prompt_json data/amy/annotations_train.json \
        --prompt_item 01-Rehab__chunk003 \
        --audio_root "C:/ai/local-llm/seed-vc/_amy_train_data" \
        --out_dir outputs/wer_soulx --runs 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def load_eval_wer():
    """Import eval_wer.py for its scoring helpers (not its inference path)."""
    spec = importlib.util.spec_from_file_location(
        "eval_wer_mod", PROJECT_ROOT / "scripts_amy" / "eval_wer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # eval_wer imports torch and the vocalrender package at module scope only
    # inside main(), so loading it here is cheap and side-effect free.
    spec.loader.exec_module(mod)
    return mod


def convert(soulx_dir: Path, py: Path, json_file: Path, item: str, out: Path) -> None:
    subprocess.run(
        [str(py), str(soulx_dir / "vocalrender_to_soulx.py"),
         "--json_file", str(json_file), "--item_name", item, "--out", str(out)],
        check=True, capture_output=True, cwd=str(soulx_dir),
    )


def generate(soulx_dir: Path, py: Path, prompt_wav: Path, prompt_meta: Path,
             target_meta: Path, save_dir: Path) -> Path:
    env_pythonpath = str(soulx_dir)
    cmd = [
        str(py), "-m", "cli.inference",
        "--device", "cuda",
        "--model_path", "pretrained_models/SoulX-Singer/model.pt",
        "--config", "soulxsinger/config/soulxsinger.yaml",
        "--prompt_wav_path", str(prompt_wav),
        "--prompt_metadata_path", str(prompt_meta),
        "--target_metadata_path", str(target_meta),
        "--phoneset_path", "soulxsinger/utils/phoneme/phone_set.json",
        "--save_dir", str(save_dir),
        "--auto_shift", "--pitch_shift", "0", "--fp16",
    ]
    import os
    env = dict(os.environ, PYTHONPATH=env_pythonpath, CUDA_VISIBLE_DEVICES="0")
    r = subprocess.run(cmd, cwd=str(soulx_dir), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"SoulX inference failed:\n{r.stderr[-1500:]}")
    return save_dir / "generated.wav"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soulx_dir", required=True)
    ap.add_argument("--json_file", required=True)
    ap.add_argument("--prompt_json", required=True,
                    help="Annotations containing the prompt item (use TRAIN to avoid leakage)")
    ap.add_argument("--prompt_item", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--out_dir", default="outputs/wer_soulx")
    ap.add_argument("--whisper_model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    soulx_dir = Path(args.soulx_dir).resolve()
    py = soulx_dir / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = soulx_dir / ".venv" / "bin" / "python"
    if not py.exists():
        print(f"No SoulX venv python under {soulx_dir}", file=sys.stderr)
        return 1

    ev = load_eval_wer()
    entries = json.load(open(args.json_file, encoding="utf-8"))
    if args.limit:
        entries = entries[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work = soulx_dir / "work" / "eval"
    work.mkdir(parents=True, exist_ok=True)

    # Prompt: converted once, reused for every item. Drawn from the training
    # split so the target lyrics are never present in the voice reference.
    prompt_meta = work / "prompt.json"
    convert(soulx_dir, py, Path(args.prompt_json).resolve(), args.prompt_item, prompt_meta)
    prompt_wav = Path(args.audio_root) / f"{args.prompt_item}.wav"
    if not prompt_wav.exists():
        print(f"Prompt wav not found: {prompt_wav}", file=sys.stderr)
        return 1

    from faster_whisper import WhisperModel
    asr = WhisperModel(args.whisper_model, device=args.device, compute_type="float16")

    run_wers = []
    for run in range(1, args.runs + 1):
        tot_err = tot_words = 0
        scored = 0
        for i, entry in enumerate(entries, 1):
            name = entry["item_name"]
            # ev.normalize / ev.reference_words return word LISTS, not strings.
            ref = ev.reference_words(entry)
            if not ref:
                continue
            tgt_meta = work / f"{name}.json"
            convert(soulx_dir, py, Path(args.json_file).resolve(), name, tgt_meta)
            save_dir = work / f"gen_run{run}" / name
            try:
                wav = generate(soulx_dir, py, prompt_wav, prompt_meta, tgt_meta, save_dir)
            except RuntimeError as e:
                print(f"[{i}/{len(entries)}] {name}: GEN FAILED {str(e)[:120]}", file=sys.stderr)
                continue

            segs, _ = asr.transcribe(str(wav), language="en", beam_size=5)
            hyp = ev.normalize(" ".join(s.text for s in segs))
            err, nw = ev.wer(ref, hyp)
            tot_err += err
            tot_words += nw
            scored += 1
            pct = 100.0 * err / nw if nw else 0.0
            print(f"[{i}/{len(entries)}] WER {err}/{nw} ({pct:5.1f}%)  "
                  f"ref: {' '.join(ref)[:44]}  |  hyp: {' '.join(hyp)[:44]}",
                  file=sys.stderr)

        run_wer = 100.0 * tot_err / tot_words if tot_words else float("nan")
        run_wers.append(run_wer)
        print(f"\n[run {run}] {scored} scored | {tot_err}/{tot_words} words wrong "
              f"| WER {run_wer:.2f}%\n", file=sys.stderr)

    print("=" * 62)
    print(f"SoulX-Singer  runs={args.runs}  " + " ".join(f"{w:.2f}" for w in run_wers))
    if len(run_wers) > 1:
        print(f"mean {st.mean(run_wers):.2f}%  sd {st.stdev(run_wers):.2f}")
    else:
        print(f"mean {run_wers[0]:.2f}%  (single run -- not interpretable alone)")
    print("Compare (same items, same references, measured 2026-08-10):")
    print("  VocalRender base      45.86% +- 3.36")
    print("  VocalRender LoRA r=32 42.55% +- 8.51")
    print("=" * 62)

    with open(out_dir / "soulx_wer.json", "w", encoding="utf-8") as f:
        json.dump({"runs": run_wers, "prompt_item": args.prompt_item}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
