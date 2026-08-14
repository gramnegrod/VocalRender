#!/usr/bin/env python3
"""Is VocalRender's word failure end-of-sequence drift, or uniform?

The listener report is "starts good, then goes weird", and the ASR transcript
agrees: "And my tears dry" correct, "get some from, wow, my God" not. VoxCPM has
open bugs (#352, #357, #213) describing exactly this shape -- hallucination and
gibberish at the END of audio, specifically for non-Chinese languages -- and a
maintainer in #195 ties it to the stop head converging far faster than the
diffusion loss, which is the signature our own training showed.

If that is what is happening, two things should be true and are testable
without training anything:

  1. Errors concentrate in the back half of each render.
  2. Longer segments are worse than shorter ones.

If both hold, the workaround is trivial (render shorter chunks and concatenate)
and no retraining is needed. If neither holds, the hallucination is something
else and the cheap fix will not work.

Method: render each held-out item, transcribe with word-level timestamps, then
assign every reference word a position in the segment by aligning the
transcript's timeline against the score's own note durations. Errors are
attributed to the half of the segment they fall in.

Usage:
    python scripts_amy/analyze_hallucination.py \
        --lora_dir checkpoints/svs_en_lora_v2/step_0002000 \
        --json_file data/amy/annotations_val.json \
        --render_dir outputs/halluc/vocalrender_v2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def load_eval_wer():
    spec = importlib.util.spec_from_file_location(
        "eval_wer_mod", PROJECT_ROOT / "scripts_amy" / "eval_wer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_missing(entries, lora_dir, json_file, prompt_wav, render_dir, cfg):
    """Render any item that is not already on disk. Renders are deterministic
    only up to sampling, so existing files are reused rather than regenerated."""
    render_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    for i, e in enumerate(entries, 1):
        name = e["item_name"]
        out = render_dir / f"{name}.wav"
        if out.exists():
            continue
        cmd = [py, str(PROJECT_ROOT / "scripts/infer_vocalrender_svs_single.py"),
               "--ckpt_dir", "pretrained_models/VocalRender",
               "--json_file", str(json_file), "--item_name", name,
               "--prompt_audio", str(prompt_wav),
               "--cfg_value", str(cfg), "--output", str(out)]
        if lora_dir:
            # Insert after the --ckpt_dir pair (index 4), NOT inside the
            # --json_file pair, which splitting at index 5 would do.
            cmd[4:4] = ["--lora_dir", str(lora_dir)]
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        status = "ok" if out.exists() else f"FAILED {r.stderr[-200:]}"
        print(f"  [{i}/{len(entries)}] {name}: {status}", file=sys.stderr, flush=True)


def align_errors(ref, hyp):
    """Levenshtein alignment backtrace -> which REFERENCE positions are wrong.

    The earlier version of this split the reference by word count and the
    hypothesis by timestamp and scored the halves independently. That is
    invalid: the two split points do not correspond, so it invented errors --
    items with 0% overall WER were reporting 30-50% in both halves.

    Aligning properly and asking "which reference index did each error land on"
    is the only way to answer where in the segment the model degrades.
    Returns a list of 0/1 flags, one per reference word.
    """
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))

    flags = [0] * n
    i, j = n, m
    while i > 0:
        if j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            flags[i - 1] = int(ref[i - 1] != hyp[j - 1])   # match or substitution
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            flags[i - 1] = 1                               # deletion
            i -= 1
        else:
            j -= 1                                         # insertion
    return flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json_file", default="data/amy/annotations_val.json")
    ap.add_argument("--lora_dir", default="checkpoints/svs_en_lora_v2/step_0002000")
    ap.add_argument("--prompt_wav",
                    default="C:/ai/local-llm/seed-vc/_amy_train_data/01-Rehab__chunk003.wav")
    ap.add_argument("--render_dir", default="outputs/halluc/vocalrender_v2")
    ap.add_argument("--cfg_value", type=float, default=3.0)
    ap.add_argument("--whisper_model", default="large-v3")
    ap.add_argument("--out_json", default="outputs/halluc/positional_stats.json")
    ap.add_argument("--skip_render", action="store_true")
    args = ap.parse_args()

    entries = json.load(open(args.json_file, encoding="utf-8"))
    render_dir = Path(args.render_dir)

    if not args.skip_render:
        print("Rendering missing items...", file=sys.stderr)
        render_missing(entries, args.lora_dir, args.json_file,
                       args.prompt_wav, render_dir, args.cfg_value)

    ev = load_eval_wer()
    from faster_whisper import WhisperModel
    import soundfile as sf
    import numpy as np

    asr = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")

    rows = []
    tot_e1 = tot_n1 = tot_e2 = tot_n2 = 0
    for e in entries:
        name = e["item_name"]
        wav = render_dir / f"{name}.wav"
        if not wav.exists():
            continue
        ref = ev.reference_words(e)
        if not ref:
            continue

        audio, sr = sf.read(str(wav))
        dur = len(audio) / sr

        segs, _ = asr.transcribe(str(wav), language="en", beam_size=5,
                                 word_timestamps=True)
        hyp_words, hyp_times = [], []
        for s in segs:
            for w in (s.words or []):
                tok = ev.normalize(w.word)
                if tok:
                    hyp_words.append(tok[0])
                    hyp_times.append(w.start)

        err, nw = ev.wer(ref, hyp_words)
        flags = align_errors(ref, hyp_words)
        half = max(1, len(flags) // 2)
        pos = (sum(flags[:half]), half,
               sum(flags[half:]), max(1, len(flags) - half))
        row = {"item": name, "duration": round(dur, 2), "flags": flags,
               "wer": round(100.0 * err / nw, 2) if nw else None,
               "errors": err, "ref_words": nw}
        if pos:
            e1, n1, e2, n2 = pos
            row.update({"first_half_wer": round(100.0 * e1 / n1, 2) if n1 else None,
                        "second_half_wer": round(100.0 * e2 / n2, 2) if n2 else None})
            tot_e1 += e1; tot_n1 += n1; tot_e2 += e2; tot_n2 += n2
        rows.append(row)
        print(f"{name[:38]:<38} {dur:5.1f}s  WER {row['wer']:6.1f}%  "
              f"1st {row.get('first_half_wer')}  2nd {row.get('second_half_wer')}",
              file=sys.stderr)

    # Correlation between segment length and error rate.
    durs = [r["duration"] for r in rows if r["wer"] is not None]
    wers = [r["wer"] for r in rows if r["wer"] is not None]
    r_val = p_val = None
    if len(durs) > 2:
        from scipy import stats
        r_val, p_val = stats.pearsonr(durs, wers)

    first = 100.0 * tot_e1 / tot_n1 if tot_n1 else float("nan")
    second = 100.0 * tot_e2 / tot_n2 if tot_n2 else float("nan")

    print("\n" + "=" * 66)
    print(f"items: {len(rows)}")
    print(f"first-half WER  : {first:6.2f}%  ({tot_e1}/{tot_n1})")
    print(f"second-half WER : {second:6.2f}%  ({tot_e2}/{tot_n2})")
    print(f"difference      : {second - first:+6.2f} pts")
    if r_val is not None:
        print(f"duration vs WER : r = {r_val:+.3f}  p = {p_val:.4f}")
    print("=" * 66)
    print("Reading it: a large positive difference plus a positive r supports")
    print("end-of-sequence drift, and chunking shorter should fix it. A flat")
    print("difference and r near zero means the failure is uniform and chunking")
    print("will not help.")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"rows": rows, "first_half_wer": first, "second_half_wer": second,
               "duration_wer_r": r_val, "duration_wer_p": p_val},
              open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
