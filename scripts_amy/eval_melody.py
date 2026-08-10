#!/usr/bin/env python3
"""
Measure whether a render actually sings the melody it was given.

WER says the words are right; the voice judge says whose voice it is. Neither
says the notes are right. This does: extract F0 from the render with RMVPE,
build the intended pitch contour from the score, DTW-align the two (the model
is free to place notes in time, so a frame-for-frame comparison would punish
expressive timing rather than wrong pitch), and report deviation in semitones.

Octave errors are reported separately from within-octave errors, because
singing the right pitch class an octave off is a different failure from
drifting off the note.

Usage:
    python scripts_amy/eval_melody.py \\
        --json_file examples/smoke_and_honey_v2.json \\
        --renders outputs/v3_lora/v2_01_intro.wav=v2_01_intro ...
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

APPLIO_ROOT = Path(
    r"C:\Users\Rodney Franklin\Development\personal\Amy-RVC\Applio"
)
FRAME_SEC = 0.01

DUR_UNITS = {
    "<NOTE_32>": 0.125, "<NOTE_DOT_32>": 0.1875, "<NOTE_16>": 0.25,
    "<NOTE_DOT_16>": 0.375, "<NOTE_8>": 0.5, "<NOTE_DOT_8>": 0.75,
    "<NOTE_4>": 1.0, "<NOTE_DOT_4>": 1.5, "<NOTE_2>": 2.0,
    "<NOTE_DOT_2>": 3.0, "<NOTE_1>": 4.0, "<NOTE_DOT_1>": 6.0,
}


def score_contour(entry):
    """Piecewise-constant intended pitch contour, 10 ms frames. Rests dropped."""
    beat = 60.0 / entry["bpm"]
    frames = []
    for pitch, note in zip(entry["pitch"], entry["note"]):
        n = max(1, int(round(DUR_UNITS.get(note, 1.0) * beat / FRAME_SEC)))
        if pitch > 0:  # rests carry no pitch to compare against
            frames.extend([float(pitch)] * n)
    return np.asarray(frames, dtype=np.float64)


def render_contour(wav_path, rmvpe):
    y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
    f0 = rmvpe.infer_from_audio(y, thred=0.03)
    voiced = f0 > 0
    if voiced.sum() == 0:
        return np.zeros(0)
    return 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)


def compare(ref, hyp):
    """DTW-align two pitch contours; return semitone error statistics."""
    if len(ref) == 0 or len(hyp) == 0:
        return None

    # DTW on absolute pitch difference. Cheap at 10 ms frames for short phrases,
    # and it absorbs the expressive timing the model is entitled to invent.
    D, wp = librosa.sequence.dtw(X=ref[np.newaxis, :], Y=hyp[np.newaxis, :],
                                 metric="euclidean")
    pairs = np.array(wp)[::-1]
    err = np.array([hyp[j] - ref[i] for i, j in pairs])

    abs_err = np.abs(err)
    # Fold octaves out to separate "wrong octave" from "off the note".
    folded = np.abs((err + 6) % 12 - 6)
    octave_frac = float(np.mean(abs_err - folded > 3))

    return {
        "frames": int(len(pairs)),
        "mean_abs_semitones": round(float(np.mean(abs_err)), 3),
        "median_abs_semitones": round(float(np.median(abs_err)), 3),
        "within_0.5st": round(float(np.mean(abs_err <= 0.5)), 4),
        "within_1st": round(float(np.mean(abs_err <= 1.0)), 4),
        "within_2st": round(float(np.mean(abs_err <= 2.0)), 4),
        "pitch_class_mean_abs": round(float(np.mean(folded)), 3),
        "octave_error_frac": round(octave_frac, 4),
        "ref_span_st": round(float(ref.max() - ref.min()), 2),
        "hyp_span_st": round(float(np.percentile(hyp, 95) - np.percentile(hyp, 5)), 2),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json_file", required=True)
    p.add_argument("--renders", nargs="+", required=True,
                   help="path=item_name pairs")
    p.add_argument("--device", default="cuda")
    p.add_argument("--label", default="")
    return p.parse_args()


def main():
    args = parse_args()
    entries = {e["item_name"]: e
               for e in json.load(open(args.json_file, encoding="utf-8"))}

    sys.path.insert(0, str(APPLIO_ROOT))
    from rvc.lib.predictors.RMVPE import RMVPE0Predictor
    rmvpe = RMVPE0Predictor(
        str(APPLIO_ROOT / "rvc" / "models" / "predictors" / "rmvpe.pt"),
        device=args.device,
    )

    rows = []
    for spec in args.renders:
        path, _, item = spec.partition("=")
        entry = entries.get(item)
        if entry is None:
            print(f"  ! no score entry for {item}")
            continue
        stats = compare(score_contour(entry), render_contour(path, rmvpe))
        if stats is None:
            print(f"  ! no pitch in {path}")
            continue
        stats["item"] = item
        rows.append(stats)
        print(f"{item:16} mean |err| {stats['mean_abs_semitones']:5.2f} st  "
              f"within1 {100*stats['within_1st']:5.1f}%  "
              f"within2 {100*stats['within_2st']:5.1f}%  "
              f"octave-err {100*stats['octave_error_frac']:4.1f}%  "
              f"range score/render {stats['ref_span_st']:.0f}/{stats['hyp_span_st']:.0f} st")

    if rows:
        mean = np.mean([r["mean_abs_semitones"] for r in rows])
        w1 = np.mean([r["within_1st"] for r in rows])
        w2 = np.mean([r["within_2st"] for r in rows])
        tag = f" [{args.label}]" if args.label else ""
        print(f"\nOVERALL{tag}: mean |err| {mean:.2f} semitones | "
              f"within 1st {100*w1:.1f}% | within 2st {100*w2:.1f}%")


if __name__ == "__main__":
    main()
