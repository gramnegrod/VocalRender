#!/usr/bin/env python3
"""Evaluate a batch of ACE-Step smoke-test outputs with one Whisper load."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def normalized_words(text: str) -> list[str]:
    """Normalize lyric or transcript text for a simple reproducible WER."""
    text = re.sub(r"\[[^]]+\]", " ", text)
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def word_error_rate(reference: str, hypothesis: str) -> dict[str, Any]:
    """Return Levenshtein word-error counts without an external dependency."""
    ref = normalized_words(reference)
    hyp = normalized_words(hypothesis)
    distance = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for index in range(len(ref) + 1):
        distance[index][0] = index
    for index in range(len(hyp) + 1):
        distance[0][index] = index
    for row in range(1, len(ref) + 1):
        for column in range(1, len(hyp) + 1):
            distance[row][column] = min(
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
                distance[row - 1][column - 1]
                + (ref[row - 1] != hyp[column - 1]),
            )
    errors = distance[-1][-1]
    return {
        "reference_words": len(ref),
        "hypothesis_words": len(hyp),
        "errors": errors,
        "wer": errors / len(ref) if ref else None,
    }


def ffmpeg_stderr(audio_path: Path, audio_filter: str) -> str:
    """Run an ffmpeg analysis filter and return its diagnostic stream."""
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(audio_path),
            "-af",
            audio_filter,
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr)
    return process.stderr


def loudness(audio_path: Path) -> dict[str, Any]:
    """Measure EBU R128-style loudness using ffmpeg's loudnorm filter."""
    stderr = ffmpeg_stderr(
        audio_path, "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json"
    )
    blocks = re.findall(r"\{\s*\"input_i\".*?\}", stderr, flags=re.DOTALL)
    if not blocks:
        raise RuntimeError("ffmpeg loudnorm output did not contain JSON")
    raw = json.loads(blocks[-1])
    return {
        "integrated_lufs": float(raw["input_i"]),
        "true_peak_dbtp": float(raw["input_tp"]),
        "loudness_range_lu": float(raw["input_lra"]),
        "threshold_lufs": float(raw["input_thresh"]),
    }


def silence(audio_path: Path, duration: float) -> dict[str, Any]:
    """Find >=1-second regions below -45 dB and identify trailing silence."""
    stderr = ffmpeg_stderr(audio_path, "silencedetect=noise=-45dB:d=1")
    starts = [float(value) for value in re.findall(r"silence_start: ([0-9.]+)", stderr)]
    endings = [
        (float(end), float(length))
        for end, length in re.findall(
            r"silence_end: ([0-9.]+) \| silence_duration: ([0-9.]+)", stderr
        )
    ]
    segments = []
    for index, start in enumerate(starts):
        end, length = endings[index] if index < len(endings) else (duration, duration - start)
        segments.append({"start_seconds": start, "end_seconds": end, "duration_seconds": length})
    trailing = 0.0
    if segments and abs(segments[-1]["end_seconds"] - duration) <= 0.05:
        trailing = segments[-1]["duration_seconds"]
    return {
        "threshold_db": -45,
        "minimum_duration_seconds": 1,
        "segments": segments,
        "trailing_silence_seconds": trailing,
    }


def main() -> int:
    """Transcribe and measure each manifest-defined completed run."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    model = WhisperModel(
        args.whisper_model,
        device=args.device,
        compute_type=args.compute_type,
    )
    evaluations = []
    for manifest_path in args.manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dir = args.output_root / manifest["test_id"]
        run = json.loads((run_dir / "run-result.json").read_text(encoding="utf-8"))
        audio_path = run_dir / manifest["output_name"]
        segments, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            vad_filter=True,
        )
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        evaluation = {
            "test_id": manifest["test_id"],
            "evaluated_unix": time.time(),
            "audio_path": str(audio_path.resolve()),
            "audio_sha256": run["audio"]["sha256"],
            "generation": {
                "harness_elapsed_seconds": run["elapsed_seconds"],
                "peak_gpu_memory_used_mib": run["gpu"]["peak_memory_used_mib"],
                "peak_gpu_utilization_percent": run["gpu"]["peak_utilization_percent"],
            },
            "loudness": loudness(audio_path),
            "silence": silence(audio_path, float(run["audio"]["duration_seconds"])),
            "asr": {
                "model": args.whisper_model,
                "language": info.language,
                "language_probability": info.language_probability,
                "transcript": transcript,
                **word_error_rate(manifest["request"]["lyrics"], transcript),
            },
        }
        (run_dir / "evaluation.json").write_text(
            json.dumps(evaluation, indent=2), encoding="utf-8"
        )
        evaluations.append(evaluation)
        print(f"{manifest['test_id']}: {evaluation['asr']['wer']:.1%} WER")

    summary_path = args.output_root / "quality-summary.json"
    summary_path.write_text(json.dumps(evaluations, indent=2), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
