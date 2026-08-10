#!/usr/bin/env python3
"""
Build VocalRender score annotations for a folder of solo-vocal stems.

The released checkpoints were trained only on Chinese singing, so English
diction is out of distribution. To finetune on English we need the same
score schema the trainer consumes -- word / pitch / note / pitch2word / bpm --
but raw vocal stems carry none of it. This script derives it:

    faster-whisper  ->  word-level timings
    RMVPE           ->  F0 contour (10 ms hop, purpose-built for singing)

Each lyric word becomes one ``word`` slot. The F0 inside that word's span is
segmented into runs of stable semitone, so a word held across several pitches
yields several (pitch, note) pairs -- melisma falls out of the segmentation
rather than needing separate handling.

Silences between words become AP (breath) or SP (silence) slots, chosen by
whether the gap actually carries breath noise.

Usage:
    python scripts_amy/annotate_amy.py \\
        --dataset_dir /c/ai/local-llm/seed-vc/_amy_train_data \\
        --out data/amy/annotations.json
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np

# RMVPE lives in the Applio checkout; it is self-contained (torch + librosa).
APPLIO_ROOT = Path(
    r"C:\Users\Rodney Franklin\Development\personal\Amy-RVC\Applio"
)

FRAME_SEC = 0.01  # RMVPE hop: 160 samples @ 16 kHz

# Note-duration tokens, in quarter-note units. Mirrors
# vocalrender.model.svs_utils.get_svs_token_maps().
DUR_UNITS = {
    "<NOTE_32>": 0.125,
    "<NOTE_DOT_32>": 0.1875,
    "<NOTE_16>": 0.25,
    "<NOTE_DOT_16>": 0.375,
    "<NOTE_8>": 0.5,
    "<NOTE_DOT_8>": 0.75,
    "<NOTE_4>": 1.0,
    "<NOTE_DOT_4>": 1.5,
    "<NOTE_2>": 2.0,
    "<NOTE_DOT_2>": 3.0,
    "<NOTE_1>": 4.0,
    "<NOTE_DOT_1>": 6.0,
}

WORD_RE = re.compile(r"[^a-z']+")


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def quantize_duration(dur_sec: float, bpm: int) -> str:
    """Snap a duration in seconds to the nearest note-duration token."""
    units = dur_sec / (60.0 / bpm)
    return min(DUR_UNITS, key=lambda tok: abs(DUR_UNITS[tok] - units))


def _merge_equal_neighbours(runs):
    """Collapse adjacent runs that landed on the same pitch."""
    out = []
    for pitch, n in runs:
        if out and out[-1][0] == pitch:
            out[-1][1] += n
        else:
            out.append([pitch, n])
    return out


def segment_pitches(midi, min_frames, max_notes, semitone_tol=1, glide_frames=20):
    """Split a per-frame MIDI contour into runs of stable semitone.

    Returns a list of (pitch, n_frames). Sung notes are approached by scoop and
    left by fall, and vibrato swings a semitone either side of centre, so naive
    run-length encoding of the rounded contour reports every transient as its
    own note. Two passes suppress that: short runs are absorbed into the
    neighbour nearest in pitch, then near-unison neighbours (within
    ``semitone_tol``) are folded into whichever side is longer.
    """
    if len(midi) == 0:
        return []

    rounded = np.round(midi).astype(int)

    runs = []
    start = 0
    for i in range(1, len(rounded) + 1):
        if i == len(rounded) or rounded[i] != rounded[start]:
            runs.append([int(rounded[start]), i - start])
            start = i

    # Pass 1: absorb sub-threshold runs into the closest-pitched neighbour.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (pitch, n) in enumerate(runs):
            if n >= min_frames:
                continue
            prev_d = abs(runs[i - 1][0] - pitch) if i > 0 else 999
            next_d = abs(runs[i + 1][0] - pitch) if i + 1 < len(runs) else 999
            if prev_d == next_d:  # tie -> give the frames to the longer note
                target = i - 1 if runs[i - 1][1] >= runs[i + 1][1] else i + 1
            else:
                target = i - 1 if prev_d < next_d else i + 1
            runs[target][1] += n
            runs.pop(i)
            changed = True
            break

    runs = _merge_equal_neighbours(runs)

    # Pass 2: fold near-unison neighbours together. A genuine melodic move is
    # either a wider interval or a long sustain; anything else is vibrato or a
    # glide caught mid-flight.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i in range(len(runs) - 1):
            (p0, n0), (p1, n1) = runs[i], runs[i + 1]
            if abs(p0 - p1) > semitone_tol:
                continue
            if min(n0, n1) > glide_frames:
                continue
            keep, drop = (i, i + 1) if n0 >= n1 else (i + 1, i)
            runs[keep][1] += runs[drop][1]
            runs.pop(drop)
            changed = True
            break

    runs = _merge_equal_neighbours(runs)

    # Cap melisma width: keep the longest runs, preserving time order.
    if len(runs) > max_notes:
        keep = sorted(sorted(range(len(runs)), key=lambda i: -runs[i][1])[:max_notes])
        merged, spare = [], 0
        for i, run in enumerate(runs):
            if i in keep:
                merged.append([run[0], run[1] + spare])
                spare = 0
            else:
                spare += run[1]
        if spare and merged:
            merged[-1][1] += spare
        runs = merged

    return [(p, n) for p, n in runs if n > 0]


def clean_word(raw: str) -> str:
    """Lowercase and strip punctuation; '' means the token carries no lyric."""
    return WORD_RE.sub("", raw.lower().strip())


def gap_is_breath(y16, t0, t1, floor_db) -> bool:
    """True when a silent gap still carries audible breath (-> AP, not SP)."""
    i0, i1 = int(t0 * 16000), int(t1 * 16000)
    chunk = y16[i0:i1]
    if len(chunk) < 160:
        return False
    rms = float(np.sqrt(np.mean(chunk ** 2)))
    if rms <= 0:
        return False
    return 20 * np.log10(rms) > floor_db


# ---------------------------------------------------------------------------
# Per-file annotation
# ---------------------------------------------------------------------------

def annotate_file(wav_path, words, f0, y16, bpm, args):
    """Build one annotation entry, or None if the segment is unusable."""
    midi = np.zeros_like(f0)
    voiced = f0 > 0
    midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)

    # Median-smooth the voiced contour so vibrato does not fragment runs.
    if voiced.sum() > 5:
        smoothed = np.copy(midi)
        idx = np.flatnonzero(voiced)
        vals = midi[idx]
        k = 5
        pad = k // 2
        padded = np.pad(vals, pad, mode="edge")
        smoothed[idx] = np.array(
            [np.median(padded[i:i + k]) for i in range(len(vals))]
        )
        midi = smoothed

    word_slots, pitches, notes, p2w = [], [], [], []
    word_durs, pitch_durs = [], []
    total_frames = len(f0)
    audio_end = total_frames * FRAME_SEC

    def add_rest(t0, t1):
        """Append an AP/SP slot for a silence of [t0, t1)."""
        dur = t1 - t0
        if dur < args.min_rest:
            return
        label = "AP" if gap_is_breath(y16, t0, t1, args.breath_floor_db) else "SP"
        word_slots.append(label)
        word_durs.append(round(dur, 3))
        pitches.append(0)
        notes.append(quantize_duration(dur, bpm))
        pitch_durs.append(round(dur, 3))
        p2w.append(len(word_slots) - 1)

    prev_end = 0.0
    last_pitch = 60
    sung = 0

    for w in words:
        text = clean_word(w["word"])
        if not text:
            continue
        start = max(0.0, float(w["start"]))
        end = min(audio_end, float(w["end"]))
        if end - start < args.min_word:
            continue

        add_rest(prev_end, start)

        i0 = int(round(start / FRAME_SEC))
        i1 = max(i0 + 1, int(round(end / FRAME_SEC)))
        span_voiced = voiced[i0:i1]
        runs = []
        if span_voiced.sum() >= args.min_note_frames:
            runs = segment_pitches(
                midi[i0:i1][span_voiced], args.min_note_frames,
                args.max_notes_per_word, args.semitone_tol, args.glide_frames,
            )

        if not runs:
            # Unvoiced word (plosive-only, or whisper timing slop): carry the
            # previous pitch rather than dropping the lyric.
            runs = [(last_pitch, i1 - i0)]
        else:
            sung += 1

        # Runs were measured over voiced frames only; rescale to the real span.
        run_total = sum(n for _, n in runs)
        span_sec = end - start
        word_slots.append(text)
        word_durs.append(round(span_sec, 3))
        wi = len(word_slots) - 1

        for pitch, n in runs:
            pitch = int(np.clip(pitch, 36, 96))
            dur = span_sec * (n / run_total)
            pitches.append(pitch)
            notes.append(quantize_duration(dur, bpm))
            pitch_durs.append(round(dur, 3))
            p2w.append(wi)
            last_pitch = pitch

        prev_end = end

    if prev_end < audio_end:
        add_rest(prev_end, audio_end)

    if sung < args.min_words or not word_slots:
        return None

    return {
        "item_name": wav_path.stem,
        "wav_fn": wav_path.name,
        "word": word_slots,
        "word_dur": word_durs,
        "pitch": pitches,
        "note": notes,
        "pitch_dur": pitch_durs,
        "pitch2word": p2w,
        "bpm": int(bpm),
    }


# ---------------------------------------------------------------------------
# BPM
# ---------------------------------------------------------------------------

def song_bpms(files, args):
    """Estimate one BPM per song from all of that song's stems concatenated.

    Per-chunk tempo tracking on an isolated vocal is far too noisy; pooling a
    whole song's stems gives the beat tracker enough onsets to lock on.
    """
    by_song = defaultdict(list)
    for f in files:
        by_song[f.stem.split("__")[0]].append(f)

    bpms = {}
    for song, chunks in sorted(by_song.items()):
        ys = []
        for c in sorted(chunks)[: args.bpm_chunks]:
            y, _ = librosa.load(str(c), sr=22050, mono=True)
            ys.append(y)
        y = np.concatenate(ys)
        tempo, _ = librosa.beat.beat_track(y=y, sr=22050)
        tempo = float(np.atleast_1d(tempo)[0])
        while tempo > args.bpm_max:
            tempo /= 2
        while 0 < tempo < args.bpm_min:
            tempo *= 2
        if not (args.bpm_min <= tempo <= args.bpm_max):
            tempo = args.bpm_default
        bpms[song] = int(round(tempo))
        print(f"  bpm {song:44} {bpms[song]}", flush=True)
    return bpms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--whisper_model", default="large-v3")
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute_type", default="float16")
    p.add_argument("--min_note_frames", type=int, default=12,
                   help="Minimum frames (10 ms each) for a note to survive")
    p.add_argument("--max_notes_per_word", type=int, default=3)
    p.add_argument("--semitone_tol", type=int, default=1,
                   help="Neighbouring runs this close in pitch may be folded")
    p.add_argument("--glide_frames", type=int, default=20,
                   help="Near-unison runs shorter than this are glide/vibrato")
    p.add_argument("--min_word", type=float, default=0.05,
                   help="Drop whisper words shorter than this (seconds)")
    p.add_argument("--min_rest", type=float, default=0.18,
                   help="Gaps shorter than this get absorbed, not marked AP/SP")
    p.add_argument("--breath_floor_db", type=float, default=-45.0)
    p.add_argument("--min_words", type=int, default=3,
                   help="Skip segments with fewer sung words than this")
    p.add_argument("--bpm_chunks", type=int, default=8,
                   help="Stems per song pooled for tempo estimation")
    p.add_argument("--bpm_min", type=float, default=55)
    p.add_argument("--bpm_max", type=float, default=150)
    p.add_argument("--bpm_default", type=int, default=90)
    p.add_argument("--limit", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.dataset_dir)
    files = sorted(root.glob("*.wav"))
    if args.limit > 0:
        files = files[: args.limit]
    print(f"[annotate] {len(files)} stems in {root}", flush=True)

    print("[annotate] estimating tempo per song", flush=True)
    bpms = song_bpms(files, args)

    sys.path.insert(0, str(APPLIO_ROOT))
    from rvc.lib.predictors.RMVPE import RMVPE0Predictor

    rmvpe_path = APPLIO_ROOT / "rvc" / "models" / "predictors" / "rmvpe.pt"
    print(f"[annotate] loading RMVPE from {rmvpe_path}", flush=True)
    rmvpe = RMVPE0Predictor(str(rmvpe_path), device=args.device)

    from faster_whisper import WhisperModel
    print(f"[annotate] loading whisper {args.whisper_model}", flush=True)
    whisper = WhisperModel(args.whisper_model, device=args.device,
                           compute_type=args.compute_type)

    entries, skipped = [], 0
    for n, wav in enumerate(files, 1):
        song = wav.stem.split("__")[0]
        bpm = bpms.get(song, args.bpm_default)

        y16, _ = librosa.load(str(wav), sr=16000, mono=True)
        f0 = rmvpe.infer_from_audio(y16, thred=0.03)

        segs, _ = whisper.transcribe(str(wav), language="en",
                                     word_timestamps=True, beam_size=5,
                                     vad_filter=True)
        words = [{"word": w.word, "start": w.start, "end": w.end}
                 for s in segs for w in (s.words or [])]

        entry = annotate_file(wav, words, f0, y16, bpm, args)
        if entry is None:
            skipped += 1
            print(f"[{n}/{len(files)}] SKIP {wav.name}", flush=True)
            continue
        entries.append(entry)
        lyric = " ".join(w for w in entry["word"] if w not in ("AP", "SP"))
        print(f"[{n}/{len(files)}] {wav.name}  bpm={bpm} "
              f"notes={len(entry['pitch'])}  {lyric[:60]}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)
    print(f"[annotate] wrote {len(entries)} entries ({skipped} skipped) -> {out}")


if __name__ == "__main__":
    main()
