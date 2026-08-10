#!/usr/bin/env python3
"""
Convert the GTSinger English subset into VocalRender's annotation schema.

GTSinger ships one JSON per audio segment, already carrying human-checked
word boundaries and per-word MIDI notes -- everything the trainer needs except
the tempo, which lives in the sibling .musicxml. This is the opposite
situation to the Amy stems: no inference required, just a field remap.

    GTSinger                     VocalRender
    --------                     -----------
    word / <SP> / <AP>       ->  word (SP / AP)
    note (MIDI per note)     ->  pitch
    note_dur (seconds)       ->  note (quantized to note-duration tokens)
    implicit word->note map  ->  pitch2word
    musicxml <per-minute>    ->  bpm

Usage:
    python scripts_amy/convert_gtsinger.py --root C:/gts/English \\
        --out data/gtsinger_en/annotations.json --audio_root C:/gts
"""

import argparse
import json
import re
from pathlib import Path

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

PER_MINUTE_RE = re.compile(r"<per-minute>\s*([0-9.]+)\s*</per-minute>")
WORD_RE = re.compile(r"[^a-z']+")


def quantize_duration(dur_sec: float, bpm: float) -> str:
    units = dur_sec / (60.0 / bpm)
    return min(DUR_UNITS, key=lambda tok: abs(DUR_UNITS[tok] - units))


def read_tempo(json_path: Path, default: float) -> float:
    """Tempo from the sibling musicxml, else the segment's own note grid.

    GTSinger quantizes note_dur onto a fixed grid, so when the musicxml is
    missing the smallest observed duration is a reliable proxy for a 16th note.
    """
    xml = json_path.with_suffix(".musicxml")
    if xml.is_file():
        m = PER_MINUTE_RE.search(xml.read_text(encoding="utf-8", errors="ignore"))
        if m:
            try:
                bpm = float(m.group(1))
                if 30 <= bpm <= 260:
                    return bpm
            except ValueError:
                pass
    return default


def infer_tempo_from_grid(entries, default: float) -> float:
    durs = [d for w in entries for d in w.get("note_dur", []) if d > 0.05]
    if not durs:
        return default
    unit = min(durs)  # smallest grid step, assumed to be a sixteenth
    bpm = 60.0 / (4.0 * unit)
    while bpm > 200:
        bpm /= 2
    while 0 < bpm < 45:
        bpm *= 2
    return bpm if 30 <= bpm <= 260 else default


def clean_word(raw: str) -> str:
    if raw == "<SP>":
        return "SP"
    if raw == "<AP>":
        return "AP"
    return WORD_RE.sub("", raw.lower().strip())


def convert_one(jpath: Path, root: Path, audio_root: Path, args):
    # Paired_Speech_Group holds spoken readings of the lyrics, not singing.
    # They carry no note fields and have no place in an SVS training set.
    if any(part.startswith("Paired_Speech") for part in jpath.parts):
        return None, "spoken"

    wav = jpath.with_suffix(".wav")
    if not wav.is_file():
        return None, "no-wav"

    try:
        data = json.load(open(jpath, encoding="utf-8"))
    except Exception:
        return None, "bad-json"
    if not data:
        return None, "empty"

    bpm = read_tempo(jpath, 0.0) or infer_tempo_from_grid(data, args.bpm_default)

    words, pitches, notes, p2w = [], [], [], []
    word_durs, pitch_durs = [], []
    sung = 0

    for w in data:
        text = clean_word(w.get("word", ""))
        if not text:
            continue
        ns = w.get("note", [])
        ds = w.get("note_dur", [])
        if not ns or len(ns) != len(ds):
            return None, "note-mismatch"

        wi = len(words)
        words.append(text)
        word_durs.append(round(float(w.get("end_time", 0)) - float(w.get("start_time", 0)), 3))
        if text not in ("SP", "AP"):
            sung += 1

        for pitch, dur in zip(ns, ds):
            pitch = int(round(float(pitch)))
            if text in ("SP", "AP"):
                pitch = 0
            elif not (36 <= pitch <= 96):
                pitch = max(36, min(96, pitch))
            pitches.append(pitch)
            notes.append(quantize_duration(float(dur), bpm))
            pitch_durs.append(round(float(dur), 3))
            p2w.append(wi)

    if sung < args.min_words:
        return None, "too-short"

    total = float(data[-1].get("end_time", 0)) - float(data[0].get("start_time", 0))
    if not (args.min_seconds <= total <= args.max_seconds):
        return None, "duration"

    # English/EN-Alto-1/Breathy/all is found/Breathy_Group/0000
    #   -> EN-Alto-1#all is found#Breathy#Breathy_Group#0000
    rel = jpath.relative_to(root).with_suffix("")
    parts = rel.parts
    if len(parts) >= 5:
        singer, technique, song, group, idx = parts[0], parts[1], parts[2], parts[3], parts[4]
        item_name = f"{singer}#{song}#{technique}#{group}#{idx}"
    else:
        item_name = "#".join(parts)

    return {
        "item_name": item_name,
        "wav_fn": wav.relative_to(audio_root).as_posix(),
        "word": words,
        "word_dur": word_durs,
        "pitch": pitches,
        "note": notes,
        "pitch_dur": pitch_durs,
        "pitch2word": p2w,
        "bpm": int(round(bpm)),
    }, None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="C:/gts/English")
    p.add_argument("--audio_root", default="C:/gts")
    p.add_argument("--out", required=True)
    p.add_argument("--bpm_default", type=float, default=90.0)
    p.add_argument("--min_words", type=int, default=3)
    p.add_argument("--min_seconds", type=float, default=1.5)
    p.add_argument("--max_seconds", type=float, default=25.0)
    return p.parse_args()


def main():
    args = parse_args()
    root, audio_root = Path(args.root), Path(args.audio_root)
    jsons = sorted(root.rglob("*.json"))
    print(f"[gtsinger] {len(jsons)} segment JSONs under {root}")

    entries = []
    reasons = {}
    for j in jsons:
        entry, why = convert_one(j, root, audio_root, args)
        if entry is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        entries.append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=1)

    bpms = [e["bpm"] for e in entries]
    print(f"[gtsinger] kept {len(entries)}  skipped {dict(reasons)}")
    if bpms:
        print(f"[gtsinger] bpm range {min(bpms)}-{max(bpms)}")
    print(f"[gtsinger] -> {out}")


if __name__ == "__main__":
    main()
