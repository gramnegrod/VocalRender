"""Fold double-time tempo readings back into a normal band and re-quantize.

librosa's beat tracker frequently locks onto the half-note pulse of a slow
soul track, reporting 144 for a song that is felt at 72. The note tokens were
quantized against that inflated tempo, so both must move together -- which is
exact here because pitch_dur holds the real duration in seconds.
"""
import json
import sys
from pathlib import Path

DUR_UNITS = {
    "<NOTE_32>": 0.125, "<NOTE_DOT_32>": 0.1875, "<NOTE_16>": 0.25,
    "<NOTE_DOT_16>": 0.375, "<NOTE_8>": 0.5, "<NOTE_DOT_8>": 0.75,
    "<NOTE_4>": 1.0, "<NOTE_DOT_4>": 1.5, "<NOTE_2>": 2.0,
    "<NOTE_DOT_2>": 3.0, "<NOTE_1>": 4.0, "<NOTE_DOT_1>": 6.0,
}
BPM_MAX = 150

path = Path(sys.argv[1])
data = json.load(open(path, encoding="utf-8"))
changed = 0
for e in data:
    bpm = e["bpm"]
    while bpm > BPM_MAX:
        bpm /= 2
    bpm = int(round(bpm))
    if bpm == e["bpm"]:
        continue
    e["bpm"] = bpm
    beat = 60.0 / bpm
    e["note"] = [
        min(DUR_UNITS, key=lambda t: abs(DUR_UNITS[t] - d / beat))
        for d in e["pitch_dur"]
    ]
    changed += 1

json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"re-tempoed {changed}/{len(data)} entries -> {path}")
print("bpm range now:", min(e["bpm"] for e in data), "-", max(e["bpm"] for e in data))
