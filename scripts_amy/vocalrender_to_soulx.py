#!/usr/bin/env python3
"""Convert VocalRender score JSON to SoulX-Singer metadata.

The two schemas describe the same thing and line up almost one-to-one, which is
the whole reason SoulX-Singer is worth testing: the annotation pipeline in
scripts_amy/annotate_amy.py already produces everything it needs.

  VocalRender                     SoulX-Singer
  ----------------------------    --------------------------------
  pitch      (MIDI, per note)  ->  note_pitch   (MIDI, per note)
  pitch_dur  (seconds)         ->  duration     (seconds)
  word + pitch2word            ->  text         (word repeated per note)
                               ->  phoneme      (ARPAbet, per note)
                               ->  note_type    (1 rest / 2 onset / 3 slur)

``note_type`` is not stored in either schema and has to be derived. Reading the
shipped en_target.json against its own text field gives the convention: 1 for a
rest, 2 when a note starts a new word, 3 when a note continues the previous word
(a melisma). ``pitch2word`` carries exactly that information, so the mapping is
direct rather than heuristic.

``phoneme`` is the one field with no counterpart. SoulX expects, per note, the
word's ARPAbet phones joined by "-" under an ``en_`` prefix -- ``en_HH-UW1`` for
"who". g2p_en produces the phones; the phone_set.json ships the 70 individual
``en_*`` tokens that the joined form is split back into at load time.

Usage:
    python scripts_amy/vocalrender_to_soulx.py \
        --json_file data/amy/annotations_val.json \
        --item_name 05-Back-To-Black__chunk000 \
        --out outputs/soulx/target.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# VocalRender marks breaths/pauses as AP (aspirate) or SP (silence); SoulX uses
# <SP> for both and merges consecutive ones itself.
REST_TOKENS = {"AP", "SP", "<AP>", "<SP>", "", "-"}


def build_g2p():
    """Return a word -> 'en_X-Y-Z' function, or None if g2p_en is missing."""
    try:
        from g2p_en import G2p
    except ImportError:
        return None
    g2p = G2p()

    def to_phoneme(word: str) -> str:
        phones = [p for p in g2p(word) if p and p[0].isalpha()]
        if not phones:
            return "<SP>"
        return "en_" + "-".join(phones)

    return to_phoneme


def convert_item(item: dict, to_phoneme) -> dict:
    words = item["word"]
    pitches = item["pitch"]
    durs = item["pitch_dur"]
    p2w = item["pitch2word"]

    n = len(pitches)
    if not (len(durs) == len(p2w) == n):
        raise ValueError(
            f"{item.get('item_name')}: pitch/pitch_dur/pitch2word disagree "
            f"({n}/{len(durs)}/{len(p2w)})"
        )

    text_toks, phone_toks, type_toks = [], [], []
    prev_word_idx = None

    for i in range(n):
        w = words[p2w[i]] if p2w[i] < len(words) else "AP"
        is_rest = (w in REST_TOKENS) or pitches[i] == 0

        if is_rest:
            text_toks.append("<SP>")
            phone_toks.append("<SP>")
            type_toks.append(1)
            prev_word_idx = None          # a rest breaks any melisma
            continue

        text_toks.append(w)
        phone_toks.append(to_phoneme(w) if to_phoneme else "<SP>")
        # Same source word as the previous sung note => this note continues it.
        type_toks.append(3 if p2w[i] == prev_word_idx else 2)
        prev_word_idx = p2w[i]

    total_ms = int(round(sum(float(d) for d in durs) * 1000))

    return {
        "index": item.get("item_name", "item"),
        "language": "English",
        "time": [0, total_ms],
        "duration": " ".join(f"{float(d):.2f}" for d in durs),
        "text": " ".join(text_toks),
        "phoneme": " ".join(phone_toks),
        "note_pitch": " ".join(str(int(p)) for p in pitches),
        "note_type": " ".join(str(t) for t in type_toks),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json_file", required=True, help="VocalRender annotations JSON")
    ap.add_argument("--item_name", help="Single item to convert (default: all)")
    ap.add_argument("--out", required=True, help="Output SoulX metadata JSON")
    ap.add_argument("--limit", type=int, default=0, help="Convert at most N items")
    args = ap.parse_args()

    data = json.load(open(args.json_file, encoding="utf-8"))
    if isinstance(data, dict):
        data = list(data.values())

    if args.item_name:
        data = [d for d in data if d.get("item_name") == args.item_name]
        if not data:
            print(f"Item not found: {args.item_name}", file=sys.stderr)
            return 1
    if args.limit:
        data = data[: args.limit]

    to_phoneme = build_g2p()
    if to_phoneme is None:
        print("[warn] g2p_en not installed -- phonemes will be <SP> placeholders "
              "and output will be unusable. pip install g2p_en", file=sys.stderr)

    out = [convert_item(item, to_phoneme) for item in data]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print(f"Wrote {len(out)} item(s) -> {out_path}", file=sys.stderr)
    for o in out[:1]:
        n_notes = len(o["note_pitch"].split())
        print(f"  {o['index']}: {n_notes} notes, {o['time'][1]/1000:.1f}s", file=sys.stderr)
        print(f"  text:   {o['text'][:100]}", file=sys.stderr)
        print(f"  phones: {o['phoneme'][:100]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
