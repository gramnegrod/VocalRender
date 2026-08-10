#!/usr/bin/env python3
"""
Split an annotations.json into train/val holdouts, grouped by song.

Splitting at the segment level would leak: consecutive segments of one song
share melody, lyrics and recording conditions, so a random split lets the
model see the validation material during training and reports a loss that
means nothing. Holding out whole songs avoids that.

The two halves are written as separate files so conf/svs_preprocess_en.yaml
can register them under distinct dataset names -- the trainer selects
train/val by dataset name, not by any per-sample flag.

Usage:
    python scripts_amy/split_train_val.py \\
        --input data/amy/annotations.json \\
        --train_out data/amy/annotations_train.json \\
        --val_out data/amy/annotations_val.json \\
        --separator __ --group_indices 0 --val_frac 0.06
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def group_key(item_name: str, separator: str, indices) -> str:
    parts = item_name.split(separator)
    picked = [parts[i] for i in indices if i < len(parts)]
    return separator.join(picked) if picked else item_name


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--train_out", required=True)
    p.add_argument("--val_out", required=True)
    p.add_argument("--separator", default="#")
    p.add_argument("--group_indices", type=int, nargs="+", default=[0, 1])
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--min_val_groups", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    data = json.load(open(args.input, encoding="utf-8"))

    groups = defaultdict(list)
    for e in data:
        groups[group_key(e["item_name"], args.separator, args.group_indices)].append(e)

    # Deterministic ordering, then take whole groups until the quota is met.
    # Largest-first, because the point is to spend as few *groups* as possible:
    # on a small corpus every held-out song is stylistic variety the model
    # never sees. Smallest-first hits the segment quota only by burning a
    # third of the songs.
    min_groups = max(args.min_val_groups, 0)
    quota = int(len(data) * args.val_frac)

    # Greedy best-fit on whole groups: each step takes the song whose size is
    # closest to what is still needed. Largest-first overshoots badly when a
    # few songs dominate (one 500-segment pick against a 144 quota throws away
    # 11% of training data); smallest-first reaches the quota only by burning
    # a third of the songs, and every held-out song is variety the model never
    # sees. Best-fit lands near target while spending few groups.
    remaining = dict(groups)
    val_groups, n_val = [], 0
    while remaining and (n_val < quota or len(val_groups) < min_groups):
        need = max(quota - n_val, 1)
        name = min(remaining, key=lambda g: (abs(len(remaining[g]) - need), g))
        val_groups.append(name)
        n_val += len(remaining.pop(name))

    val_set = set(val_groups)
    train = [e for e in data
             if group_key(e["item_name"], args.separator, args.group_indices) not in val_set]
    val = [e for e in data
           if group_key(e["item_name"], args.separator, args.group_indices) in val_set]

    for path, rows in ((args.train_out, train), (args.val_out, val)):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"{Path(args.input).name}: {len(data)} segments in {len(groups)} groups")
    print(f"  train {len(train):5}  ({len(groups) - len(val_set)} groups) -> {args.train_out}")
    print(f"  val   {len(val):5}  ({len(val_set)} groups) -> {args.val_out}")
    print(f"  held out: {', '.join(sorted(val_set))}")


if __name__ == "__main__":
    main()
