"""Fetch the GTSinger English subset.

Two things make the obvious approach slow. The hf CLI serialises metadata
lookups across all 25k matched files, so it stalls for many minutes before any
bytes move; snapshot_download with max_workers parallelises both the HEADs and
the transfers. And the subset is dominated by files we never read:

    Paired_Speech_Group   2065 wavs of spoken lyrics, not singing
    *.TextGrid            5209 phoneme alignment files we do not consume

Excluding those cuts the transfer by roughly a third and the file count -- the
real bottleneck at ~0.5 MB/s across many small files -- by considerably more.
We keep .wav (audio), .json (word/note annotations) and .musicxml (tempo).

A single snapshot_download call also dies on the first httpx.ReadTimeout,
throwing away nothing (the cache resumes) but stopping thousands of files
short. Retrying in a loop is what actually gets to the end over a long
transfer of many small files.
"""

import os
import time

# Default is 10s, which this many concurrent small requests routinely exceeds.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import snapshot_download

ATTEMPTS = 12

for attempt in range(1, ATTEMPTS + 1):
    try:
        path = snapshot_download(
            repo_id="GTSinger/GTSinger",
            repo_type="dataset",
            allow_patterns=[
                "English/**/*.wav",
                "English/**/*.json",
                "English/**/*.musicxml",
            ],
            ignore_patterns=["*Paired_Speech_Group*"],
            local_dir="C:/gts",
            max_workers=16,
        )
        print("done ->", path)
        break
    except Exception as exc:
        print(f"attempt {attempt}/{ATTEMPTS} failed: {type(exc).__name__}: {exc}",
              flush=True)
        if attempt == ATTEMPTS:
            raise
        time.sleep(min(30, 5 * attempt))
