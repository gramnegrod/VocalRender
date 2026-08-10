#!/usr/bin/env python3
"""
Generate a song with MiniMax music-3.0.

Used here for one decisive comparison first: give MiniMax the same lyrics
VocalRender sang, ask only for Amy Winehouse style, and score both through the
same calibrated gpt-audio voice judge. VocalRender currently sits at 3/10
against real Amy's 9/10. If MiniMax lands high, its output is worth distilling
into training data for the LoRA; if it lands at 4, the whole idea is weak and
we should know that before spending tokens on a corpus.

MiniMax has no voice-cloning or timbre-reference parameter -- style comes from
the text prompt alone -- so this also tests how far a text prompt gets on an
artist the model may simply know.

Credentials live in ~/.minimax-key.txt (the sk- line).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

KEY_FILE = Path(r"C:/Users/Rodney Franklin/.minimax-key.txt")
ENDPOINT = "https://api.minimax.io/v1/music_generation"


def load_key() -> str:
    for line in KEY_FILE.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("sk-"):
            return line
        if "=" in line and line.split("=", 1)[1].strip().startswith("sk-"):
            return line.split("=", 1)[1].strip()
    raise SystemExit("No sk- key found in the key file")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True)
    p.add_argument("--lyrics", required=True, help="Path to a lyrics text file")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="music-3.0")
    p.add_argument("--timeout", type=int, default=300)
    return p.parse_args()


def main():
    args = parse_args()
    key = load_key()
    lyrics = Path(args.lyrics).read_text(encoding="utf-8")

    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "lyrics": lyrics,
        "audio_setting": {"sample_rate": 44100, "bitrate": 256000, "format": "mp3"},
        "output_format": "url",
    }

    print(f"[minimax] POST {ENDPOINT}  model={args.model}", flush=True)
    r = requests.post(
        ENDPOINT,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        json=payload,
        timeout=args.timeout,
    )
    print(f"[minimax] HTTP {r.status_code}", flush=True)

    try:
        data = r.json()
    except Exception:
        print("[minimax] non-JSON response:", r.text[:800])
        return 1

    # Log the envelope (minus any audio blob) so failures are diagnosable.
    skeleton = json.loads(json.dumps(data))
    audio_field = None
    if isinstance(skeleton.get("data"), dict):
        audio_field = skeleton["data"].get("audio")
        if audio_field and len(str(audio_field)) > 120:
            skeleton["data"]["audio"] = f"<{len(str(audio_field))} chars>"
    print("[minimax] response:", json.dumps(skeleton)[:900], flush=True)

    base = (data.get("base_resp") or {})
    if base.get("status_code") not in (0, None):
        print(f"[minimax] API error {base.get('status_code')}: {base.get('status_msg')}")
        return 1

    audio = (data.get("data") or {}).get("audio")
    if not audio:
        print("[minimax] no audio in response")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if str(audio).startswith("http"):
        print(f"[minimax] downloading {str(audio)[:80]}...", flush=True)
        blob = requests.get(audio, timeout=args.timeout).content
    else:
        # Some responses return raw hex rather than a URL.
        blob = bytes.fromhex(audio)

    out.write_bytes(blob)
    print(f"[minimax] wrote {out}  ({len(blob)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
