# Workflow bake-off results

## 2026-08-27: ACE-Step smoke tests on RTX 4070

The 4070 can run ACE-Step 1.5 with LM thinking enabled. The supported 0.6B LM
produced clearly recognizable verse lyrics without an OOM; this passes the
technical gate and advances to human listening.

| Test | Thinking | Result | Harness time | Peak VRAM | Automated vocal check |
|---|---:|---|---:|---:|---|
| `ace_smoke_001` | Off | Audio generated, vocal gate failed | 2.18 s | 9,070 MiB | No intended lyrics recognized |
| `ace_smoke_002_lm06b` | On | Audio generated, technical gate passed | 28.46 s | 7,188 MiB | Verse WER 8.3%; chorus absent, full WER 56.0% |

LM-run audio facts: 30.0 seconds, stereo, 48 kHz, -12.92 LUFS, -0.99 dBTP,
and 2.57 seconds of trailing silence. Whisper large-v3 transcribed:

> Street lights fade into the rain, the last train hums my name. I kept the
> door held by a chain, now morning knows my shame.

The style formatter drifted from the requested retro soul prompt to pop-rock.
Human listening should therefore score vocal quality, lyric delivery, and style
adherence separately.

### Important model provenance note

The server was asked for the installed 1.7B LM. GPU tier detection measured the
4070 at 11.99 GB, rejected 1.7B, downloaded and loaded the supported 0.6B LM,
and logged that fallback. ACE-Step's API response incorrectly retained the
requested `1.7B` label; the server load log is authoritative for this run.

### Next gate

Listen to both clips blind, keep the LM clip only if its vocal naturalness is at
least 3/5, then run three LM seeds with caption rewriting disabled to prevent
the observed genre drift.

## 2026-08-27: three-seed quality round

All three runs used ACE-Step Turbo with the supported 0.6B LM on the RTX 4070,
`thinking=true`, and caption/style rewriting disabled. The API output preserved
the requested retro-soul prompt verbatim.

| Seed | Harness time | Peak VRAM | LUFS | True peak | LRA | Trailing silence | Whisper WER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 101103 | 18.50 s | 6,956 MiB | -16.12 | -1.00 dBTP | 6.5 LU | 1.09 s | 90% |
| 424242 | 18.44 s | 7,134 MiB | -19.46 | -1.00 dBTP | 7.4 LU | 2.05 s | 48% |
| 808317 | 18.40 s | 7,128 MiB | -15.32 | -0.99 dBTP | 3.3 LU | 1.61 s | 40% |

Whisper large-v3 transcripts:

- **101103:** Street lights fading to the red Strain, how's my name? Catching
  now morning dawn
- **424242:** Streetlights fade into the rain The last train haunts my name I
  kept the door held by a chain Now morning dawn let the blue sparks out of
- **808317:** Street lights fade into the rain, the last train, I kept the door
  held by a chain, now morning does my shame, let the blue, I will never borrow
  another face, I will sing

Audio SHA-256:

- **101103:** `57a9d651705de33ed23bf16eb1e2efa93f0a6320a8aeea6ed2d0a5ce32114eac`
- **424242:** `2a63f3da5b3b0ee352d13bd2906d1151bab80ca541cac1283b414d8506a4040f`
- **808317:** `08e5d5f86bc2708dedd2f74a7f081e5e335d499fd1e5127c4facfcb6dd1c8ecc`

Blind-review mapping: Clip A = 808317, Clip B = 101103, Clip C = 424242.
Whisper WER is diagnostic only for singing; the human quality scores remain the
decision gate.

| Clip | Naturalness | Lyric clarity | Musical appeal | Decision |
|---|---:|---:|---:|---|
| A | Pending | Pending | Pending | Pending |
| B | Pending | Pending | Pending | Pending |
| C | Pending | Pending | Pending | Pending |
