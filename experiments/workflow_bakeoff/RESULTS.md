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
