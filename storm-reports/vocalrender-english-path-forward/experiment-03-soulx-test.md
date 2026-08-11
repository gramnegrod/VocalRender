# Experiment 03 — SoulX-Singer, tested

Run 2026-08-10. The frontier question from `report.html` §08 was "is
SoulX-Singer's English actually good?" It is now answered.

## Setup

Apache-2.0 weights (2.8 GB, `model.pt`, 704M params) from
`Soul-AILab/SoulX-Singer`, run on the RTX 4070 in its own virtualenv
(torch 2.11+cu128, transformers pinned to 4.41.2). Two local changes were
needed and neither touches the model:

- `soulxsinger/utils/audio_utils.py` — torchaudio ≥2.9 routes `load()` through
  TorchCodec, which needs a system FFmpeg. Added a librosa fallback.
- `transformers` had to be pinned to the repo's 4.41.2; newer versions reject
  the positional `LlamaConfig(...)` call in `models/modules/llama.py`.

Data conversion: `scripts_amy/vocalrender_to_soulx.py`. The two schemas line up
almost one-to-one — `pitch`→`note_pitch`, `pitch_dur`→`duration`, `word` +
`pitch2word`→per-note text. Only `note_type` (1 rest / 2 word onset / 3 melisma)
had to be derived, and `pitch2word` encodes exactly that. Phonemes come from
`g2p_en` as ARPAbet.

Evaluation: `scripts_amy/eval_wer_soulx.py`, which imports `normalize`, `wer`
and `reference_words` **from the existing `eval_wer.py`** so the numbers are
comparable by construction — same 15 held-out Amy segments, same 141 reference
words, same faster-whisper large-v3, same normalisation. Prompt clip is
`01-Rehab__chunk003`, drawn from the **training** split so no target lyric
appears in the voice reference.

## Result

| system | run 1 | run 2 | run 3 | mean | sd |
|---|---:|---:|---:|---:|---:|
| VocalRender base | 43.26 | 49.65 | 44.68 | 45.86 | 3.36 |
| VocalRender + LoRA r=32 | 34.04 | 42.55 | 51.06 | 42.55 | 8.51 |
| **SoulX-Singer** | **19.86** | **21.99** | **25.53** | **22.46** | **2.86** |

Welch's t-test, two-sided:

| comparison | difference | p |
|---|---:|---:|
| SoulX vs VocalRender base | **23.40 pts better** | **0.0009** |
| SoulX vs VocalRender LoRA r=32 | **20.09 pts better** | **0.0434** |

**This is the first statistically significant result the project has produced.**
Every prior comparison — base vs LoRA, r=16 vs r=32, embedding-fix vs not — had
p > 0.58. SoulX roughly halves the word error rate on the same items, with a
tighter spread than either VocalRender condition.

## Reading the per-item output

The remaining errors are mostly *not* mispronunciation. Four of the fifteen
segments are single-word errors where the transcriber dropped a leading word
(`"you go back to her…"` → `"go back to her…"`), which is as likely an ASR
onset-clipping artefact as a synthesis failure. The one total failure is
segment 7, whose reference is the 3-word fragment `"u log blow"` — a garbage
reference from the original Whisper pass, not something any model could hit.

Compare against VocalRender's failure mode on the same items: `"i go back
alcua"`, `"they go bagel to her"`, and one empty output.

## What this does and does not establish

**Established.** On this eval, with these items, SoulX-Singer is substantially
more intelligible than VocalRender with or without the English LoRA, and the
gap is far larger than the measurement noise that swamped every previous
comparison.

**Not established.** Whether it *sounds like Amy* — timbre similarity was not
measured here and WER says nothing about it. Whether it holds up on material
outside these 15 segments. Whether the audio is musically good: WER rewards
intelligibility, and the SP-MCQA result in the main report is a reminder that
low WER does not imply overall quality. And the eval remains the weak
instrument described in `experiment-02` — 141 words, ASR-derived references,
five duplicated lyric lines. The effect here is large enough to survive those
caveats; a smaller one would not have been.

**Still worth doing:** the timbre-similarity comparison, and a listening pass
over `outputs/soulx_demo/` before committing to a direction.

## Consequence

The main report ranked "try SoulX-Singer before writing another line of
training code" as action 01, on the reasoning that an afternoon's test could
make the entire finetuning programme unnecessary. That test has now run and it
supports the pivot: a model that already speaks English, needs no training,
takes the existing annotation pipeline's output almost unmodified, and halves
the error rate.
