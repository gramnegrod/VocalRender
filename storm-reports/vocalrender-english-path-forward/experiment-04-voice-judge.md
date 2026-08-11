# Experiment 04 — Voice identity: SoulX vs VocalRender

Run 2026-08-11 with the project's existing gpt-audio voice judge
(`Music-Eval-Ensemble`, `--mode voice -n 3`), which is the same instrument that
produced the 8.0-for-real-Amy figures in `HANDOFF.md` §4.

Experiment 03 showed SoulX-Singer roughly halves WER. WER says nothing about
timbre, so that was explicitly left unmeasured. This closes it.

## Setup

Three items from the held-out set, all rendered from the **same** Amy prompt
clip (`01-Rehab__chunk003`, drawn from the training split). Real Amy recordings
of the same lines included in the batch as controls, per `HANDOFF.md` §9.

Both known traps handled: the dead machine-level `OPENAI_API_KEY` was
overridden from `.env`, and the per-stem cache was cleared before each call.
A third trap surfaced — inherited `PYTHONHOME`/`PYTHONPATH` from a calling venv
makes mee's interpreter load the wrong stdlib and die with "SRE module
mismatch"; those vars have to be stripped.

## Result

| audio | voice score (n=3) | raw |
|---|---:|---|
| **Real Amy** — Back-To-Black c000 | **9.0** | 9 / 9 / 9 |
| **Real Amy** — Back-To-Black c009 | **9.0** | 9 / 9 / 6 |
| VocalRender v2 — c000 | **9.0** | 9 / 9 / 9 |
| VocalRender v2 — c009 | **9.0** | 9 / 9 / 9 |
| VocalRender v2 — c011 | **9.0** | 6 / 9 / 9 |
| SoulX-Singer — c000 | **2.0** | 2 / 2 / 2 |
| SoulX-Singer — c009 | **4.0** | 4 / 3 / 4 |
| SoulX-Singer — c011 | **4.0** | 4 / 4 / 3 |

Judge critiques of SoulX are consistent and specific: *"lacks Amy Winehouse's
characteristic grit, weight, and nuanced vibrato, sounding much thinner."*
Critiques of VocalRender are indistinguishable from those of the real
recordings: *"the same distinctive rich timbre, raspy texture, and soulful
inflections."*

## The two measurements together

| | intelligibility (WER, n=3) | voice identity (n=3 items) |
|---|---:|---:|
| VocalRender v2 | 42.55 % ± 8.51 | **9.0** (= real Amy) |
| SoulX-Singer | **22.46 % ± 2.86** | 2.0–4.0 |
| Real Amy | — | 9.0 |

**A clean trade-off, in opposite directions.** VocalRender reproduces Amy's
voice at a level the judge cannot separate from her actual recordings, and
garbles the words. SoulX gets the words right and sounds like someone else.

Transcripts of the same line make it concrete:

- Reference: *"and my tears dry get home without my guy"*
- SoulX: *"my tears dry again without my guy"*
- VocalRender v2: *"And my tears dry, get some from, wow, my God"*

## Verification performed

The VocalRender 9.0 was checked for prompt leakage before being believed —
if the render had contained the real Amy prompt audio, the score would be an
artefact. It does not: ASR returns the *target* lyrics (Back-To-Black), not the
prompt's (Rehab), and durations match the target score (8.80 s vs 8.72 s
reference), not target-plus-prompt.

## Caveats

- **Three items, one song, one judge.** Enough to establish the direction and
  the size of the gap; not a full evaluation.
- **This contradicts `HANDOFF.md` §4, which recorded VocalRender at 3.0.** Most
  likely explanation: that figure came from *different material* — the
  `smoke_and_honey` original compositions rendered with the v1 LoRA and
  different prompts — not from held-out Amy segments with a real Amy prompt.
  The two are not comparable, and this run is the more relevant one. But the
  discrepancy is unexplained in detail and should not be papered over.
- The judge may be rewarding recording character as well as voice. VocalRender
  copies its prompt's room and production through the VAE; SoulX resamples and
  normalises. Some of the gap could be channel rather than timbre.
- Spread was 0 on most calls, which is suspiciously uniform for a sampled
  judge and suggests it is close to deterministic on clear-cut cases.

## Consequence for the plan

SoulX-Singer is **not** a drop-in replacement, and Experiment 03's framing —
"a model that already does the whole job" — was premature. It does half the job
very well.

The two halves are complementary, and the pairing the first research pass
recommended now has direct measured support: **generate the performance with
the model that gets the words right, then apply per-artist conversion for
timbre.** The community-validated route is RVC trained on 10–30 minutes of
clean isolated vocal; there are ~24 minutes of Amy stems on disk, which lands
squarely in that range. `SoulX-Singer-SVC` (`model-svc.pt`, 2.66 GB, same repo)
is an untested alternative for the conversion stage.

Notably, VocalRender's timbre result also means the LoRA work was not wasted —
whatever else is wrong with it, the prompt-conditioning path reproduces this
voice extremely well.
