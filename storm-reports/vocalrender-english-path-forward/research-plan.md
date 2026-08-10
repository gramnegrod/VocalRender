# Research plan — How to actually make VocalRender sing good English

Date: 2026-08-10. Mode: **deep**. Follow-up to `prompt-conditioned-svs-english`.

## Research question

For a solo practitioner on one RTX 4070 12 GB: what is the intervention that
actually produces good English from VocalRender (VoxCPM2-based, prompt-
conditioned SVS, arXiv 2607.27768) — and is that outcome reachable at all at
this scale, or should the base model / pipeline be replaced?

## What changed since the last pass — measured, reproducible

These findings constrain every branch below and must not be re-derived.

| finding | measurement |
|---|---|
| The WER eval cannot steer anything | 15 segs / 141 words. n=3: base 45.86±3.36, LoRA r=16 41.14±13.81, LoRA r=32 42.55±8.51. All Welch p>0.58 |
| The prior 44.68→29.79 result is retracted | Both were single draws; 44.68 reproduces as an ordinary base sample, 29.79 was v1's favourable tail |
| Training saturates almost immediately | val loss/diff flat 0.4884/0.4885/0.4887 at steps 1000/1500/2000; val loss/stop rises 0.021→0.035 (overfit onset ~12 epochs over 5,485 samples) |
| A real bug was fixed, with no measurable gain | 151M embedding params (73,850×2,048) trained but never saved; adapter 18 MB→324 MB; no WER change this instrument can see |
| Timbre is not weight-resident | LoRA never moves identity — expected for a prompt-conditioned architecture |
| Grit survives the VAE | SHR drops only ~10.9 % median over 4,421 voiced frames → the weakness is in **generation**, not representation |
| The VAE is asymmetric | Encodes 16 kHz, decodes 48 kHz → everything above 8 kHz is decoder invention |

Assets: 12.7 h GTSinger English (3 singers), 0.4 h target-artist audio, an
**unused 180-segment GTSinger English val set**, RTX 4070 12 GB, Windows.

## Audience and decision

One practitioner. The decision is **where the next block of effort goes**, chosen
from: (a) fix the evaluation first, (b) change the phoneme/tokenizer frontend,
(c) get more or synthetic English data, (d) switch from LoRA to full finetune,
(e) switch base model or pipeline entirely, (f) stop — declare it out of reach at
this scale.

## Search branches (breadth 7, depth 3)

| # | Branch | Lens | Source classes |
|---|---|---|---|
| C1 | **The authors themselves** — GitHub issues/PRs/discussions, HF model+dataset threads, arXiv v2, author blogs, Zhihu/Bilibili/WeChat, Chinese-language sources. Do they say anything about English, multilingual plans, or hyperparameters? Has anyone shipped an English checkpoint? | Practitioner + primary | Repo, HF, Chinese web |
| C2 | **Community cross-lingual finetunes that worked** — VoxCPM/VoxCPM2 to a new language, plus SoulX-Singer, ACE-Step, DiffRhythm, YuE, Seed-VC, GPT-SoVITS, CosyVoice, F5-TTS. Settings, data volume, what they measured | Practitioner | GitHub, HF, forums |
| C3 | **Evaluation methodology** — how to build an intelligibility eval that is actually sensitive: item counts, seeds, greedy vs sampled decoding, CER vs WER, reported variance, minimum-eval-size guidance | Academic | Papers, eval toolkits |
| C4 | **Which intervention actually fixes cross-lingual singing** — IPA/G2P/romanization frontend vs more data vs full FT vs new base. Evidence that any of these fixed a real model | Academic + skeptic | arXiv, ablations |
| C5 | **Why does loss plateau at 13 h** — is early saturation expected for LoRA at this data scale? What do capacity/data-scaling results say, and is plateaued val loss even the right stopping signal for generative audio? | Academic | arXiv |
| C6 | **Alternative bases and pipelines** — if VocalRender is wrong for English, what is right? English-native or multilingual SVS with open weights, and the generate-then-convert route | Practitioner + economist | Model cards, benchmarks |
| C7 | **Blunt feasibility / counterargument** — evidence that solo cross-lingual SVS at 12 GB fails; where comparable projects died; what the realistic ceiling is | Skeptic | Post-mortems, issues |

## Search budget

| target | value |
|---|---|
| distinct search queries | ≥ 70 |
| candidate sources considered | ≥ 70 |
| source-ledger rows | ≥ 35 |
| final citations | 20–30 |

## Likely blind spots

- **Chinese-language sources.** The authors are a Chinese lab; the most useful
  guidance may never appear in English. Last pass did not search Chinese at all.
- **Survivorship bias.** Successful finetunes get posted; failures go silent, so
  "someone did it" evidence will look stronger than the base rate warrants.
- **Believing the demo.** Excellent Mandarin demos are in-domain, cherry-picked,
  and prompt-matched. They are weak evidence about English.
- **Confusing "loss plateaued" with "model converged."** Diffusion/flow losses are
  known to correlate poorly with perceptual quality; the plateau may be
  uninformative rather than terminal.
- **Sunk cost.** Substantial work exists on this path. The report must be willing
  to recommend abandoning it.

## Stop conditions

- Each of the five priority questions has a traceable answer or an explicit
  "no evidence exists" backed by ≥3 negative searches.
- The feasibility question gets a direct yes/no with stated conditions.
- Query budget met and no unfilled high-value gap remains.

## Expected artifacts

`research-plan.md` (this), `research-metrics.md`, `source-ledger.md`,
`evidence-map.md`, `verification-audit.md`, `report.html`, `quality-review.md`.
