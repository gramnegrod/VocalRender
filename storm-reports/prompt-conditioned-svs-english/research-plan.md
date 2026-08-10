# Research plan — Prompt-conditioned SVS: synthetic data, held-out timbre, cross-lingual finetune, English corpora

Date: 2026-08-09. Mode: **deep**.

## Research question

For a prompt-conditioned (zero-shot timbre) singing voice synthesis model trained
on Mandarin (VocalRender, arXiv 2607.27768 — 2,300 h, ~87 % SunoV5-synthetic;
WER 4.44 / SIM 0.922 / RPA 0.72, Mandarin in-domain):

1. How much synthetic singing data before diminishing returns, and does synthetic
   pretraining transfer **across languages** (Mandarin → English)?
2. Has anyone scaled a prompt-conditioned / zero-shot SVS to a genuinely
   **held-out artist**, and what SIM value corresponds to *human-convincing* timbre?
3. Is there a published recipe for finetuning a Chinese-trained singing model to
   English — LoRA vs full finetune, hours needed, phoneme/tokenizer handling?
4. What English singing corpora beat GTSinger's 3 English singers (~13 h) —
   licensing, singer count, hours, annotation quality?

## Audience and decision

Audience: single practitioner, RTX 4070 12 GB, `.venv` torch 2.10 cu128.
Assets: 12.7 h GTSinger English (3 singers), 0.4 h target-artist stems, an r=16
LoRA on LM+DiT that moved WER 44.7 % → 29.8 % and voice identity by **zero**.

Decision supported: **where to spend the next block of compute.** Concretely —
(a) build a synthetic English singing corpus and pretrain, (b) chase timbre by a
different mechanism than LoRA (speaker encoder, longer/multi prompt, PVC), or
(c) abandon the held-out-artist goal as out of reach for this architecture and
optimize only for general English quality.

## Search branches (breadth 6, depth 3)

| # | Branch | Lens | Source classes |
|---|---|---|---|
| B1 | Synthetic training data for speech/singing: scaling, diminishing returns, model collapse, cross-lingual transfer | Academic | arXiv, ICASSP/Interspeech/ACL, ISMIR |
| B2 | Zero-shot / prompt-conditioned timbre: held-out speaker & artist generalization; what SIM means | Skeptic + Academic | arXiv, VC challenge results, eval-metric papers |
| B3 | Cross-lingual finetuning of speech/singing LLMs: LoRA vs full FT, hours, tokenizer/phoneme handling | Practitioner | arXiv, GitHub issues, model cards, HF discussions |
| B4 | English singing corpora: inventory, licensing, singer count, annotation | Practitioner + Economist(licensing) | dataset papers, HF datasets, Zenodo, corpus sites |
| B5 | Adjacent industry reality: commercial voice cloning for singing (ElevenLabs, Suno, ACE-Step, Sovits/RVC/DiffSinger community) — what actually achieves convincing artist timbre | Missing stakeholder (end listener) | vendor docs, practitioner forums, community benchmarks |
| B6 | Counterargument & risk: evidence that this is unachievable at 12 GB / small data; legal & licensing exposure of artist voice cloning | Skeptic + Historian | law/policy, ELVIS Act, NO FAKES, CC BY-NC-SA analysis |

Depth: each branch runs up to 3 rounds; round N+1 questions come from
contradictions and gaps in round N.

## Search budget

| target | value |
|---|---|
| distinct search queries | ≥ 60 |
| candidate sources considered | ≥ 60 |
| source-ledger rows | ≥ 30 |
| final citations | 20–30 |

## Likely blind spots

- Conflating **speaker similarity metrics** (SIM/SECS from a speaker-verification
  encoder trained on *speech*) with perceived singing timbre. The encoder may be
  near-blind to rasp/grit — the exact thing the judge flagged.
- Publication bias: nobody publishes "we tried a held-out famous artist and it
  failed", so absence of evidence will look like absence of the technique.
- Synthetic-data results from TTS may not transfer to SVS (pitch/melody
  conditioning is a different information channel).
- Legal chill: the strongest work on artist voice cloning may be unpublished or
  commercial-closed precisely because of ELVIS Act / NO FAKES exposure.
- Chinese-language literature and Chinese-only leaderboards may hold the answer
  to Q3 and be missed by English-only search.

## Stop conditions

- All four questions have either a traceable answer or an explicit
  "no published evidence exists" backed by ≥3 negative searches.
- Deepening pass produces no new source class.
- Query budget met and evidence map has no unfilled high-value gap.

## Expected artifacts

`research-plan.md` (this), `research-metrics.md`, `source-ledger.md`,
`evidence-map.md`, `verification-audit.md`, `report.html`, `quality-review.md`.
