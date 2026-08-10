# Research metrics

Run 2026-08-10. Mode: deep. Follow-up to `prompt-conditioned-svs-english`.

## Per-branch

| branch | scope | queries executed | queries blocked | status |
|---|---|---:|---:|---|
| C1 | authors: repo, HF, arXiv, Chinese sources | 22 | 0 | complete |
| C2 | community cross-lingual finetunes | 14 | 5 | complete |
| C3 | evaluation methodology | 20 | 3 | complete |
| C4 | which intervention actually works | 8 | 2 | **degraded** |
| C5 | what a flat diffusion loss means | 8 | 2 | **degraded** |
| C6 | alternative bases and pipelines | 10 | 4 | **degraded** |
| C7 | blunt feasibility / skeptic | 6 | 6 | **degraded** |

## Totals

| metric | value | notes |
|---|---:|---|
| distinct_search_queries | **88** | Target ≥70. Met |
| queries blocked by session cap | **22** | See below — this is the run's main limitation |
| candidate_sources_considered | ~110 | |
| source_ledger_rows | **116** | Target ≥35. Met, 3× |
| final_citations | 34 | Target 20–30. Slightly over |
| confirmed_sources | 78 | |
| corrected_or_contested | 21 | |
| demoted_or_rejected | 17 | |
| budget_result | **Partially met** | Query target met; four branches truncated |

## The limitation, stated plainly

The **session-wide WebSearch cap of 200 calls was exhausted mid-run** — the
previous deep-research run on this topic consumed ~120 of them. C4, C5, C6 and
C7 each lost between 2 and 6 planned queries and compensated with direct
WebFetch retrieval of primary sources, which is a partial but not equivalent
substitute.

Concretely uncovered as a result:

- C7 (the skeptic branch) could not run its "projects that died" searches at all.
  Its Q2 is unanswered, and the absence of found failures is **uninformative**,
  not reassuring — abandoned hobbyist attempts are exactly the class of work
  that never gets indexed.
- C6 never reached DiffRhythm, Muskits/ESPnet-Muskits, VISinger2 English
  recipes, NANSY, Vocaloid 6, Suno/Udio, ElevenLabs Music, or MiniMax.
- C4 could not answer its Q5 (separate/elevated learning rate for the embedding
  during language adaptation).
- C2 never reached Seed-VC, MegaTTS3, Qwen3-TTS tokenizer, Fish-Speech, or
  DiffRhythm/YuE language transfer.

Unretrieved sources flagged by branches: Transinger full text (fetch failed
twice), BiSinger hours/steps/LR (binary PDF), the Interspeech 2025 PEFT-TTS PDF
(would not decode), and the horstmann.tech CosyVoice2 write-up (HTTP 403).

This run is therefore **not** claimed as complete coverage, and the quality
score is capped accordingly.
