# Research metrics

Run: 2026-08-09. Mode: deep. Topic: prompt-conditioned SVS — synthetic data,
held-out timbre, cross-lingual finetune, English corpora.

## Per-branch query counts

| branch | scope | queries | status |
|---|---|---:|---|
| B1 | synthetic data scaling, model collapse, cross-lingual transfer | 20 | complete |
| B2 | held-out artist timbre, what SIM measures | 17 | complete |
| B3 | Chinese→English finetune recipes, LoRA vs full FT | 18 | complete |
| B4 | English singing corpora inventory | 18 | complete |
| B5 | commercial/practitioner artist-timbre reality | 14 | complete |
| B6 | counterargument: compute wall, small data, legal | 15 | complete |
| DV | deepening/verification pass, 6 unverified load-bearing claims | 18 | complete |
| LOCAL | repository code inspection (not a web query) | — | complete |

## Totals

| metric | value | notes |
|---|---:|---|
| distinct_search_queries | **120** | Target ≥60. **Met, 2×** |
| search_result_pages_opened | ~150 | 203 subagent tool calls total; the majority were WebFetch/WebSearch |
| candidate_sources_considered | ~145 | Target ≥60. Met |
| source_ledger_rows | **135** | 132 web + 3 local code. Target ≥30. Met, 4.5× |
| final_citations | **43** | Target 20–30. Slightly over — kept because each carries a distinct load-bearing number |
| confirmed_sources | 98 | |
| corrected_or_contested_sources | 24 | 6 corrected during the deepening pass, 18 contested |
| demoted_or_rejected_sources | 13 | 8 demoted, 5 rejected outright |
| findings_graded | 81 | 41 safe to assert · 27 assert-with-caveat · 13 do-not-assert |
| budget_result | **Met** | All deep-mode targets exceeded |

## Query accounting note

Counts are the per-branch totals each subagent reported at the end of its run,
summed. They are queries *submitted*, not sources opened. Sources opened is
estimated from tool-call counts rather than logged exactly, and is marked
approximate for that reason.

## Retrieval failures — recorded, not papered over

| target | failure | consequence |
|---|---|---|
| S.4591 statutory text | congress.gov and govtrack both HTTP 403 | Tool-liability and postmortem-term details rest on law-firm/EFF quotation, flagged in the audit |
| Minixhofer Interspeech 2025 | binary PDF defeated 3 fetch attempts | Scaling-law exponents unextracted; claim kept at abstract level |
| elevenlabs.io (first attempt) | WebFetch domain-safety block in branch B5 | **Resolved** — the deepening pass fetched it successfully and confirmed the quote verbatim |
| Kwon et al. Interspeech 2025 PDF | unparseable in B3 | **Resolved** — deepening pass extracted the full ablation table |
| Phir Hera Fairy PDF | fetch failed in B3 | **Resolved** — arXiv HTML worked; claim narrowed as a result |
| PopBuTFy | no public download URL exists | Corpus excluded from actionable recommendations |
| Transinger full text | ResearchGate gated, PMC 500 | Abstract-level only; numbers unverified |

The prior session's WebSearch cap was **not** hit this run — no branch reported
truncation or rate-limiting.
