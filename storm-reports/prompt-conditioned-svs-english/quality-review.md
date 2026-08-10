# Quality review

Self-evaluation against the skill's evaluator rubric. Scores 1–5.

| dimension | score | justification |
|---|---:|---|
| Source authority | 5 | 14 peer-reviewed/journal, 48 arXiv (fetched in full for numeric claims), 6 primary legal/official (statute text, EU Commission FAQ, a federal court decision, Creative Commons' own page), 21 verified dataset records. Vendor and community sources were used only where the literature is silent, and explicitly demoted. Three rows are primary source code — the strongest class available. |
| Citation accuracy | 5 | Every numeric claim traced. The deepening pass re-fetched six load-bearing claims that first-pass branches could not verify; **four were narrowed or refuted as a result**, including two that had already been reported to the user and were corrected in place. Zero fabrications found. |
| Coverage breadth | 5 | Six source classes swept: academic, official/regulatory, vendor/product, practitioner/community, dataset-primary, and counterargument/risk. The corpus inventory covers 38 datasets verified per record. Legal coverage spans three jurisdictions plus licensing. |
| Recursive depth | 4 | Depth 3 within branches, plus a targeted seventh pass. Genuinely recursive in places — the voice-science search was triggered by the judge's "lacks grit" critique, which then reframed the whole timbre analysis. Not a 5 because the frontier question was identified rather than answered, and two branches' follow-ups were bounded by PDF-extraction failures rather than by exhaustion. |
| Contradiction handling | 5 | Five direct conflicts surfaced and none papered over: the hours question (five irreconcilable sources, with step count named as the uncontrolled hidden variable), LoRA-vs-full-FT forgetting (three papers, three answers), zero-shot-vs-conversion (Seed-VC contradicts the architectural argument — reconciliation stated, simple claim refused), synthetic:real ratio, and ShareAlike reach. Two dedicated "contested signal" cards in the report tell the reader not to assert. |
| Decision usefulness | 5 | The report changes what to do. It removes two items from the user's plan (ElevenLabs PVC is prohibited for singing; longer prompts are futile), reframes a third as expected-not-broken (zero timbre movement), supplies concrete corrected hyperparameters against the current run, and names one afternoon-scale experiment that would settle the project's central uncertainty. |
| Specificity | 5 | Findings carry numbers and locations: SIM 0.922 against a 0.918–0.929 topline; ~40 % EER on VocalSet; 37–44 % technique accuracy; SingStyle111 English = 372 min across 6 of 8 singers; `svs_utils.py:250` and `:115–130` as bug sites; r=16 vs r=32 CER 4.906 vs 4.570. |
| Uncertainty handling | 5 | Three-tier claim safety with 13 items explicitly marked do-not-assert, including the report's own most attractive hypothesis ("grit dies in the VAE"), which is labelled untested rather than concluded. Retrieval failures are listed in the metrics file rather than hidden. |
| Missing-stakeholder handling | 4 | The listener was seated deliberately, and doing so produced the single most useful mechanistic finding (subharmonics) by searching voice science instead of ML. The rights-holder/estate seat is named in the assumptions section as an *evidentiary* gap — the reason "nobody has published this" may reflect legal chill rather than difficulty. Not a 5 because no attempt was made to reach actual singers, vocal coaches, or mastering engineers as a source class, and their judgement of what makes a voice convincing would likely sharpen the analysis further. |
| Writing coherence | 4 | Written once from the ledger, evidence map and audit, not stitched from branch outputs, and the hidden-connection section does real synthesis work by joining two literatures that do not cite each other. Marked down one because the report is long, and the corpus inventory in particular is a reference table interrupting an argument rather than part of it. |

**Total: 47 / 50.**

## Score caps checked

| cap | applies? |
|---|---|
| No `research-metrics.md` → cap 38 | No — present and populated from per-branch reported counts |
| <60 distinct queries → cap 42 | No — **120 queries**, twice the deep-mode floor |
| No rejected or demoted claims → cap 40 | No — 13 do-not-assert, 8 demoted, 5 rejected outright, listed in a dedicated section |
| No verification audit → cap 35 | No — `verification-audit.md` grades all 81 findings |

No cap binds. 47/50 stands, comfortably above the 35/50 completeness threshold.

## Honest weaknesses

1. **The frontier question is unanswered, not just unresolved by the literature.** It is answerable on this machine in an afternoon, and the report would be materially stronger if the measurement had been run before writing.
2. **Six branches share one framing.** They were author-constructed from a single problem statement, so their agreement is a strong hypothesis rather than independent corroboration — flagged in the report's "how to read this" but worth repeating.
3. **Publication bias is unquantifiable here.** Every "no published work exists" finding could reflect legal chill around artist voice cloning rather than technical difficulty. The report says so; it cannot correct for it.
4. **Two secondary-source dependencies survive.** The NO FAKES tool-liability and postmortem provisions rest on law-firm and EFF quotation because congress.gov returned 403 twice. A future pass should retry or use a govinfo bulk-data endpoint.
5. **The 43 citations exceed the 20–30 target.** Defensible — each carries a distinct load-bearing number — but it makes the evidence base longer to audit than the rubric intends.

## Comparison note

This run is **not** claimed as ChatGPT Deep Research parity. It clears the
100-query benchmark-mode threshold, and it has search telemetry, a source
ledger, a verification audit, and rejected/demoted claims. But it was scoped and
executed as `deep`, not `benchmark`: there was no "missed by first pass" sweep
driven by a competing report's terminology. The deepening pass targeted six
known-unverified claims rather than hunting for unknown unknowns.
