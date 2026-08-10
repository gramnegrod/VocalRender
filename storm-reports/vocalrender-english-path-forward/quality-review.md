# Quality review

| dimension | score | justification |
|---|---:|---|
| Source authority | 5 | Heavily primary: the model's own code, the base model's repo and default config files, maintainer replies in issue threads, the authors' own HuggingFace statement, and the dataset's NeurIPS paper. Community and vendor sources were used only where the literature is silent and were demoted explicitly. |
| Citation accuracy | 5 | Two headline claims from the *previous* run were retracted after checking primary sources this time — the phoneme-frontend recommendation and the "model converged" reading. Three further figures were corrected (data hours, step-count interpretation, stop-head meaning). Nothing was carried forward unchecked. |
| Coverage breadth | 4 | Seven branches spanning authors, community, methodology, interventions, training dynamics, alternatives, and feasibility. Marked down because four branches were truncated by the search cap and whole classes of alternative system were never reached. |
| Recursive depth | 4 | Genuinely recursive in the places that mattered — reading the *base* model's repo rather than the fork's exposed the tokenizer-free architecture that invalidated the prior recommendation; reading GTSinger's own paper rather than a corpus summary halved the data estimate. Not a 5 because the deepening was opportunistic rather than budgeted, the cap having already bitten. |
| Contradiction handling | 5 | Five direct conflicts surfaced and none resolved by fiat: the 0.4–500 h data spread, LoRA-vs-full-FT on forgetting, whether the frontend is the dominant lever (strong evidence, zero applicability), whether to stop on a loss plateau, and whether 12 epochs is a lot. The report keeps the pro-stopping counterweight (Improved DDPM) rather than burying it. |
| Decision usefulness | 5 | It reverses a decision made the previous day, removes an intervention from the plan, surfaces a model that may make the whole programme unnecessary, and orders the next steps so the cheapest plan-changing experiments come first. |
| Specificity | 5 | Numbers and locations throughout: 6.71 h / 3 singers / 2 altos; r=32 α=32 LR 1e-4 1000 iters; WER 90–93 % vs 29–49 %; 1.91 h → 38.2 h; Whisper 35.5–37.7 % on sung audio; ±4.2 pp binomial SE; ~2,500 words needed. |
| Uncertainty handling | 5 | 5 do-not-assert items including the tempting 15-minute result and the report's own most attractive lead. The frontier question explicitly names SoulX-Singer's evidence as first-party and unreplicated rather than presenting it as an answer. |
| Missing-stakeholder handling | 4 | The evaluator was seated deliberately and produced the run's sharpest methodological finding — Whisper's own sung-audio error rate is the same magnitude as the signal. Not a 5 because no singer, vocal coach, or mixing engineer was consulted on what "convincing English singing" would even require. |
| Writing coherence | 4 | Written once from the ledger and audit; the hidden-connection section does real work by unifying four separate disappointments into one cause. Marked down for length and for a Q4 table that is a reference artefact interrupting an argument. |

**Total: 46 / 50.**

## Score caps checked

| cap | applies? |
|---|---|
| No `research-metrics.md` → cap 38 | No — present, and it discloses the truncation |
| <60 distinct queries → cap 42 | No — 88 executed |
| No rejected/demoted claims → cap 40 | No — 5 do-not-assert, 7 retracted/rejected in a dedicated section |
| No verification audit → cap 35 | No — all 67 findings graded |

No cap binds.

## Honest weaknesses

1. **Four of seven branches were truncated.** The skeptic branch in particular never ran its "projects that died" searches, so its verdict rests on data-scale arguments rather than on observed failures. The report says the absence is uninformative, but that is a real hole in the most consequential branch.
2. **The central recommendation depends on an untested model.** SoulX-Singer's every quality claim is first-party. If it disappoints, the report's cheerful "the goal is reachable, the method was wrong" framing weakens considerably.
3. **The data-tier claims lean on a vendor blog.** The 30–100 h / 10–30 speaker numbers underpinning "this is ruled out" have their clearest statement in a source selling vocal data. Direction is corroborated by SingNet and DiTSinger; the specific tiers are not.
4. **This run reverses the previous run.** That is the process working, but it should lower confidence in *this* run's conclusions by the same token — the prior report was also internally coherent and well-cited, and it was wrong about the frontend.
5. **No hands-on verification.** Every recommendation here is literature-derived. The three diagnostics proposed in §05 are exactly the things that would ground it, and none have been run yet.
