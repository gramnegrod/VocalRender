# Verification audit

Audit date: 2026-08-16. Each high-impact statement was checked against a primary source or direct local evidence. Vendor quality claims remain labeled as vendor claims.

| # | Claim audited | Evidence | Result | Safe wording |
|---:|---|---|---|---|
| 1 | VocalRender currently produces compelling Amy-like identity for only ~2–3 seconds. | Local Experiment 05 notes and outputs. | Verified locally, small sample. | “The current best local result is unusually convincing in the opening ~2–3 seconds, then drifts.” |
| 2 | Current WER is about 27%. | Experiment 05 reports CFG 3.0 mean 26.95% ± 2.13%. | Verified within that evaluation path. | “About 27% in the current code path,” not a universal model score. |
| 3 | Chunk re-anchoring is proven to solve the drift. | Local failure timing plus VoxCPM issue #302. | Not proven. | “Highest-information next experiment.” |
| 4 | ACE-Step 1.5 can run now. | Official 8–16 GB guidance; live RTX 4070 has 12 GB. | Plausible and documentation-backed, not yet executed here. | “Immediate install/bake-off candidate.” |
| 5 | ACE-Step 1.5 matches or beats Suno. | ACE author benchmark and arena. | Contested/self-reported. | “Competitive author-reported results; must be independently tested.” |
| 6 | MiniMax Music3 is released. | Live official GitHub and Hugging Face repositories. | Verified. | “Released open-weight candidate.” |
| 7 | MiniMax Music3 runs on this machine now. | Official two-GPU inference; live system exposes one 12 GB GPU. | False under current visibility. | “Blocked until both GPUs are visible and memory fit is tested.” |
| 8 | HeartMuLa preserves a reference singer. | Paper representation excludes speaker timbre; released repo lacks reference conditioning. | Not supported. | “Full-song baseline, not a current identity solution.” |
| 9 | LeVo2 is an available commercial foundation. | Paper exists, but official repository returned 404 and model access failed. | Not supported. | “Promising paper; unavailable/restricted in live verification.” |
| 10 | Stable Audio 3 is a full lyrical-song replacement. | Official model card emphasizes instrumental prompts and editing. | Not supported. | “Instrumental/accompaniment and inpainting specialist.” |
| 11 | Lyria 3 Pro costs $0.08 per full song. | Current official Gemini API pricing. | Verified as preview pricing. | Include “preview” and date. |
| 12 | Eleven Music costs $0.15/minute and can be silently resold. | Current pricing and API/model terms. | Price verified; silent resale contradicted. | “Useful benchmark; pure-play co-branding/rights constraints apply.” |
| 13 | A local frontier model can be trained from scratch on the 3090/4070. | Frontier papers disclose billion-scale stacks, very large datasets, and fleets. | Not realistic. | “Use local GPUs for inference, adapters, and narrow fine-tuning.” |
| 14 | Suno output can seed competitive training. | Suno terms prohibit using service/output to build or train a competing service. | Prohibited by current terms. | “Do not use Suno output as distillation/training data.” |
| 15 | The current Amy path can ship commercially. | Local data note, model/service consent clauses, and voice-rights law. | Not supportable. | “Replace celebrity imitation with consented/original voice partners.” |
| 16 | Automated WER/similarity or an audio LLM can decide the winner. | SongBench, MAD/FAD study, AudioJudge bias findings, and local opening-vs-tail miss. | Contradicted. | “Use metrics diagnostically; blind, time-sliced listener preference is the primary gate.” |

## Contradictions resolved

- **Model rankings:** left unresolved and explicitly attributed because each author tests different versions and protocols.
- **LeVo2 availability:** changed from leading candidate to watchlist after live 404/access failures.
- **HeartMuLa identity conditioning:** corrected from a possible voice-reference candidate to a full-song baseline.
- **Local hardware:** historical dual-GPU assumptions were replaced by the current `nvidia-smi` result: one visible RTX 4070.
- **Commercial API economics:** Lyria 3 Pro is far cheaper than expected at current preview pricing, but preview stability and single-turn limitations prevent it from being the sole product architecture.

## Claims deliberately excluded

- “Any open model beats Suno” as an objective fact.
- Exact MiniMax Music3 VRAM fit, because the project does not publish a decisive requirement and it was not run locally.
- Commercial usability of LeVo2 while its live artifacts and exact terms cannot be verified.
- A federal ban on all AI voice imitation; federal proposals were not treated as enacted law.
