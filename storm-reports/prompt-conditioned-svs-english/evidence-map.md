# Evidence map

Written after the first pass, used to drive the deepening round. Records where
the evidence is strong, where it conflicts, and what is simply absent.

---

## 1. What a domain expert would expect that we had not checked

| Expectation | Checked? | Result |
|---|---|---|
| That SIM is validated against human similarity judgements for singing | Yes | **Absent.** VoxSim (ρ≈0.75 in-domain, 0.50 OOD) is speech. SVCC gives ρ≈0.63 (2023) / ≈0.8 (2025) for singer-embedding distance but publishes no cosine→perceived-identity mapping |
| That the SIM encoder is validated on singing at all | Yes | **It is not.** Speech-trained SV models hit ~40 % EER on VocalSet (B2-05) while the field computes singing SIM with exactly those encoders (B2-16) |
| That someone has done Mandarin-synthetic → English-singing transfer | Yes | **Nobody has.** Nearest is Mandarin-synthetic → Japanese, +0.10 MOS (B1-12) |
| That the base model's own docs cover vocabulary expansion | Yes | **They do not.** VoxCPM finetuning docs are silent on embedding resize (DV target 6) |
| That the embedding-init failure mode from the LLM literature applies here | Yes — read the code | **Refuted locally.** Sinusoidal init for score tokens (LOCAL-01) |
| That prompt length is a lever | Yes | **It is not.** SIM flattens by ~3 s (B2-13); matches the user's 8.6 s null |
| That grit/rasp has a known acoustic carrier | Yes | **Subharmonics / period-doubling** (B5-15) — and it breaks the periodic+aperiodic assumption every singing vocoder makes (B5-17) |

## 2. What source would change the recommendation

- A published SVS system evaluated on a **named distinctive artist** with human ratings. Would tell us whether the goal is reachable at all. **Does not exist.**
- A **clean-vs-separated-stems ablation** on the same SVS model and singers. Would price the cost of building a corpus by source separation. **Does not exist** (B4 gap).
- A **subharmonic-preservation measurement through a neural VAE round-trip.** Would confirm or kill the "grit dies in the latent" hypothesis. **Does not exist** — and is cheap to run locally.
- A **rank-64 row** in a cross-lingual LoRA ablation. Only 16 vs 32 is tabulated (B3-03).
- Any **SVS data-scaling curve past 530 h.** DiTSinger shows no saturation by 530 h (B1-13); the 2,000 h regime is unmeasured.

## 3. Claims resting on vendors, anecdotes, or self-evaluation — flagged

| Claim | Weakness |
|---|---|
| VoxCPM "500+ hours for a new language", r=32/r=64 guidance | Vendor documentation with **no supporting experiment published** |
| MiniMax "up to 99 % vocal similarity" | Vendor marketing, speech not singing (B5-12) |
| Seed-VC beats per-speaker RVC | **Author's own eval**, objective metrics only (B5-09) |
| RVC 10–30 min / 300–500 epochs | Community wiki consensus, not peer-reviewed (B5-04) |
| F5-TTS "100 h good, 300 h perfect" | GitHub folklore, no controlled numbers (B3-11) |
| Suno persona identity fidelity | **No measured evaluation exists anywhere** (B5-11) |
| ELVIS Act / NO FAKES interpretation | Law-firm alerts and EFF advocacy; one primary statute text obtained, S.4591 primary text blocked (403) |

## 4. Where sources directly conflict

1. **Hours needed for a new language.** VoxCPM 500+ h · F5-TTS practitioners 100–300 h · a 60 h Vietnamese run **failed** at 160k steps · Phir Hera Fairy 10 h/L ≈ 100 h/L · Muskits adapted on 42.6 min. **Not reconcilable from the evidence.** They differ in architecture, in whether the target language was already in pretraining, and in step count. **Step count is a plausible hidden variable no source isolates.**
2. **LoRA vs full finetune for forgetting.** Three papers, three answers (B3-19). No clean winner.
3. **Zero-shot vs per-speaker conversion for identity.** B5-07/08 argue architecture favours retrieval; B5-09 reports the opposite on SECS while losing on DNSMOS. **Reconciliation adopted:** zero-shot can match on embedding-space similarity while losing on audio quality and style dynamics — which is what a human judge grades.
4. **Synthetic:real ratio.** 10–30 % synthetic optimal in ASR (B1-05), non-monotonic, versus VocalRender's 87.5 % synthetic. Unresolved; different tasks.
5. **Does ShareAlike reach trained weights.** Creative Commons says "it depends" and leans yes-if-published; Guadamuz and others say the derivative-work case is legally limited (B6-10 vs B6-11). Unlitigated.

## 5. Missing stakeholder

The seat that changes the conclusion is **the listener**, and the literature
systematically fails to seat them. Every zero-shot SVS paper reports SIM; SVCC
is the only line of work that runs proper human similarity panels, and it is
precisely there that the field's story falls apart (naturalness reaches human
level, identity and technique do not). A secondary missing seat is **the
estate/rights-holder** — legally the party with standing, and the reason the
most capable artist-cloning work is likely commercial and unpublished.

## 6. Adjacent queries that exposed unknown unknowns

- Searching **voice-science** literature rather than ML literature produced the
  single most useful mechanistic finding (subharmonics, B5-15) — a result no
  amount of SVS-paper reading would have surfaced.
- Searching the **AI-cover community** rather than academia produced the only
  empirically-validated recipe for artist timbre (B5-04, B5-07).
- Reading the **local repository** refuted a literature-derived hypothesis
  (LOCAL-01) and found an unreported second instance of a known bug (LOCAL-03).

## 7. Time-sensitive facts needing freshest-source verification

| Fact | Status |
|---|---|
| NO FAKES Act S.4591 status | Verified to 18 Jun 2026 committee advance; **no floor action found as of Aug 2026**. Not law |
| EU AI Act Art. 50 applicability | **Applies from 2 Aug 2026** — i.e. one week before this report. Grace to 2 Dec 2026 |
| Suno litigation | Munich ruling July 2026; RIAA Boston ongoing; Sony unsettled |
| DAMP dataset access | Zenodo records verified live Aug 2026; **Smule approval latency in 2026 untested** |
| SingNet audio release | Checkpoints only as of Aug 2026; audio release unconfirmed |
| so-vits-svc | Officially discontinued/archived; RVC/Applio is the 2026 default |

## 8. Gaps accepted as unresolvable within this run

- S.4591 primary statutory text (congress.gov and govtrack both returned 403).
- Minixhofer Interspeech 2025 scaling-law exponents (binary PDF, three fetch attempts).
- PopBuTFy availability and true English hour count (18 h vs 40.4 h conflict, no download link).
- Whether Smule still grants DAMP access requests in 2026.
- OpenSinger hours (50 h/66 singers vs 85 h/93 singers, unresolved against a primary page).
