# Verification audit

Every candidate finding, traced and graded. Three verdicts:
**SAFE TO ASSERT** · **ASSERT WITH CAVEAT** · **DO NOT ASSERT**.

---

## Q1 — Synthetic data: how much, and does it cross languages?

| # | Finding | Primary source | Verdict |
|---|---|---|---|
| 1.1 | Model collapse does not threaten a synth-pretrain → real-finetune schedule; collapse in Shumailov et al. arises from **replacing** real data each generation, and Gerstgrasser et al. prove accumulation bounds the error | Nature 631:755–759; arXiv 2404.01413 | **SAFE TO ASSERT** |
| 1.2 | Cherry-picking only good-sounding synthetic outputs costs diversity (MAD) | arXiv 2307.01850, ICLR 2024 | **SAFE TO ASSERT** |
| 1.3 | Synthetic singing pretraining transfers across a language boundary | ACE-Opencpop, arXiv 2401.17619v2 | **ASSERT WITH CAVEAT** — single datapoint, Mandarin→Japanese, **+0.10 MOS**. Do not generalize the magnitude to Mandarin→English |
| 1.4 | No published Mandarin-synthetic → English-singing transfer exists | 3+ negative searches (B1 GAPS) | **SAFE TO ASSERT** as a negative result |
| 1.5 | 135 h of synthetic singing on a speech-pretrained backbone reaches SOTA-comparable SVS quality | arXiv 2512.14657 | **SAFE TO ASSERT** — SingMOS 4.09 vs 4.08. Note F0 RMSE lagged 62.79 vs 55.83 Hz |
| 1.6 | The synthetic-data penalty in SVS concentrates in pitch, not intelligibility | Synthesis of 2607.27768 + 2512.14657 | **ASSERT WITH CAVEAT** — this is our synthesis across two papers, not any single paper's stated claim |
| 1.7 | Diminishing-returns knee for synthetic data is ~200–300 h | arXiv 2601.00935 | **ASSERT WITH CAVEAT** — **ASR, not SVS.** The only SVS curve (DiTSinger, 30→530 h) shows *no* saturation |
| 1.8 | Synthetic beyond ~20 h can actively degrade a small-scale finetune | arXiv 2604.27273 | **ASSERT WITH CAVEAT** — accented ASR; the regime matches a hobby-scale corpus but the task does not |
| 1.9 | Curation beats volume | arXiv 2506.23859v1 | **ASSERT WITH CAVEAT** — speech enhancement, adjacent task |
| 1.10 | Synthetic-only training fails; mixing real recovers most of the loss | B1-04, B1-05, B1-26 converging | **SAFE TO ASSERT** |
| 1.11 | There is no published scaling curve for synthetic **singing** data in hours | B1 GAPS | **SAFE TO ASSERT** as a negative result |
| 1.12 | Suno-output licensability is legally unsettled after the July 2026 Munich ruling | techtimes news-tier | **DO NOT ASSERT** as fact. State only that a ruling occurred and the downstream question is unresolved |

## Q2 — Held-out artist, and what SIM means

| # | Finding | Primary source | Verdict |
|---|---|---|---|
| 2.1 | VocalRender's SIM 0.922 sits inside its own ground-truth topline band of 0.918–0.929 — the metric is **saturated** | arXiv 2607.27768, fetched | **SAFE TO ASSERT** — the strongest single finding in this report |
| 2.2 | Singing SIM is computed with speech-trained encoders (WavLM-TDNN / ECAPA on VoxCeleb / CN-Celeb) | 2312.04919 + VocalRender + SoulX | **SAFE TO ASSERT** |
| 2.3 | Those encoders score ~40 % EER on VocalSet — near-chance on extended vocal technique | arXiv 2401.05064v1 | **SAFE TO ASSERT** |
| 2.4 | Therefore the reported SIM is near-blind to rasp/growl | 2.2 + 2.3 | **ASSERT WITH CAVEAT** — a sound inference from two confirmed facts, but **no study directly probes voice quality in speaker embeddings** (B2-07 gap). Present as inference, not measurement |
| 2.5 | VocalRender's prompt is drawn from the **same song** as the target, a channel confound that inflates SIM | arXiv 2607.27768 §prompt; confound mechanism from 2507.02176 | **SAFE TO ASSERT** for the protocol; **ASSERT WITH CAVEAT** for the magnitude of inflation, which nobody has measured in SVS |
| 2.6 | No published prompt-conditioned SVS has been evaluated on a named, commercially distinctive recording artist as a held-out target | Exhaustive negative search across SVCC 2023/2025, SoulX, TCSinger, StyleSinger, VocalRender | **SAFE TO ASSERT** as a negative result |
| 2.7 | There is no published cosine value that certifies human-convincing timbre | B2 GAPS | **SAFE TO ASSERT** as a negative result |
| 2.8 | SIM saturates by ~3 s of prompt; longer prompts do not help | VALL-E 2, Voicebox | **SAFE TO ASSERT** |
| 2.9 | SVCC 2023: no team matched target-speaker similarity (~0.4 MOS gap) while naturalness reached human level | arXiv 2306.14422 | **SAFE TO ASSERT** |
| 2.10 | SVCC 2025: identity comparable to GT, but **vocal technique reproduced at only 37–44 %** | arXiv 2509.15629 | **SAFE TO ASSERT** |
| 2.11 | "Smoother, lighter" is the signature of mean-seeking generation losing aperiodic detail | XiaoiceSing2 + 2405.09940v2 | **ASSERT WITH CAVEAT** — mechanism is well established for MSE-trained models; applying it to this specific diffusion+VAE stack is inference |
| 2.12 | SECS ≥ 0.80 means "same speaker" | Practitioner convention | **DO NOT ASSERT.** Encoder- and normalization-specific, no perceptual validation |
| 2.13 | There is an information-theoretic bound on prompt-carried identity | — | **DO NOT ASSERT.** No such published bound exists; only an empirical saturation curve |

## Q3 — Cross-lingual finetune recipe

| # | Finding | Primary source | Verdict |
|---|---|---|---|
| 3.1 | Timbre in a prompt-conditioned architecture is a **conditioning input**, not a weight-resident property; LoRA on LM+DiT cannot move it | MiniMax-Speech 2505.07916 (explicitly uses LoRA for emotion, not timbre); UtterTune (speaker sim ~0.69 unchanged across all systems) | **SAFE TO ASSERT.** Two independent confirmations. The user's zero timbre movement is the **expected** outcome |
| 3.2 | In architectures **without** a speaker-prompt encoder, DiT LoRA *can* move timbre | ACE-Step issue #259 | **ASSERT WITH CAVEAT** — practitioner report, weak source, but it is the contrast case that makes 3.1 coherent |
| 3.3 | VoxCPM docs recommend r=32 speaker / r=64 style-language, 500+ h for a new language, 10–20 % original-data mixing, lr 1e-4 LoRA vs 1e-5 full-FT, `enable_dit: true` | voxcpm.readthedocs.io, verified verbatim | **SAFE TO ASSERT** as vendor guidance. **Flag that no supporting experiment is published** |
| 3.4 | VoxCPM recommends full finetune over LoRA for a new language | Same | **DO NOT ASSERT as doc wording.** The 500+ h figure sits under the full-FT section; the recommendation is **implicit**. Corrected during the deepening pass |
| 3.5 | Raising LoRA rank 16→64 loses speaker-adaptation ability and degrades audio quality | Kwon et al., Interspeech 2025, quoted verbatim | **ASSERT WITH CAVEAT.** This is the authors' **prose**; the only tabulated ablation is r=16 vs r=32, where r=32 improved CER 4.906→4.570 and WER 12.625→11.338 and SIM-O 0.640→0.645 but **worsened UTMOS 3.338→3.264**. No r=64 row exists |
| 3.6 | 20k steps, not 1k, is the relevant order for a language LoRA; UtterTune moved accent correctness 0.472→0.975 in <1 h on a 4090 | arXiv 2508.09767v1 | **SAFE TO ASSERT** |
| 3.7 | LoRA should be applied to Q, K, V, **and** O of every self-attention block, not just q/v | UtterTune | **SAFE TO ASSERT** for that system's configuration |
| 3.8 | 10 h/language suffices, beating 100 h | Phir Hera Fairy 2505.20693v1 | **ASSERT WITH CAVEAT — narrowed.** True for **WER in a monoglot setup** (31.3 % vs 40.0 %). On MUSHRA the ordering is 1h 33.7 < 10h 61.5 < **100h 64.3**. Defensible claim: 10 h ≈ 100 h at ~0.8 % average metric loss, **not** "beats" |
| 3.9 | The hours question has no consensus and step count is an uncontrolled hidden variable | Five mutually inconsistent sources | **SAFE TO ASSERT** — and it is the honest answer to Q3 |
| 3.10 | Cross-lingual failure in these systems is a **phoneme-representation** problem more than a data-volume problem | DiaMoE-TTS (unified IPA), Transinger (IPA letters+diacritics), BiSinger (shared CMU-dict phone space), GPT-SoVITS (asymmetric frontend → unfixable by acoustic finetuning), XTTS (romanization) | **SAFE TO ASSERT.** Five independent systems converge; this is the strongest convergent finding for Q3 |
| 3.11 | A 60 h Vietnamese F5-TTS run at 160k steps produced unintelligible output | GitHub issue 849 | **ASSERT WITH CAVEAT** — single unresolved issue report; lr 5e-6 is plausibly the cause, not the hours |
| 3.12 | Random init of new embedding rows explains the English WER floor | LLM vocab-expansion literature | **DO NOT ASSERT — REFUTED locally.** `svs_utils.py:140–248` sinusoidally initializes pitch/BPM/duration tokens |
| 3.13 | `svs_utils.py:250` installing a fresh `nn.Embedding` is the root cause of the 151M-unsaved-params bug | Direct code read | **CORRECTED — DO NOT ASSERT AS WRITTEN.** The runtime log shows `svs_utils` took its early-return path (`Added 0 SVS tokens`; vocab already 73,850). The real site is `voxcpm2.py:1608–1617`, where `from_local` auto-resizes the embedding to match the checkpoint *after* the freeze loop at 1561–1567. Mechanism identical, location wrong |
| 3.13b | `voxcpm2.py:1608–1617` is the root cause | Code read **plus runtime confirmation**: `Auto-resizing embedding: 73448 -> 73850`, then `non-LoRA: 151,244,800 in 1 tensors — base_lm.embed_tokens.weight` | **SAFE TO ASSERT** — this is now measured, not inferred |
| 3.14 | `score_lm_head` has the same unsaved-parameter problem | Direct code read, lines 115–130 | **ASSERT WITH CAVEAT.** The defect is real in the code but **dormant in this configuration** — it did not fire in the v2 run because the vocab already matched. Latent, not active |
| 3.16 | r=32 LoRA costs ~18M parameters, and the old config's "r=32 = 169M trainable" note was measuring the phantom embedding | Runtime: 169,332,736 total − 151,244,800 embedding = **18,087,936** | **SAFE TO ASSERT** — measured exactly as predicted |
| 3.15 | Commercial cross-lingual SVS still ships an audible non-native accent | Synthesizer V product docs | **ASSERT WITH CAVEAT** — SynthV *deliberately* retains accent as a design choice, which is not the same as being unable to remove it |

## Q4 — English singing corpora

| # | Finding | Primary source | Verdict |
|---|---|---|---|
| 4.1 | SingStyle111's English portion is **372 min = 6.2 h across 6 of 8 singers**, with only one native English speaker | CMU paper PDF, verified in deepening pass | **SAFE TO ASSERT** |
| 4.2 | SingStyle111 is CC BY 4.0 — commercially usable, unlike GTSinger | Zenodo record | **SAFE TO ASSERT** |
| 4.3 | It is therefore **not** an upgrade on GTSinger English for volume (6.2 h vs 12.7 h), only for licence | 4.1 + 4.2 | **SAFE TO ASSERT** |
| 4.4 | DAMP links are **not** dead; Zenodo records are live with files gated behind a Smule licence request | zenodo.org/records/2747436 etc., verified | **SAFE TO ASSERT** — corrects a widespread claim |
| 4.5 | DSing30 (149.1 h) is the largest English singing audio corpus | Kaldi-Dsing-task | **ASSERT WITH CAVEAT** — amateur phone recordings, sentence-level lyrics only, **no note annotation**, NC licence, and requires Smule source access |
| 4.6 | DALI v2 (7,756 songs) is the largest English-dominant note-level annotation set | GitHub README | **ASSERT WITH CAVEAT** — **annotations only; audio is not distributed** and must be scraped from YouTube, with link rot |
| 4.7 | Nothing publicly available beats GTSinger English on hours + note-level annotation + shipped audio | Full inventory sweep | **SAFE TO ASSERT** |
| 4.8 | Only VocalSet, SingStyle111, vocadito, Erkomaishvili, Dagstuhl ChoirSet, AVP (CC BY 4.0) and CrawlSinger-OS (MIT) permit commercial use | Per-record licence check | **ASSERT WITH CAVEAT** — CrawlSinger-OS's MIT tag covers the packaging, not the underlying crawled recordings' copyright |
| 4.9 | SingNet (~3,000 h) does not change the picture because the audio is not released | arXiv 2505.09325 | **SAFE TO ASSERT** — checkpoints only; English hours and licence unstated |
| 4.10 | Source separation is now the mainstream corpus-building strategy | SingNet (MDX23), CrawlSinger-OS (mel-RoFormer + de-reverb), DeepSinger | **SAFE TO ASSERT** |
| 4.11 | Separated-vocal training costs roughly a third to half a MOS point | SingNet 3.52 vs 3.64; Demucs vs Conv-TasNet 3.22 vs 2.85 | **DO NOT ASSERT as a measurement.** **No clean-vs-separated SVS ablation exists.** The SingNet gap conflates vocoder and separation loss. State as an unquantified estimate only |
| 4.12 | PopBuTFy is ~40.4 h of English | NSVB paper | **DO NOT ASSERT.** No public download found; hours conflict (18 h vs 40.4 h) across papers |

## Q5 — Practitioner reality (feeds the recommendation)

| # | Finding | Primary source | Verdict |
|---|---|---|---|
| 5.1 | ElevenLabs PVC **does not support singing** — "Audio recordings must consist of spoken voice only" | Official docs, quoted verbatim after re-fetch | **SAFE TO ASSERT.** Removes PVC from the option set |
| 5.2 | ElevenLabs PVC needs 30 min minimum, 2–3 h for best results | Same | **SAFE TO ASSERT** (narrowed from "~3 h") |
| 5.3 | RVC achieves artist timbre from 10–30 min of clean isolated vocal at 250–500 epochs with RMVPE | AI Hub wiki + voice-models catalogue | **ASSERT WITH CAVEAT** — community consensus, not peer-reviewed, but corroborated by 30,000+ shipped models |
| 5.4 | RVC keeps a FAISS index of **all** training feature vectors and retrieves top-K=8 at inference, rather than compressing identity into one embedding | Annotated-RVC code walkthrough | **SAFE TO ASSERT** |
| 5.5 | This architectural difference explains why per-artist conversion beats zero-shot prompting on identity | 5.4 + 2501.13870v1 ("a single fixed-size embedding does not suffice to capture how the target speaker performs various singing techniques") | **ASSERT WITH CAVEAT** — **contested by B5-09**, where Seed-VC zero-shot beat RVC on SECS while losing on DNSMOS. Present the reconciliation, not the simple claim |
| 5.6 | Vocal grit is carried by **subharmonics / period-doubling**, and voices can be perceptually rough with normal jitter and shimmer | Journal of Voice, peer-reviewed | **SAFE TO ASSERT** |
| 5.7 | Singing vocoders assume waveform = periodic + aperiodic, an assumption subharmonic phonation violates | Period Singer 2406.09894; ISMIR 2022 | **SAFE TO ASSERT** for the assumption; **ASSERT WITH CAVEAT** for the conclusion that this is why grit dies — that step is our inference |
| 5.8 | Audio VAEs lose high-frequency fidelity | Stable Audio Open analysis, OpenReview | **SAFE TO ASSERT** |
| 5.9 | Therefore grit does not survive VocalRender's VAE | 5.6 + 5.7 + 5.8 | **DO NOT ASSERT as established.** This is a **hypothesis** built from three confirmed premises. **No paper measures subharmonic preservation through a codec or VAE.** It is testable locally and should be labelled as the report's central open experiment |
| 5.10 | The hybrid generate-then-convert pipeline is standard practice, and conversion should come last | AICoverGen, AICoverMaker, community reports | **ASSERT WITH CAVEAT** — existence and ordering are well supported; **no published quality numbers** |
| 5.11 | Suno/Udio personas hold a consistent voice | Practitioner blogs | **DO NOT ASSERT.** No measured evaluation exists anywhere |

## Q6 — Feasibility and legal

| # | Finding | Primary source | Verdict |
|---|---|---|---|
| 6.1 | 0.4 h of target audio is above every published few-shot cloning threshold | GPT-SoVITS (1 min), few-shot literature (2–10 min), 2309.00284 (15 min SVS finetune) | **SAFE TO ASSERT** |
| 6.2 | Cleanliness matters more than duration past ~30 min | so-vits-svc guidance + industry | **ASSERT WITH CAVEAT** — practitioner consensus, not peer-reviewed |
| 6.3 | No competitive SVS/TTS model has been **pretrained** on a single ≤24 GB consumer GPU | Multiple framings searched; Fish-Speech 8×H100·week, Muyan-TTS 1.34K A100-h, LAPS-Diff 2 GPU-days on an **A100 80 GB** | **SAFE TO ASSERT** as a negative result |
| 6.4 | 12 GB is workable for LoRA/QLoRA finetuning at a ~1.3–2× wall-clock penalty | RTX 4060 profiling + vendor guides | **ASSERT WITH CAVEAT** — **all figures are text-LLM benchmarks.** Codec-LM SVS has different activation profiles; transferring the percentages is an assumption |
| 6.5 | 12.7 h is 2–3 orders of magnitude below where TTS quality emerges (1,000→10,000 h transition) | BASE TTS 2402.08093 | **SAFE TO ASSERT** |
| 6.6 | Historically data won over architecture, and the recurring bottleneck across Vocaloid → UTAU → DiffSinger → codec-LM is aligned annotated singing data | SVS survey + primary papers | **ASSERT WITH CAVEAT** — a historical reading, defensible but interpretive |
| 6.7 | Private, non-commercial experimentation is outside EU AI Act scope | European Commission Art. 50 FAQ (primary) | **SAFE TO ASSERT.** Note Art. 50 applies **from 2 Aug 2026** |
| 6.8 | The ELVIS Act reaches **tool/model distribution** where the primary purpose is producing one identifiable person's voice, and exempts scholarship | TN Code §47-25-1105(a)(3), §47-25-1107, primary text | **SAFE TO ASSERT** for the text; **ASSERT WITH CAVEAT** for application — **no reported decisions in two years; the prong is judicially untested** |
| 6.9 | Copyright is not the exposure vector; state right of publicity is | *Lehrman v. Lovo*, 790 F.Supp.3d 348 (S.D.N.Y. 2025) | **SAFE TO ASSERT** |
| 6.10 | NO FAKES has **no general personal-use carve-out** | EFF + Holland & Knight, enumerated exceptions | **SAFE TO ASSERT — and this refutes the natural assumption.** Exceptions are speech-purpose-based (news, commentary, criticism, scholarship, satire, parody, fleeting use), not use-scale-based |
| 6.11 | NO FAKES creates liability for distributing a tool primarily designed to produce replicas of a specific individual | Sec. 2(c)(2)(B) via EFF | **ASSERT WITH CAVEAT** — S.4591 primary text was **not obtainable** (403); relies on secondary quotation |
| 6.12 | NO FAKES is law | — | **DO NOT ASSERT.** As of Aug 2026 it is reported out of Senate Judiciary (18 Jun 2026) and awaiting the full Senate |
| 6.13 | ShareAlike propagates to a trained adapter | CC "It depends" vs Guadamuz | **DO NOT ASSERT either way.** Genuinely unresolved and unlitigated. The actionable read: **publishing** weights is the risk-bearing act; keeping them local is not |

---

## Summary counts

| verdict | count |
|---|---:|
| SAFE TO ASSERT | 41 |
| ASSERT WITH CAVEAT | 27 |
| DO NOT ASSERT | 13 |
