# Verification audit

Verdicts: **SAFE TO ASSERT** · **ASSERT WITH CAVEAT** · **DO NOT ASSERT**.

## Q1 — What the authors say

| # | Finding | Source | Verdict |
|---|---|---|---|
| 1.1 | The only author statement on other languages is *"We just train the model on Chinese because of data and computing limitation. You may train your own checkpoint with target language."* | HF discussion #1, verbatim | **SAFE** |
| 1.2 | There is no roadmap, no planned English checkpoint, no finetuning recipe, and no reply to the repo's only issue | Repo + HF, exhaustive | **SAFE** as a negative result |
| 1.3 | No English VocalRender checkpoint exists anywhere; the two other HF repos are byte-identical mirrors | HF API | **SAFE** |
| 1.4 | VocalRender feeds lyrics as raw BPE + 396 score tokens; there is no phoneme/pinyin/G2P step in the model path | Code read of `setup_svs_tokenizer.py` | **SAFE** |
| 1.5 | VoxCPM2 is tokenizer-free and covers 30 languages including English | Official repo/card | **SAFE** |
| 1.6 | Upstream VoxCPM2's default LoRA config is r=32, α=32, LM+DiT, LR 1e-4, **1,000 iterations** | `voxcpm_finetune_lora.yaml` | **SAFE** — and it means our "plateau at 1000" matches the vendor schedule rather than indicating failure |
| 1.7 | A VoxCPM maintainer describes stop-loss converging far faster than diffusion loss as a known failure mode, with fixes | Issue #195 | **SAFE** for the statement; **CAVEAT** that their case was 9,000 h full-FT TTS, not a 12.7 h singing LoRA |
| 1.8 | A maintainer recommends mixing original-language data and building code-switched utterances to prevent forgetting | Issue #178 | **SAFE** |
| 1.9 | CrawlSinger-OS contains no English | Dataset card `language: ["zh"]` | **ASSERT WITH CAVEAT** — the card asserts zh; GTSinger upstream is multilingual, so an English fraction cannot be fully excluded without inspecting the data |

## Q2 — Community

| # | Finding | Source | Verdict |
|---|---|---|---|
| 2.1 | Zero cross-lingual VoxCPM/VoxCPM2 finetunes exist publicly — no checkpoint, no writeup, no metrics | GitHub + HF sweep, both maintainer asks unanswered | **SAFE** as a negative result |
| 2.2 | No community English checkpoint exists for any Chinese-origin singing model | Sweep | **SAFE** as a negative result |
| 2.3 | No community report anywhere pairs a pre- and post-finetune metric | Sweep | **SAFE** as a negative result |
| 2.4 | BiSinger is the direct academic precedent: shared CMUdict+pinyin phone space, bilingual data manufactured by singing-voice-conversion | ASRU 2023, fetched | **SAFE** |
| 2.5 | BiSinger used only **1.91 h of real English singing**, expanded by SVC to 38.2 h, plus 5.68 h English speech pitch-shifted into 17.5 h pseudo-singing | Table 8 | **SAFE** — and it is the most encouraging datapoint in the run |
| 2.6 | BiSinger's frontend gain was +0.09 MOS; augmentation did the heavy lifting | Table 8 | **SAFE** |
| 2.7 | A 15 h Turkish finetune produced "just unintelligible noise" | GitHub issue | **ASSERT WITH CAVEAT** — single unreplicated report, no maintainer diagnosis |
| 2.8 | English→German DiffSinger transfer worked at 15–30 min | BA thesis | **DO NOT ASSERT as a number.** PDF unparseable, single target singer, small acoustic model, different task class. Directionally useful only |
| 2.9 | Every successful recipe makes the phone/vocab set an explicit first step, and every failure is phonetic or catastrophic | 4-for-4 in each direction | **ASSERT WITH CAVEAT** — correlational, not causal, and nobody ran the controlled experiment |

## Q3 — Evaluation

| # | Finding | Source | Verdict |
|---|---|---|---|
| 3.1 | At WER≈0.45 on 141 words the binomial SE is ≈4.2 pp per run, and within-utterance clustering makes the true SE larger | Derivation + C3-12 | **SAFE** — arithmetic, and it exactly predicts the observed 29.79/56.74 swing |
| 3.2 | Real papers use 802 / 1,127 / 1,234 / ~1,000–2,000 items | SoulX-Singer, F5-TTS, VALL-E 2, SEED-TTS | **SAFE** |
| 3.3 | F5-TTS reports the average of three random-seed runs | Paper | **SAFE** |
| 3.4 | Most papers — including VocalRender itself — state no seed and no run count | Sweep | **SAFE** |
| 3.5 | Whisper's WER on **sung** audio is 35–45 %, and large-v3 is not reliably better than large-v2 | Jam-ALT, fetched | **SAFE** — this is the most damaging methodological finding for the project |
| 3.6 | Two careful **human** reference versions of the same lyrics differ by 11.1 % WER | Jam-ALT | **SAFE** |
| 3.7 | Different evaluator ASR families can reverse system rankings | arXiv 2607.08256 | **SAFE** |
| 3.8 | Test-set version alone swings WER 2.63→4.22 for the same model | arXiv 2510.06927 | **SAFE** |
| 3.9 | CER is more stable than WER on short utterances and low-resource conditions | NAACL 2025 Findings | **ASSERT WITH CAVEAT** — evidence is from multilingual ASR, extrapolated to singing |
| 3.10 | Low WER does not imply intelligibility | SP-MCQA, ICASSP 2026 | **SAFE** |
| 3.11 | Detecting a 5 pp difference needs ~2,500–3,000 reference words per condition | Own power derivation | **ASSERT WITH CAVEAT** — clearly labelled as our derivation; **no published minimum-eval-size guidance for objective TTS/SVS metrics exists** |
| 3.12 | Blockwise bootstrap is the correct significance test, with a ready implementation | Interspeech 2020 + repo | **SAFE** |

## Q4 — Interventions

| # | Finding | Source | Verdict |
|---|---|---|---|
| 4.1 | A unified IPA frontend beats pinyin by a huge margin: WER 29–49 % vs 90–93 %, MOS 2.22–3.15 vs 1.19–1.23 | DiaMoE-TTS Table 3, fetched | **SAFE** for that system |
| 4.2 | **That fix does not apply here.** Every frontend result assumes a phoneme-token input layer; VoxCPM2 has none | C4-16 + C1-01 | **SAFE** — and it retracts the previous run's headline recommendation |
| 4.3 | Adding a phoneme frontend to a tokenizer-free model is a research project, not a fix | Inference from 4.2 | **ASSERT WITH CAVEAT** — reasonable but untested; nobody has published on frontend intervention for tokenizer-free TTS |
| 4.4 | Embedding-only retraining "yields poor results when transferring a multilingual model" | Trans-tokenization line | **SAFE** — decisive against that intervention here |
| 4.5 | A practical middle ground is embedding + output head + top-2 and bottom-2 layers, freezing the middle | EACL 2026 | **ASSERT WITH CAVEAT** — from the LLM literature, not audio |
| 4.6 | LoRA gives no clear advantage over full FT at preventing forgetting | MRL@EMNLP 2025 | **ASSERT WITH CAVEAT** — MT domain; contradicts widespread folklore, so worth flagging both ways |
| 4.7 | The standard answer when pretraining has the language but not the task is two-stage training with original-domain replay | Three preprints | **ASSERT WITH CAVEAT** — converging but none in SVS |
| 4.8 | There is zero direct evidence that Mandarin singing finetuning damaged VoxCPM2's English | Exhaustive | **SAFE** as a negative — and it is cheaply testable locally |

## Q5 — The loss plateau

| # | Finding | Source | Verdict |
|---|---|---|---|
| 5.1 | The optimal diffusion loss is nonzero and dataset-dependent, so a flat value cannot distinguish saturation from convergence-to-optimum from undertraining | arXiv 2506.13763, direct quote | **SAFE** |
| 5.2 | The loss-gap↔FID correlation changes sign across noise levels | Same | **SAFE** |
| 5.3 | Timestep gradient variance spans up to ~100×, so the batch mean stabilises long before learning stops | Improved DDPM + others | **ASSERT WITH CAVEAT** — mechanism well established; the specific "stable to 4 dp" inference is ours |
| 5.4 | In the closest published analogue (LoRA, LLM-TTS, 3.5–18.5 h/speaker) validation loss and DNS-MOS moved in **opposite directions**, and the authors concluded checkpoint selection must be perceptual | arXiv 2603.10904, fetched | **SAFE** |
| 5.5 | Perceptual quality can recover later in training as the frozen backbone reasserts its prior | Same | **SAFE** for that system |
| 5.6 | No flow/diffusion speech paper selects checkpoints on validation loss | Survey of practice | **ASSERT WITH CAVEAT** — absence across a sample, not an exhaustive proof |
| 5.7 | The rising stop-head loss is an expected pathology of a one-positive-frame classifier, not grounds to halt a diffusion trunk | Stop-token literature | **SAFE** |
| 5.8 | **Stopping the run at step 2000 on flat diffusion loss was a mistake** | Synthesis of 5.1–5.7 | **ASSERT WITH CAVEAT** — the loss carried almost no information, so the decision was unfounded; that does not prove the model *was* still improving |
| 5.9 | Prolonged diffusion training can genuinely degrade quality | Improved DDPM | **SAFE** — the honest counterweight; they detected it with FID, not loss |

## Q6 — Alternatives

| # | Finding | Source | Verdict |
|---|---|---|---|
| 6.1 | SoulX-Singer is Apache-2.0, natively English, MIDI-score-conditioned, zero-shot timbre, 42k h | Model card + report | **SAFE** for the capabilities as documented |
| 6.2 | SoulX-Singer is the best fit for this project's data shape and goal | Comparison | **ASSERT WITH CAVEAT** — on paper. Quality is self-reported, adoption is low, and it is inference-only |
| 6.3 | Its quality claims have no independent replication | Sweep | **SAFE** as a negative |
| 6.4 | NNSVS English support documents 1 h prototype / 4 h decent / 8 h+ very high quality | Community repo | **ASSERT WITH CAVEAT** — single source, no measurements |
| 6.5 | DiffSinger's original training cost ~28 GPU-hours on a V100 | Paper | **SAFE** |
| 6.6 | YuE needs 24 GB for two sessions and 80 GB+ for full songs | Repo | **SAFE** — disqualifies it on a 4070 |
| 6.7 | Seed-VC is archived and GPL-3.0 | Repo | **SAFE** |
| 6.8 | Generate-then-convert throws away the score-annotated data and melody control | Inference | **SAFE** |
| 6.9 | TCSinger 2 has released weights | — | **DO NOT ASSERT.** Unconfirmed |

## Q7 — Feasibility

| # | Finding | Source | Verdict |
|---|---|---|---|
| 7.1 | GTSinger English is **6.71 h of actual singing from 3 singers, two of them altos** | NeurIPS D&B paper | **SAFE** — the single most damaging finding in the run |
| 7.2 | The practitioner's effective data is closer to two voices than to twelve hours | 7.1 | **ASSERT WITH CAVEAT** — a reasonable reading, not a measured quantity |
| 7.3 | Multi-singer generalization in a new language is tiered at 30–100 h / 10–30 speakers minimum | Vendor blog, corroborated by SingNet and DiTSinger | **ASSERT WITH CAVEAT** — the specific tiers come from a vendor with an interest |
| 7.4 | 2026 SOTA answers scarcity by synthesizing 500 h, not by finetuning on 12 h | DiTSinger; matches VocalRender's own 87 %-synthetic recipe | **SAFE** |
| 7.5 | A speech-pretrained codec sets a decoder-side quality ceiling that LM adaptation cannot cross | arXiv 2512.14657 | **SAFE** for that system; **CAVEAT** applying it to VocalRender, whose VAE we measured as adequate for grit |
| 7.6 | Accent is baked into multilingual model internals and is mitigated by accent *diversity*, which 3 singers cannot supply | Two preprints | **ASSERT WITH CAVEAT** — speech-domain |
| 7.7 | Every published multilingual singing success trains multilingually from the start | TCSinger 2, Transinger, SoulX-Singer | **SAFE** |
| 7.8 | Continuing to finetune 2.3B parameters on 6.7 h from two voice types is ruled out by the evidence | Synthesis | **ASSERT WITH CAVEAT** — strong, but it rests on the data-tier claims in 7.3, whose best source is a vendor blog |
| 7.9 | "No abandoned projects were found, so failure is rare" | — | **DO NOT ASSERT.** Those searches never ran; absence here is uninformative |

## Summary

| verdict | count |
|---|---:|
| SAFE TO ASSERT | 38 |
| ASSERT WITH CAVEAT | 24 |
| DO NOT ASSERT | 5 |
