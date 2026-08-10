# Source ledger — VocalRender English path forward

116 source-claim rows across 7 branches. 88 distinct queries. Status key as in
the prior run: `confirmed` / `corrected` / `contested` / `demoted` / `rejected`.

---

## C1 — What the authors say (primary sources)

| id | url | type | quality | claim | status | notes |
|---|---|---|---|---|---|---|
| C1-01 | github.com/pymaster17/VocalRender | primary repo | strong | Inits from **VoxCPM2**; tokenizer is VoxCPM2's `LlamaTokenizerFast` **extended with 128 pitch + 12 note-duration + 256 BPM tokens**; embeddings resized at train start | confirmed (code read) | **Lyrics enter as raw BPE subword text. No phoneme/pinyin/G2P step in the model path** |
| C1-02 | huggingface.co/pymaster/VocalRender | primary | strong | "The released checkpoints primarily target Mandarin Chinese singing." Tag `language: zh` only | confirmed | English not listed |
| C1-03 | .../discussions/1 | **author reply** | strong | pymaster: *"We just train the model on Chinese because of data and computing limitation. You may train your own checkpoint with target language."* | confirmed verbatim | **The only author statement on multilingual that exists** |
| C1-04 | .../issues | primary | strong | Exactly **1 issue ever**, opened 2026-08-10, **0 comments** | confirmed | Authors have never replied |
| C1-05 | same repo | primary | strong | **Zero PRs, no Discussions activity, no docs/ English material** | confirmed (negative) | |
| C1-06 | commit history | primary | strong | 16 commits, all docs/demo/init. **No English/multilingual work** | confirmed (negative) | |
| C1-07 | arxiv.org/abs/2607.27768 | primary | strong | Only **v1**. No revision | confirmed | |
| C1-08 | arxiv html v1 | primary | strong | Stated limitation is score-transcription mismatch. **Language is not mentioned as a limitation at all** | confirmed | |
| C1-09 | same | primary | strong | Pro: 160k steps, 32,768-token batch, LR 1e-4, 5k warmup, 4×H100 ~4 days. Base: 40k synthetic pretrain + **20k real finetune at halved LR**. Full FT, no LoRA, nothing frozen mentioned | confirmed | Closest thing to an author recipe |
| C1-10 | same | primary | moderate | SOFA + G2PW used for **data-prep alignment** (Mandarin polyphone disambiguation), not model input | confirmed | An English pipeline needs an English aligner, not a model change |
| C1-11 | hf datasets/pymaster/CrawlSinger-OS | primary | strong | `language: ["zh"]`, 2,300 h, ~110 GB. **No English component declared** | confirmed | |
| C1-12 | .../discussions | primary | strong | **0 discussions** | confirmed (negative) | |
| C1-13 | HF API search=VocalRender | primary | strong | Only two other repos, **byte-identical mirrors, 0 downloads**. **No English VocalRender checkpoint exists anywhere** | confirmed (negative) | |
| C1-14 | HF API search=VoxCPM2 | secondary | strong | ~70 derivatives; language LoRAs for Punjabi, Indic, Khmer, Nepali, Thai, Hungarian, Krio, Akan, Yoruba, Kazakh — **none for singing** | confirmed | LoRA language adaptation of this base is a working, common pattern |
| C1-15 | github.com/OpenBMB/VoxCPM | primary (base) | strong | VoxCPM2 is **tokenizer-free**, 30 languages **including English**, no language tag needed; supports full SFT and LoRA | confirmed | **The base already speaks English. This is not a vocabulary problem** |
| C1-16 | conf/voxcpm_v2/voxcpm_finetune_lora.yaml | primary | strong | Upstream default: **r=32, alpha=32, dropout 0, lm+dit true, proj false, LR 1e-4, warmup 100, 1000 iters**, batch 2 × accum 8, max_batch_tokens 8192, λ_diff=λ_stop=1.0 | confirmed | **Our "plateau by 1000 steps" matches the vendor's default schedule** |
| C1-17 | VoxCPM issues/195 | **maintainer** | strong | 9,000 h Thai+English full FT: maintainer states **stop-loss converges far faster than diff loss → model fails to stop**; fix = raise λ_stop, audit trailing silence, use retry mechanism | confirmed | **Exactly our symptom** |
| C1-18 | VoxCPM issues/178 | **maintainer** | strong | 1,000 h Japanese FT: *"training exclusively on Japanese likely causes the model to confuse kanji/hanzi and forget Chinese pronunciation"* → **mix in Chinese data**; build **code-switched utterances** | confirmed | Strongest actionable author-side guidance found |
| C1-19 | VoxCPM issues/306 | user | moderate | Catalan SFT 45k clips, 90/10 Catalan/English, LR 2e-5, `freeze_code_predictor: true` → target language **overridden by nearest pretrained accent (Spanish)**. Never answered | confirmed, unanswered | Accent-bleed failure mode |
| C1-20 | VoxCPM issues/15, /33 | maintainer | strong | "We plan to support more languages"; gated on data collection | confirmed | About VoxCPM, not VocalRender |
| C1-21 | VoxCPM issues/352, /357, /213 | primary | moderate | Open bugs: **EOS failure / gibberish at end of audio in Vietnamese**, hallucination on short Polish, word-end cutoff Polish | confirmed | Non-Chinese EOS failure is a known unfixed class |
| C1-22 | Chinese-language search (中文) | secondary | strong | No Zhihu/Bilibili/WeChat article exists; `README.zh-CN.md` is a translation only | confirmed (negative) | |

## C2 — Community cross-lingual results

| id | url | type | quality | claim | status | notes |
|---|---|---|---|---|---|---|
| C2-01 | voxcpm.readthedocs.io/finetuning/faq | vendor doc | strong | New language: **full finetune, 500+ h mixed with ZH/EN, LR 1e-5**. LoRA LR 1e-4, **rank 32–64 for language adaptation**; keep `training_cfg_rate=0.1`; single-speaker 1–3 epochs, "training beyond that often hurts" | confirmed | Conflicts with the "5–10 min" claim, which is speaker-level not language-level. **Says nothing about vocab expansion** |
| C2-02 | VoxCPM issues/114 | GH | weak | Direct ask: new languages, how many hours? **No maintainer response** | rejected (negative) | |
| C2-03 | hf openbmb/VoxCPM2/discussions/15 | HF | weak | **Zero community VoxCPM/VoxCPM2 non-native-language checkpoints or result reports exist** | confirmed (negative) | |
| C2-04 | hf o6Dool/JP_CosyVoice2_finetune | HF card | weak | CosyVoice2 on JVS (~30 h, 100 speakers) Japanese | contested | Checkpoint exists, **no steps/LR/metrics** |
| C2-05 | horstmann.tech CosyVoice2 | practitioner | weak | ~100–500 monolingual h; **LM finetune is the key driver**, vocoder+tokenizer can stay frozen | **unverified (HTTP 403)** | Snippet only |
| C2-06 | arxiv 2412.10117 | paper | strong | CosyVoice2 CER: **Japanese 18.79 %, Korean 7.98 %** | confirmed | Even a well-resourced multilingual model degrades on under-covered languages |
| C2-07 | github.com/SWivid/F5-TTS | repo | strong | Tokenizer 'pinyin'/'char'/'custom'; vocab 2546; new language needs a **new vocab.txt**. **Only full finetune supported — no LoRA** | confirmed | Vocab file is a mandatory explicit artifact |
| C2-08 | F5-TTS issues/988 | GH | weak | Latin-script African langs vocab question, "help wanted", never answered | rejected (negative) | |
| C2-09 | F5-TTS discussions/453 | GH | weak | Czech from scratch **failed**; "2 hours is not enough"; residual failure was **phonetic** ("wrong reading of J", "skipping ě") | confirmed failure | Failure mode was frontend, not fidelity |
| C2-10 | github.com/anhnh2002/XTTSv2-Finetuning-for-New-Languages | community repo | moderate | Explicit recipe: `--extended_vocab_size 2000`, GPT LR **5e-6**, batch 8, accum 4, 5 epochs; DVAE skippable at ≥20 h. **Finetuning HiFiGAN made it worse** | confirmed | Vocab extension is step 1, unskippable |
| C2-11 | coqui-ai/TTS issues/3992 | GH | weak | Same author: "works well with over 100 hours" | confirmed quote | Only concrete XTTS threshold |
| C2-12 | hf coqui/XTTS-v2/discussions/43 | HF | weak | Persian hours question closed with no answer | rejected (negative) | |
| C2-14 | QwenLM/Qwen3-TTS issues/27 | GH | moderate | **15 h Turkish single speaker → "just unintelligible noise."** No maintainer diagnosis | confirmed failure | **Catastrophic, not graceful** — implies frontend/tokenizer mismatch |
| C2-15 | HF/GH snippets | forum | weak | ~60 h Indian-English finetune "did not resemble English at all"; claim that a custom tokenizer **may require training from scratch** | **unverified** | Most load-bearing unverified claim in the branch |
| C2-16 | github.com/RVC-Boss/GPT-SoVITS | repo | strong | English G2P = **CMUdict + neural OOV fallback**; 1 min trains a usable voice | confirmed | 1 min is *speaker* adaptation on a model that already has an English frontend |
| C2-17 | GPT-SoVITS issues/901 | GH | weak | **BERT embeddings used for Chinese, zero-padded for English/Japanese** | confirmed mechanism | Chinese-first model with a structurally blanked English path |
| C2-18 | ar5iv 2309.14089 (BiSinger) | ASRU 2023 | **strong** | Shared CMUdict + pinyin→CMU mapping + language-ID token. Bilingual data **manufactured by SVC** of NUS-48E English singers into M4Singer speakers. Improved English and code-switch SVS **without degrading Chinese** | confirmed | **The direct precedent for VocalRender** |
| C2-19 | RUG thesis 668 + DiffSinger demo | MSc thesis | moderate | English DiffSinger → German via **PHOIBLE phoneme mapping**; base 3 h, finetunes at **30 min and 15 min**, "comparable or better" than large monolingual training. At 15 min, **data quality dominates** | contested (numbers unverified) | **Smallest successful cross-lingual SVS on record** |
| C2-20 | Transinger (semanticscholar) | paper | moderate | IPA as unifying frontend for cross-lingual SVS | **unverified (fetch failed)** | Third independent IPA-unification precedent |
| C2-21 | github.com/Soul-AILab/SoulX-Singer | repo+paper | strong | **>42,000 h** vocal data; natively **Mandarin, English, Cantonese**; MIDI- or melody-conditioned. **Inference-only — no training code released** | confirmed | Baseline/alternative, not a finetune target |
| C2-22 | ace-step + HF chinese-rap-LoRA | repo+HF | moderate | LoRA finetuning of the DiT is officially supported and routine; **no LoRA whose purpose is ZH→EN language transfer, no before/after numbers** | confirmed | Community LoRAs target genre/timbre, not language |
| C2-23 | index-tts issues/448, /501 | GH | weak | Other-language finetune request, no maintainer reply | rejected (negative) | |
| C2-24 | Spark-TTS issues/73 + community fork | GH | weak | Unofficial multilingual finetune repo exists, **no results/settings** | demoted | |
| C2-25 | gokhaneraslan/chatterbox-finetuning | repo | moderate | 23 languages with **"smart vocabulary extension"** as a named first-class feature | confirmed | Fourth toolkit where vocab extension is a required step |
| C2-26 | MIT Press CL vocab-expansion + forgetting lit | peer-reviewed | strong | **Embedding initialization is critical** in low-resource vocab expansion; forgetting mitigated by mixing original-language data, regularization, merging | confirmed (LLM domain) | Mechanistic justification for VoxCPM's mixing advice |

## C3 — Evaluation methodology

| id | url | type | quality | claim | status | notes |
|---|---|---|---|---|---|---|
| C3-01 | arxiv 2607.27768 | primary | strong | VocalRender WER 4.44 measured with **Qwen3-ASR**, Mandarin; 10 diffusion steps, CFG 2.0. **No seed, no run count, no per-benchmark item count stated.** References are ASR-derived | confirmed | "Matching the paper" is not a defensible bar |
| C3-02 | seed-tts-eval | benchmark repo | strong | **~2,000 zh + ~1,000 en** items; Whisper-large-v3 (en), Paraformer (zh) | confirmed | Industry-standard objective eval is 1k–2k items, not 15 |
| C3-03 | arxiv 2410.06885 (F5-TTS) | paper+code | strong | LibriSpeech-PC subset = **1,127 samples**; reports **the average of three random-seed runs** for their model and baselines | confirmed | Strongest precedent for multi-seed averaging |
| C3-04 | VALL-E 2 / Voicebox | papers | strong | Test subsets ~1,234 utterances; **Voicebox reports WER variance 0.005** across runs | confirmed | Rare explicit variance reporting |
| C3-05 | SoulX-Singer-Eval | paper+repo | strong | GMO-SVS = **802 samples**; SoulX-Singer-Eval = **100 segments / 50 unseen singers**, **manual** melody annotation; references are **human sentence-level annotations**, ASR reconciled against them | confirmed | Even its small set is 100 items with manual references |
| C3-06 | StyleSinger / TCSinger | papers | strong | Use MCD, FFE, singer cosine + MOS — **no WER at all** | confirmed | WER is not the historic SVS intelligibility metric |
| C3-07 | DiffSinger / WeSinger | papers | moderate | Objective = MCD, F0 RMSE, semitone accuracy; MOS over 30 groups / 20 raters | confirmed | No WER |
| C3-08 | arxiv 2306.14422 (SVCC 2023) | challenge | strong | ~24 song phrases per source singer, but **12,720 EN / 38,160 JP ratings** | confirmed | Power comes from the **rater** dimension, which we have none of |
| C3-09 | arxiv 2509.15629 (SVCC 2025) | challenge | strong | **480 ratings per system**; objective↔subjective SRCC tops ≈0.8 (naturalness 0.930, style similarity **0.746**); authors: objective metrics "cannot yet become a true replacement" | confirmed | Any single objective proxy is weak |
| C3-10 | arxiv 2510.06927 | position paper | strong | **Test-set version alone swings WER 2.63 → 4.22 for the same model**; prompt ordering affects results; "small differences should not be interpreted as genuine performance gains" | confirmed | **Best citation for "your eval config, not your model, produced your deltas"** |
| C3-11 | arxiv 2607.08256 | paper | strong | Same audio, **different evaluator ASR ⇒ reversed system rankings**; recommends reporting under **≥2 ASR families with disjoint lineages** | confirmed | |
| C3-12 | arxiv 1912.09508 + ogunlao/asr_stat_significance | paper+tool | strong | Standard bootstrap invalid (within-speaker correlation); **blockwise bootstrap** gives a consistent variance estimator for the WER *difference*. 1,000 replicates, 95 % CI | confirmed | **The correct machinery, with a ready-made tool** |
| C3-13 | arxiv 2311.13987 (Jam-ALT) | benchmark | strong | On sung audio **Whisper large-v2 = 35.7 %, large-v3 = 35.5 % WER**; English v3 = **37.7 %**; vocal separation generally *degraded* Whisper | confirmed | **Whisper's own floor on singing is the same order as our entire signal** |
| C3-14 | arxiv 2506.15514 + AudioShake | paper+vendor | moderate | Whisper v3 + separation ≈ **44.9 %** on Jam-ALT; LyricWhiz / ALT-specific systems beat raw Whisper | contested | Suggests an LLM-corrected recognizer |
| C3-15 | Jam-ALT | benchmark | strong | Revised **human** references differ from the originals by **11.1 % WER** | confirmed | Quantifies reference-noise floor |
| C3-16 | arxiv 2506.11089, 2311.00430 | papers | moderate | Pseudo-label errors propagate; noisy labels distort objectives | confirmed for training, **extrapolated** to evaluation references | |
| C3-17 | NAACL 2025 Findings + arxiv 2208.12888 | papers | strong | WER "performs badly under low-resource"; **CER and FER more stable**; WER especially unstable on **short utterances** | confirmed | 15 short sung segments is the worst case for WER |
| C3-18 | SoulX-Singer + YingMusic-Singer | papers | strong | Cross-lingual SVS WER spans **Vevosing 0.717 vs SoulX-Singer 0.110** on the same task | confirmed | Real system differences dwarf the 4 pp we were chasing |
| C3-19 | arxiv 2510.26190 (SP-MCQA) | paper | strong | **Low WER does not imply intelligibility**: FishSpeech lowest WER (5.739 %) but worst key-information accuracy (81.19 %); CosyVoice 2 higher WER (9.04 %) but best accuracy (90.40 %) | confirmed | Validated alternative proxy is comprehension-based |
| C3-20 | Wester 2015 / Kirkland 2023 / Wells 2024 | papers | strong | Strong listener-count effect below ~30 listeners; scale wording shifts MOS | confirmed | Sample-size literature exists for **listeners**, not for objective item counts |

## C4 — Which intervention works

| id | url | type | quality | claim | status | notes |
|---|---|---|---|---|---|---|
| C4-01 | arxiv 2509.22727 (DiaMoE-TTS) | preprint | strong | **IPA vs pinyin frontend, all else equal: WER 29–49 % vs 90–93 %; MOS 2.22–3.15 vs 1.19–1.23.** Adapts with **~3 h** per new dialect via **LoRA r=16 + conditioning adapters, backbone frozen** | confirmed (Table 3) | Cleanest frontend ablation found; LoRA suffices at 3 h **when the frontend is unified** |
| C4-02 | ar5iv 2309.14089 (BiSinger) | ASRU 2023 | strong | Shared CMU phone space + language-ID: MOS 3.87→**3.96**. Pitch-shifted speech as pseudo-singing: 3.78→**3.87**. Only **1.91 h real English singing**, expanded by SVC to **38.2 h** across 20 singers, plus 5.68 h English speech → 17.5 h pseudo-singing | confirmed (Table 8) | Frontend gain is real but small (+0.09 MOS); **augmentation does the heavy lifting** |
| C4-03 | PMC12542845 (Transinger) | journal | moderate | IPA decomposed into **letters + diacritics** generalizes to unseen languages | contested (ablation not retrieved) | Direction solid, magnitudes unverified |
| C4-04 | arxiv 2309.12672 (CrossSinger) | ASRU 2023 | strong | IPA unification + conditional LayerNorm on language ID + **gradient reversal to strip singer bias from lyrics** enables cross-lingual singing from monolingual singers | confirmed | **With one singer per language, singer identity confounds language** |
| C4-05 | arxiv 2510.19546 | MRL@EMNLP 2025 | strong | Model-to-data scale ratio is the primary determinant of forgetting; **PEFT/LoRA gives no clear advantage over full FT at preventing forgetting** | confirmed (abstract) | Contradicts "LoRA protects the base" |
| C4-06 | Interspeech 2025 kwon25 | paper | strong | Adapters preserve pretrained language capability | **unverified (PDF would not decode)** | Directional only |
| C4-07 | arxiv 2506.13763 | preprint | strong | **Optimal diffusion loss is nonzero and dataset-dependent**, so flat val loss cannot distinguish "hit capacity" / "converged to the irreducible optimum" / "undertrained." Loss-gap↔FID correlation **changes sign** across noise levels | confirmed (fetched) | **The single most decision-relevant source in this run** |
| C4-09 | Interspeech 2025 wang25s + FlowTTS-GRPO | papers | moderate | Flow/diffusion speech work does **not** use val loss for model selection — panel is WER, UTMOS/DNSMOS/NISQA, speaker-sim, MOS | confirmed | Field norm |
| C4-11 | ar5iv 2408.04303 + WECHSEL | ACL-line | strong | Embedding-only retraining with frozen transformer is established for **monolingual** models, but **"yields poor results when transferring a multilingual model"** | confirmed | **Decisive against embedding-only here — our base is 30-language multilingual** |
| C4-12 | aclanthology 2026.eacl-long.357 | EACL 2026 | strong | Practical middle ground: finetune **embedding table + output head + top-2 and bottom-2 layers**, freeze the middle 28 | confirmed | Concrete recipe between pure-LoRA and pure-embedding |
| C4-13 | arxiv 2511.03310, 2510.00499, 2601.13802 | preprints | moderate | Standard two-stage: **stage 1 freeze backbone, train new components for alignment; stage 2 unfreeze more AND mix original-domain data back in** | confirmed | Bundles the replay fix |
| C4-14 | arxiv 2512.14657 | preprint | strong | 1.7B TTS-pretrained SLM → SVS used **135 h** synthetic singing, LR **5e-6**, full-model ZeRO-2, **no LoRA** | confirmed | 12.7 h is ~10× below |
| C4-15 | arxiv 2402.01520, 2406.02429 | preprints | moderate | Other end: SVC adaptation from **0.41 h**; Karaoker-SSL builds SVS from **speech only, zero singing data** | confirmed | Hours spread 0.4→200 h; **volume is not the discriminator** |
| C4-16 | VoxCPM2 repo/card | official | strong | VoxCPM2 is **2B, 30 languages, tokenizer-free** — consumes raw text, no phoneme frontend, no language tags | confirmed | **Voids intervention A as literally described** |

## C5 — What a flat diffusion loss means

| id | url | type | quality | claim | status | notes |
|---|---|---|---|---|---|---|
| C5-01 | arxiv 2506.13763 | preprint | strong | *"The loss of diffusion models is not indicative of absolute data-fitting quality, since its optimal value is typically not zero but unknown, leading to the confusion between large optimal loss and insufficient model capacity."* | confirmed (direct quote) | 0.4884 is mostly the irreducible term |
| C5-02 | same | preprint | strong | Loss *gap* correlates with FID only in a restricted noise band; **near the critical σ the correlation is negative** | confirmed | Flat aggregate loss can hide improvement *and* regression |
| C5-03 | arxiv 2102.09672 (Improved DDPM) | ICML | strong | On ImageNet 64×64 "FID started becoming worse over the course of training"; recommend early stopping | confirmed | **The pro-stopping citation — but they stopped on FID, not loss** |
| C5-04 | same | ICML | strong | "the gradient of L_vlb was much noisier than that of L_hybrid"; add importance sampling over timesteps weighted by E[L_t²] | confirmed | Mechanism: the objective is high-variance across t |
| C5-05 | same | ICML | strong | FID follows a clean power law in scale; "the NLL curve does not fit a power law as cleanly" | confirmed | Loss and perceptual quality decouple with scale |
| C5-06 | arxiv 2603.10904 | preprint | strong | LoRA on LLM-based TTS, **3.5–18.5 h per speaker**: explicit **loss–quality divergence** — val loss improves monotonically while DNS-MOS **degrades**. *"Checkpoint selection must therefore be guided by perceptual evaluation rather than loss convergence alone."* | confirmed (fetched) | **Closest analogue to our setup; brackets our 12.7 h** |
| C5-07 | same | preprint | strong | Perceptual quality can **recover later** in training as "the frozen backbone reasserts its pretrained acoustic prior" | confirmed | Direct argument against stopping at 2000 |
| C5-08 | arxiv 2510.12995 (attribution) | preprint | weak | "loss decreases monotonically, but evaluation WER first decreases and then increases… early stopping by lowest validation WER" | **unverified — could not locate in the PDF** | Do not cite; practice corroborated by C5-06 |
| C5-09 | arxiv 2411.09998 + OpenReview | papers | moderate | Stochastic-gradient variance differs across timesteps by up to **~100×**; the variance is "an artifact of the training dynamic" | contested (snippets) | With ~32 samples/step the batch mean is stable to 4 dp long before learning stops |
| C5-10 | arxiv 2303.09556 (Min-SNR) | ICCV 2023 | strong | Diffusion training is multi-task with conflicting gradients; **high-SNR timesteps dominate the optimization**; reweighting gives 3.4× speedup | confirmed | The aggregate under-reports progress on the timesteps that matter |
| C5-11 | arxiv 2605.10790 | preprint | moderate | "a total-loss curve can obscure continued learning in the large-noise regime since its value is dominated by the [high-noise] contribution" | contested (snippet) | Most on-the-nose statement of "flat loss ≠ no learning" |
| C5-12 | arxiv 2510.09016 (DiTSinger) | preprint | strong | **100,000 iterations**, 4×A100, ~530 h, eff. batch ~192, 3–7 days; evaluated by MOS/MCD/FFE/F0RMSE — **no loss-based stopping** | confirmed | 2k steps is very short by this field's standards |
| C5-13 | arxiv 2105.02446 (DiffSinger) | AAAI | strong | 30k steps aux + **160k steps** main diffusion "until convergence"; reported metric is MOS | confirmed | Convergence asserted via MOS, never a flat loss |
| C5-14 | unsloth LoRA guide | vendor doc | weak | ">3 epochs gives diminishing returns"; val/train divergence after epoch 1–2 = overfitting | demoted | **Text-LLM guidance; transferring it to an audio diffusion head is the error under investigation** |
| C5-15 | voxcpm docs + GLM-TTS 2512.14291 | docs+report | moderate | ~1 h suffices for voice adaptation; LoRA r=32 ≈ 98 % of full-FT speaker similarity; "smaller datasets overfit faster, optimal checkpoint within a few hundred steps" | contested | Timbre ≠ phonetics/prosody; 12.7 h multi-song is a different regime |
| C5-16 | arxiv 2011.00935 + non-attentive Tacotron | papers | moderate | Stop-token frames are "extremely imbalanced" (one positive per utterance), need positive weight ~100; the Tacotron2 stop head "causes stop early problems" | confirmed | **Our stop head memorised 5,485 lengths; rising val loss there is expected and near-costless** |

## C6 — Alternative bases

| id | url | type | quality | claim | status | notes |
|---|---|---|---|---|---|---|
| C6-01 | r9y9.github.io/projects/nnsvs | project | strong | NNSVS modular, Sinsy-inspired, multi-stream + neural vocoders | confirmed | |
| C6-02 | OpenUtau wiki | wiki | moderate | ML voicebanks need a GPU; **no minimum dataset size documented** | confirmed (negative) | Explicit doc gap |
| C6-03 | ace-step/ACE-Step-1.5 | repo | strong | **MIT**, <4 GB inference, LoRA "8 songs, 1 h on a 3090", **no MIDI input** | confirmed | Their "3090 (12GB)" label is an internal error |
| C6-05 | multimodal-art-projection/YuE | repo | strong | Apache-2.0, commercial OK; **24 GB for 2 sessions, 80 GB+ full songs**; no score input | confirmed | Hardware disqualifies a 12 GB card |
| C6-06 | ICLR YuE paper | ICLR | strong | Matches/surpasses some proprietary systems in musicality | confirmed | |
| C6-07 | hf Soul-AILab/SoulX-Singer | model card | strong | **Apache-2.0, EN+ZH+Cantonese, MIDI score OR F0 conditioning, zero-shot timbre** | confirmed | **1,028 downloads/mo — low adoption** |
| C6-08 | arxiv 2602.07803 | report | moderate | 42k h data, SOTA on **its own** benchmark | contested (self-reported) | No independent replication found |
| C6-09 | AaronZ345/TCSinger2 | repo | strong | MIT, lyrics+notes input, multilingual, **8-GPU training setup** | confirmed | **Pretrained weight release unconfirmed** |
| C6-11 | intunist/nnsvs-english-support | repo | strong | **1 h prototype / 4 h decent / 8 h+ very high quality**; ARPAbet-derived phone set; no English pretrain | confirmed | **Best direct answer to the data-volume question** |
| C6-12 | arxiv 2105.02446 | paper | strong | DiffSinger: 1×V100, ~28 GPU-hours total | confirmed | Comfortably within a consumer GPU |
| C6-13 | openvpi/DiffSinger | repo | strong | Apache-2.0, 44.1 kHz, lyrics+MIDI input, pitch/energy/breathiness control | confirmed | English pretrain not confirmed |
| C6-14 | KVR forum t=626750 | forum | moderate | SynthV English pronunciation superior to ACE Studio; but some words won't pronounce correctly even with tweaking | confirmed (anecdotal) | |
| C6-16 | Plachtaa/seed-vc | repo | strong | **GPL-3.0**, 44.1 kHz singing VC, zero-shot 1–30 s ref, **archived 2025-11-21** | confirmed | License + abandonment both blockers |
| C6-17 | Amphion vevosing | repo | strong | Zero-shot speech+singing imitation, EN supported, **no note-based input** | confirmed | Good for the convert half only |

## C7 — Feasibility / skeptic

| id | url | type | quality | claim | status | notes |
|---|---|---|---|---|---|---|
| C7-01 | RUG thesis 668 | BA thesis | moderate | DiffSinger EN→German at **15–30 min**, matching large-scale-trained models | contested | Single target singer, phoneme-mapped, small acoustic model. **Not open-domain prompt-conditioned generation** |
| C7-02 | arxiv 2409.13832v5 (GTSinger) | NeurIPS D&B | **strong** | GTSinger English = **13.13 h total, only 6.71 h of actual singing, 3 singers** (EN-Tenor-1, EN-Alto-1, EN-Alto-2) | confirmed | **Two of three are altos → effectively 2 timbre classes** |
| C7-03 | same | NeurIPS D&B | strong | Authors concede phoneme-duration segmentation and technique labels "remain challenging for human ears" | confirmed | Annotation noise admitted by the authors |
| C7-04 | arxiv 2505.09325 (SingNet) | preprint | strong | Studio corpora span 0.5–51.8 h, "limited to Pop Songs", "limited to Chinese Singing"; in-the-wild scale (3,000 h) needed for genuine diversity | confirmed | 58× gap to in-the-wild |
| C7-05 | thevocalmarket.com | industry blog | weak | ~5 h single-singer floor; **10–30 speakers / 30–100 h** moderate; **50–200 speakers / 100–500 h** strong. "Adding a new speaker improves generalization more than more data from an existing speaker" | demoted (vendor) | Tiering matches C7-04/C7-07 independently |
| C7-06 | same | industry blog | weak | "Clean English singing data at scale essentially does not exist in the open-source pool"; ~230 h total, heavily Mandarin | demoted | Explains why nobody has published this result |
| C7-07 | arxiv 2510.09016 | preprint | strong | DiTSinger needed **530 h** (30 h human seed + 500 h synthetic) from **40 vocalists**; monotonic improvement 30→530 h | confirmed | The 2026 answer to scarcity is **synthesize more**, not finetune on 12 h |
| C7-08 | arxiv 2512.14657 | preprint | strong | Speech-pretrained codec "lacks the ability to faithfully resynthesize singing, resulting in a **performance upper bound set by the decoder side**"; 135 h still shipped boundary glitches, pitch lag | confirmed | **Ceiling evidence** |
| C7-09 | intunist/diffsinger-english-support | repo | moderate | English support in DiffSinger exists as a community **phoneme dictionary** over a Mandarin-trained core | confirmed | The field's working solution is phoneme mapping, not language finetuning |
| C7-11 | pubmed 40648229 (Transinger) | peer-reviewed | strong | Cross-lingual generalization requires unified IPA; "fragmented, language-specific phoneme encodings hinder unified phonetic modeling" | confirmed (abstract) | **But see C4-16 — VocalRender has no phoneme layer at all** |
| C7-12 | arxiv 2508.07426, 2603.07534 | preprints | moderate | Accent is near-perfectly recoverable from multilingual model internals; mitigation requires **many accents in training** | confirmed | **3 singers cannot supply accent diversity** |
| C7-13 | arxiv 2505.14910 (TCSinger 2) | ACL Findings | strong | Multilingual zero-shot SVS achieved by architecture + **multilingual pretraining**, not post-hoc language finetuning | confirmed | Every published multilingual singing success trains multilingually from the start |
