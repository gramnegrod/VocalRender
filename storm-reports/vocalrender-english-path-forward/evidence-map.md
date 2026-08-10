# Evidence map

## 1. What a domain expert would expect that we had not checked

| Expectation | Result |
|---|---|
| That VocalRender has a phoneme frontend to unify | **False.** VoxCPM2 is tokenizer-free; lyrics enter as raw BPE. The IPA fix that five other systems used has no attachment point here (C1-01, C4-16) |
| That GTSinger English is 12.7 h | **False.** 13.13 h total but **6.71 h of actual singing**, 3 singers, two of them altos (C7-02) |
| That a flat validation loss means convergence | **False.** Optimal diffusion loss is nonzero and dataset-dependent; the number is dominated by an irreducible term (C4-07, C5-01) |
| That 1,000 steps is too few | **Probably false.** Upstream VoxCPM2's default LoRA schedule *is* 1,000 iterations (C1-16) |
| That the rising stop-head loss signals overfitting worth stopping for | **Misread.** That head is a one-positive-frame-per-utterance classifier; memorising 5,485 lengths is expected (C5-16). A maintainer describes the same imbalance as a known issue with a known fix (C1-17) |
| That someone has done this before | **Nobody has.** Zero cross-lingual VoxCPM finetunes, zero community English checkpoints for any Chinese-origin singing model (C2-03, C1-13) |
| That better English SVS options don't exist | **False.** SoulX-Singer is Apache-2.0, English-native, MIDI-score-conditioned, zero-shot timbre (C6-07) |

## 2. What source would change the recommendation

- **An independent evaluation of SoulX-Singer's English.** All quality claims are self-reported on its own benchmark, with no replication and ~1k downloads/month (C6-08). If it is as good as claimed, the entire finetuning effort is unnecessary. **This is the highest-value unknown in the report.**
- **Audio evaluation of the existing step-1000/1500/2000 checkpoints.** If they are already better than the flat loss suggested, the "saturates immediately" conclusion collapses (C4-07, C5-06).
- **An English-retention probe of the Mandarin checkpoint.** If the Mandarin singing finetune damaged VoxCPM2's English, the fix is replay, not more English data (C1-18, C4-05).
- **Confirmation that TCSinger 2 weights exist.** It takes lyrics+notes, is MIT, and is multilingual — but release is unconfirmed (C6-09).

## 3. Claims resting on vendors, self-report, or anecdote — flagged

| Claim | Weakness |
|---|---|
| SoulX-Singer SOTA English singing | **Self-reported on its own benchmark**, no independent replication |
| VoxCPM "500+ h for a new language" | Vendor doc, **never publicly validated by anyone** (C2-01, C2-03) |
| "5–10 min adapts speaker/language/domain" | Same doc, contradicts its own 500 h figure; the 5–10 min number is speaker-level |
| NNSVS 1 h / 4 h / 8 h tiers | Single community repo, no measurements published |
| ACE-Step "LoRA on a 3090" | Doc contains an internal contradiction ("3090 (12GB VRAM)") |
| SynthV English superiority | Forum anecdote from practitioners |
| ~230 h of clean public singing exists | Vendor blog with an interest in selling data |
| 15–30 min German DiffSinger transfer | BA thesis, numbers unverified (PDF unparseable) |

## 4. Where sources directly conflict

1. **Data volume needed.** 15–30 min (phoneme-mapped DiffSinger) · 0.41 h (SVC) · 3 h (DiaMoE per dialect) · 20 h (XTTS) · 100 h (XTTS "works well") · 135 h (SLM→SVS) · 500 h (VoxCPM vendor) · 530 h (DiTSinger). And yet **15 h Turkish produced pure noise** and **60 h produced non-English**. Volume alone does not predict outcome; the low-data successes are exactly the ones where a phone inventory was *mapped*.
2. **Does LoRA protect against forgetting?** VoxCPM docs and PEFT folklore say yes; C4-05 finds **no clear advantage over full FT**.
3. **Is the frontend the dominant lever?** C4-01 shows a 2–3× WER swing from frontend alone — but every frontend result assumes a phoneme-token input layer that this model does not have (C4-16). **Strong evidence, zero applicability.**
4. **Should you stop on a loss plateau?** C5-03 (Improved DDPM) genuinely documents diffusion degrading with prolonged training — but detected it with FID. C5-06 documents loss and quality moving in *opposite* directions. Both agree the loss is not the signal.
5. **Is 12 epochs a lot?** Text-LLM guidance says yes (C5-14); SVS practice is 100k–160k steps (C5-12, C5-13).

## 5. Missing stakeholder

Still the **listener** — and now also the **evaluator**. This run's sharpest
methodological finding is that Whisper's own error rate on *sung* audio is
35–45 % (C3-13), which is the same magnitude as the entire signal being
measured. The project has been using a ruler whose graduations are as wide as
the object. Two careful *human* reference versions of the same lyrics differ by
11.1 % WER (C3-15); ASR-derived references are worse than that.

## 6. Adjacent queries that exposed unknown unknowns

- Reading the **base model's** repo rather than VocalRender's revealed the
  tokenizer-free architecture that invalidates the previous run's headline
  recommendation.
- Reading **GTSinger's own paper** rather than trusting the corpus summary
  revealed 6.71 h and two alto voices.
- Reading **VoxCPM's issue tracker** surfaced a maintainer describing our exact
  stop-loss/diff-loss symptom, and another prescribing original-language mixing.

## 7. Time-sensitive

| Fact | Status |
|---|---|
| SoulX-Singer release | Feb 2026, ~1k downloads/mo, inference-only |
| ACE-Step 1.5 / XL | Jan + Apr 2026, MIT |
| Seed-VC | **Archived 2025-11-21**, unmaintained, GPL-3.0 |
| VocalRender repo | 1 issue, 0 replies, 16 commits, no roadmap |
| TCSinger 2 weights | Unconfirmed as of Aug 2026 |

## 8. Gaps accepted as unresolved

- Whether SoulX-Singer's English is actually good (needs hands-on testing).
- Whether the Mandarin finetune damaged VoxCPM2's English (needs a local probe).
- Whether the existing checkpoints already improved (needs audio evaluation).
- Transinger's ablation magnitudes; BiSinger's hours/steps/LR.
- Any measurement of 3-singer vs 30-singer at fixed hours in singing.
- Q2 of the skeptic branch — "projects that died" — went entirely unsearched.
