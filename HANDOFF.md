# VocalRender — English / Amy Winehouse work: handoff

Session of 2026-08-09. Everything below was measured on this machine, not assumed.
Where a number is uncertain or a comparison is unfair, it says so.

---

## 1. Goal

Two goals, and the evidence says they are **not** the same project:

1. **Train VocalRender to sing English well, for many singers.** Achievable. The
   recipe is known (see §7). This is a build.
2. **Get a convincingly Amy Winehouse voice.** Five methods have now failed this
   by measurement (§4). This is a research bet, not a build.

Do not conflate them. Prior sessions repeatedly did, and the notes in
`~/.claude/projects/C--Windows-System32/memory/` document the cost.

---

## 2. The problem in one paragraph

VocalRender renders a symbolic score (words + MIDI pitch + note durations + BPM)
into 48 kHz singing, taking timbre zero-shot from a 2–8 s prompt clip. The
released checkpoints were finetuned and evaluated **only on Mandarin**, so
English comes out with Chinese-accented, mangled vowels. The base model
underneath (VoxCPM2) is a ~30-language speech model, so English phonetics exist
in pretraining but were never reinforced for singing. The author confirms this
directly on HuggingFace discussion #1: *"We just train the model on Chinese
because of data and computing limitation. You may train your own checkpoint with
target language."*

---

## 3. What exists and works

Environment: `.venv` (Python 3.10), torch 2.10.0+**cu128**, RTX 4070 12 GB.

| Path | What it does |
|---|---|
| `scripts_amy/annotate_amy.py` | Any folder of clean English vocal stems → VocalRender score JSON. faster-whisper word timings + RMVPE F0, segmented into notes. **This is the reusable asset** — it turns any acapella corpus into training data. |
| `scripts_amy/convert_gtsinger.py` | GTSinger English → VocalRender schema. Tempo from sibling `.musicxml`. Skips `Paired_Speech_Group` (spoken, not sung). |
| `scripts_amy/split_train_val.py` | Song-grouped train/val split, greedy best-fit to a segment quota. Grouped by song so no lyric/melody leaks across the split. |
| `scripts_amy/eval_wer.py` | Synthesize → whisper → Levenshtein WER. Imports the real inference path so the metric can't drift. Supports `--lora_dir`. |
| `scripts_amy/eval_melody.py` | **Written, never run.** DTW-aligns rendered F0 against the written score → semitone error. Answers "is it singing the right notes". |
| `scripts_amy/minimax_gen.py` | MiniMax music-3.0 generation. Key at `C:\Users\Rodney Franklin\.minimax-key.txt` (`sk-` line). |
| `scripts_amy/get_gtsinger.py` | Resumable, filtered GTSinger download. |
| `scripts_amy/fix_bpm.py` | Folds double-time tempo estimates and re-quantizes note tokens from stored second-durations. |
| `scripts/infer_vocalrender_svs_single.py` | **Patched** — added `--lora_dir` to load a LoRA card over the base. |

Data ready to train on:

```
data/gtsinger_en/annotations_train.json   4630 segs  (12.7 h, 3 singers, 65k words)
data/gtsinger_en/annotations_val.json      180 segs  (3 held-out songs)
data/amy/annotations_train.json            171 segs  (~0.4 h, 31 songs)
data/amy/annotations_val.json               15 segs  (Back-To-Black, Fuck-Me-Pumps)
data/preprocessed_en/                      4996 samples, Arrow latents, 328 MB
```

Trained adapter: `checkpoints/svs_en_lora/step_0001000/lora_weights.safetensors`
(18 MB, r=16, LM+DiT).

---

## 4. Measured results — the important section

### English intelligibility (WER) — ⚠️ RETRACTED 2026-08-10, see below

**The numbers in this subsection do not survive replication. Do not quote them.**
Full analysis: `storm-reports/prompt-conditioned-svs-english/experiment-02-wer-variance.md`.

Both figures below were **single runs**. Re-measured at n=3 per condition in one
session, this eval has sd ≈ 3–14 points, and all conditions overlap:

| condition | mean WER | sd | runs |
|---|---:|---:|---|
| Released checkpoint (base) | **45.86 %** | 3.36 | 43.26 / 49.65 / 44.68 |
| + English LoRA v1 (r=16, step 1000) | **41.14 %** | 13.81 | 56.74 / 36.17 / 30.50 |
| + English LoRA v2 (r=32, step 2000) | **42.55 %** | 8.51 | 34.04 / 42.55 / 51.06 |

Welch p: base vs v1 0.62, base vs v2 0.58, v1 vs v2 0.89. **Nothing significant.**

The original 44.68 % reproduces exactly as one of three base draws, and 29.79 %
is below the lowest of three v1 draws — it was the favourable tail. The claimed
15-point gain is an artefact of one draw against another; the defensible
estimate is ~3–5 points and is not resolvable at n=3.

Original (retracted) table, kept for the record:

| | WER |
|---|---|
| ~~Released checkpoint~~ | ~~44.68 %~~ |
| ~~+ English LoRA @ step 1000~~ | ~~**29.79 %**~~ |
| Paper, Mandarin, Opencpop | 4.44 |

Qualitatively it stopped collapsing into Mandarin syllables:
`"but eli pai pai ai ai ah tai pai re re dai ba"` → `"it does a pi pi man like tiny pin rolling up"`.

Caveat: reference lyrics came from whisper on Amy's originals, so reference
errors are baked in — the true figure is better than it reads. Also only
141 reference words; small sample. The Mandarin 4.44 is likely character-level,
so it is indicative, not like-for-like.

### Voice identity (mee gpt-audio judge, n=3, calibrated Amy anchor)

| audio | score |
|---|---|
| **Real Amy — "You Sent Me Flying"** | **8.0** (8/8/8) |
| **Real Amy — "Some Unholy War"** | **8.0** (8/9/8) |
| MiniMax music-3.0, prompt "Amy Winehouse" | 3.0 |
| VocalRender + English LoRA | 3.0 |
| VocalRender + LoRA, 8.6 s prompt | 3.0 |
| VocalRender base (v1, v2) | 3.0 / 3.0 |

The real-Amy controls separating at 8 prove the judge discriminates — the flat
3.0 is a genuine verdict, not the saturation failure seen with RVC previously.
Every critique is the same: *"smoother, lighter, lacks the grit, rasp and weight."*

~~**The LoRA moved WER by 15 points and voice identity by exactly zero.**~~
**Corrected 2026-08-10:** the WER half of this sentence is withdrawn — the
15-point figure is not reproducible (see the retraction above). The voice-identity
half stands, and is now explained rather than merely observed: in a
prompt-conditioned architecture timbre is a *conditioning input*, not a
weight-resident property, so a LoRA on LM+DiT is topologically incapable of
moving it. MiniMax-Speech states outright that it uses LoRA for emotion control
and not for timbre; UtterTune independently reports speaker similarity pinned at
~0.69 across every variant while accent correctness moved 0.472 → 0.975.
Zero identity movement was the expected result, not a bug.

### Training

`conf/svs_train_en.yaml`, stopped at step 1370/3000 because val plateaued:

```
val   0: 0.6262      val 500: 0.5112
val 250: 0.5148      val 750: 0.5100
                     val 1000: 0.5150     val 1250: 0.5132
```

`loss/stop` reached 0.0001 (solved). `loss/diff` flat from step 250 on.

---

## 5. Known bugs / traps — read before touching anything

1. **151M trainable params are never saved.** Training reports 160,288,768
   trainable but `lora_weights.safetensors` holds only 9,043,968. The difference
   is 73,850 × 2048 = the token embedding, which is resized *after* the model
   freezes non-LoRA weights and so comes back `requires_grad=True`. The save
   path filters on `"lora_" in key`, dropping it. **Fix before the next run:**
   freeze embeddings explicitly after resize, or save them alongside the card.
   The current card is therefore incomplete, and 29.79 % WER was achieved
   *despite* that — there is headroom.
2. **`train_precision: amp_bf16` will not fit.** It means fp32 params: 8.70 GiB
   of weights, 12.56 GiB reserved on a 12 GB card, driver spills to system RAM,
   **524 s/step**. Use `bf16` (true bf16 params): 4.33 GiB, 6.95 GiB reserved,
   **6.5 s/step**. 80× difference.
3. **`num_workers` must be 0 on Windows.** DataLoader uses spawn;
   `CharTokenizerWrapper` is defined inside `mask_multichar_chinese_tokens()`
   and cannot be pickled. Linux forks and never hits this.
4. **`prompt_max_frames` counts patches, not latent steps.** 1 patch = 4 latent
   steps at 25 Hz = 0.16 s, so the default **50 = 8 seconds**, already the top of
   the paper's 2–8 s range. Raising it does nothing. (I got this wrong once.)
5. **`OPENAI_API_KEY` in the machine environment is dead (401)** and *shadows*
   the good key in `Music-Eval-Ensemble/.env`, because mee reads env first.
   Worse, mee swallows the exception and **caches the null result** — delete
   `data/cache/voice_consensus/<stem>_n<N>.json` before re-running.
6. **GTSinger paths blow past Windows MAX_PATH** under a deep repo dir. It is
   downloaded to `C:\gts` for that reason. Keep it there.
7. **The 3090 is not visible to `nvidia-smi`** — only the 4070. Any advice
   assuming 24 GB is wrong on this box today.
8. Chinese-token masking only touches pure-CJK tokens (`\u4e00`–`\u9fff`),
   so it does **not** interfere with English. Checked.

---

## 6. Licensing — matters if this ever leaves the house

GTSinger is **CC BY-NC-SA 4.0**: non-commercial *and* share-alike. Anything
trained on it is research/personal only, and share-alike arguably reaches a
distributed adapter. Amy's stems are a separate and more obvious problem. Fine
for private experimentation; a hard blocker on anything sellable.

---

## 7. The recipe, from the paper (arXiv 2607.27768)

This is the single most useful thing found this session.

- **2,300 h total, of which ~2,013 h (87 %) is SYNTHETIC** — the "Muse" corpus is
  **SunoV5-generated**. Real data is MuChin 158 h + SongFormDB 93 h +
  OpenSinger 53 h.
- **Two stages:** 40k steps pretraining on synthetic → 20k steps finetuning on real.
- Batch **32,768 continuous tokens**, AdamW + DeepSpeed ZeRO-3, lr 1e-4 (halved
  for finetune), 5k warmup, **4×H100, ~1.5 days**.
- Prompt audio: random 2–8 s segment **from the same song**.
- Metrics: WER 4.44, **SIM 0.922** (speaker similarity), RPA 0.72 (melody),
  IOU 0.62 (rhythm), SingMOS 4.59.
- Ablation: removing CrawlSinger-OS costs +0.12 WER, >0.03 SIM, >0.4 RPA.
- **No ablation on singer count or synthetic ratio.** Nobody has published the
  number you'd want.

Our config runs 16,384 tokens/batch — half theirs — on ~1/20th the compute.

---

## 8. What to do next

**First, in the new session:** run `/storm-deep-research`. This session's
WebSearch budget hit its 200-call cap (spent by an earlier `/deep-research`
run), so the research the user actually asked for never happened. WebFetch
still worked, which is how the paper above got read. Questions worth asking:

- How much synthetic singing data before diminishing returns, and does synthetic
  pretraining transfer across languages?
- Has anyone scaled a prompt-conditioned SVS to a *held-out* artist, and what
  SIM value corresponds to human-convincing timbre? (Paper's 0.922 is
  in-domain Mandarin.)
- Is there a published English SVS finetune recipe for a Chinese-trained model?
- Better English singing corpora than GTSinger's 3 singers.

**Goal 1 — English, many singers (do this):**
1. Fix bug §5.1 (freeze or save embeddings), retrain, beat 29.79 % WER.
2. Run `eval_melody.py` — melody accuracy is unmeasured and the user's complaint
   was "not melodic", not just "wrong words". Paper's own RPA is only 0.72.
3. Build the synthetic corpus the way the authors did: MiniMax/ACE-Step generate
   English singing at volume → Demucs isolate vocals → `annotate_amy.py` →
   pretrain, then finetune on the real 13 h of GTSinger.
4. Raise `grad_accum_steps` 8 → 16 to match the paper's 32,768-token batch.

**Goal 2 — convincing Amy (be honest about this):**
Measured against a judge that puts real Amy at 8: ACE-Step 4–5, SEED-VC 4,
RVC 4, MiniMax 3, VocalRender+LoRA 3. Prompt length doesn't move it; the English
LoRA doesn't move it. Timbre rides on ~8 s of VAE latents, which is a thin
channel. Untried levers are ElevenLabs PVC and better source data — *not*
another LoRA. Do not distill MiniMax hoping for Amy timbre: it scores 3/10, so
you would be distilling a 3/10 voice.

---

## 9. How to reproduce the key commands

```bash
cd "C:/Users/Rodney Franklin/Development/personal/VocalRender"

# render a score (add --lora_dir to apply a card)
.venv/Scripts/python.exe scripts/infer_vocalrender_svs_single.py \
  --ckpt_dir pretrained_models/VocalRender \
  --lora_dir checkpoints/svs_en_lora/step_0001000 \
  --json_file examples/smoke_and_honey_v2.json --item_name v2_03_hook \
  --prompt_audio "C:/ai/local-llm/seed-vc/_amy_train_data/06-Love-Is-A-Losing-Game__chunk005.wav" \
  --cfg_value 3.0 --output outputs/test.wav

# WER
.venv/Scripts/python.exe scripts_amy/eval_wer.py \
  --ckpt_dir pretrained_models/VocalRender --lora_dir checkpoints/svs_en_lora/step_0001000 \
  --json_file data/amy/annotations_val.json \
  --audio_root "C:/ai/local-llm/seed-vc/_amy_train_data" \
  --out_dir outputs/wer_test --separator "__" --group_indices 0 --limit 20

# preprocess + train
.venv/Scripts/python.exe scripts/preprocess_svs_data.py conf/svs_preprocess_en.yaml
.venv/Scripts/python.exe scripts/train_vocalrender_svs.py --config_path conf/svs_train_en.yaml

# voice judge (note the key override and cache deletion)
cd "../Music-Eval-Ensemble"
KEY=$(grep "^OPENAI_API_KEY" .env | cut -d= -f2- | tr -d '"'"' \r')
rm -f data/cache/voice_consensus/<stem>_n3.json
OPENAI_API_KEY="$KEY" .venv/Scripts/python.exe mee_judge_cli.py \
  --mode voice --audio <file.mp3> -n 3
```

Always judge with a **real Amy control in the same batch** — it is the only way
to know the judge is discriminating rather than saturated.

---

## 10. Audio to compare

`outputs/smoke_and_honey.wav` (v1, first attempt, Aeolian melody, no LoRA) ·
`_v2.wav` (blues/Dorian rewrite, whole-word tokens, cfg 3.0) ·
`_v3_lora.wav` (v2 + English LoRA) · `_v4.wav` (v3 + 8.6 s prompt).
MP3s for judging in `outputs/judge/`. User's verdict on v3: not Chinese any
more, but "not melodic and not the smokey bar scene Amy was known for" — the
first half is confirmed by WER, the second half by the 3.0 voice score.
