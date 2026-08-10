# Handoff — session 2, 2026-08-09

**Read `HANDOFF.md` first.** It is the substantive one: measured results, the
recipe from the paper, and the eight bugs/traps in §5 that will cost you a day
each if you skip them. This file only records what happened *after* it was
written, so you don't redo work or lose the thread.

---

## 1. What this session actually did

Almost nothing was executed. The sequence was:

1. Read `HANDOFF.md`.
2. Asked the user which of the three §8 next-steps to start with.
   **User chose `/storm-deep-research`** over (a) fixing the §5.1 embedding-save
   bug and (b) running `eval_melody.py`. That choice still stands — resume there.
3. Invoked the `storm-deep-research` skill in **deep** mode and wrote the plan file.
4. Session was interrupted by a `CLAUDE_CODE_CHILD_SESSION` transcript-saving
   problem (see §4) before a single search ran.

**Zero searches were executed. Zero code changed. Nothing in git differs because
of this session** except the two new files listed in §3.

---

## 2. Where to pick up — the research run

The plan is already written and is the entry point:

```
storm-reports/prompt-conditioned-svs-english/research-plan.md
```

It fully specifies the run: four research questions, 6 search branches
(B1 synthetic-data scaling / B2 held-out-artist timbre + what SIM means /
B3 cross-lingual finetune recipes / B4 English corpora / B5 commercial voice
cloning reality / B6 counterargument + legal), depth 3, budget ≥60 distinct
queries, ≥30 ledger rows, 20–30 citations, plus a written-down blind-spots list.

**To resume:** invoke the `storm-deep-research` skill again and tell it the plan
already exists at that path — it should skip step 2 and go straight to step 3
(perspective questions) / step 4 (recursive search). Fan the 6 branches out as
parallel subagents; the skill is explicit that running 60+ searches in the main
context will overflow it. Each subagent returns **only** ledger rows plus its
query count, never raw page content.

Artifacts still owed by the skill, none of which exist yet:
`research-metrics.md`, `source-ledger.md`, `evidence-map.md`,
`verification-audit.md`, `report.html`, `quality-review.md`.

### The one blind spot worth carrying forward in your head

The most load-bearing hypothesis in the plan, and the reason the research is
worth doing at all: **SIM / SECS is computed by a speaker-verification encoder
trained on _speech_.** It may be close to blind to rasp, grit, and weight —
precisely the qualities the gpt-audio judge said were missing every single time
(`HANDOFF.md` §4). If that is true, the paper's SIM 0.922 does not mean what it
looks like it means, and optimizing toward it would be optimizing toward the
wrong target. Branch B2 exists to settle this. Do not let it get dropped.

### Budget warning

The *previous* session burned its 200-call WebSearch cap on an earlier
`/deep-research` run, which is why the research never happened then either.
WebFetch kept working (that is how arXiv 2607.27768 got read). Watch the
WebSearch budget — spend it on B1/B2/B3, and prefer WebFetch for any source
whose URL you can already guess or construct (arXiv listings, HF dataset pages).

---

## 3. Files added this session

| Path | Status |
|---|---|
| `storm-reports/prompt-conditioned-svs-english/research-plan.md` | New, complete, untracked |
| `HANDOFF_2026-08-09_session2.md` | This file |

Nothing else was created, edited, or deleted. `scripts/infer_vocalrender_svs_single.py`
still shows as modified in git — that is the `--lora_dir` patch from the *prior*
session (`HANDOFF.md` §3), not from this one. It is uncommitted and you probably
want to commit it, since `eval_wer.py` imports that inference path.

Also still untracked from the prior session, for orientation: `HANDOFF.md`,
`conf/svs_preprocess_amy_only.yaml`, `conf/svs_preprocess_en.yaml`,
`conf/svs_train_en.yaml`, `examples/smoke_and_honey.json`,
`examples/smoke_and_honey_v2.json`, `run_smoke_and_honey.sh`, `scripts_amy/`.

---

## 4. Why this session ended — read before you assume it is fixed

The session reported: *"Transcript saving is off — inherited
`CLAUDE_CODE_CHILD_SESSION` marker."* Diagnosed:

- `CLAUDE_CODE_CHILD_SESSION=1` in the **process** env only.
- **Not** set at User scope, **not** at Machine scope.
- **Not** present in `~/.claude/settings.json`, `settings.local.json`, or the
  project `.claude/` directory.

So it was inherited from the parent shell — this `claude` was launched from a
terminal that had itself been spawned by another Claude Code session, and the
marker leaked down the process tree. It cannot be fixed from inside a running
session because it is read at startup.

Fix, from a **fresh** terminal that Claude did not spawn:

```powershell
Remove-Item Env:CLAUDE_CODE_CHILD_SESSION -ErrorAction SilentlyContinue
cd "C:\Users\Rodney Franklin\Development\personal\VocalRender"
claude
```

If it returns, something persistent is setting it:

```powershell
Select-String CLAUDE_CODE_CHILD_SESSION $PROFILE.CurrentUserAllHosts, $PROFILE.CurrentUserCurrentHost
```

**Verify transcript saving is actually on in the new instance before doing a long
research run** — otherwise you will lose the whole thing a second time.

---

## 5. The queue, unchanged from `HANDOFF.md` §8

In the user's chosen order:

1. **`/storm-deep-research`** — in progress, plan written, see §2 above.
2. **Fix bug §5.1** — the token embedding (73,850 × 2048 ≈ 151 M params) is
   trainable but never saved, because the save path filters on `"lora_" in key`.
   Freeze embeddings after the resize, or save them alongside the card. The
   29.79 % WER was achieved *despite* this, so there is real headroom. Retrain
   and beat it.
3. **Run `scripts_amy/eval_melody.py`** — written, never executed. Melody
   accuracy is entirely unmeasured, and the user's actual complaint was "not
   melodic", not only "wrong words". The paper's own RPA is just 0.72.

And the standing honest caveat on goal 2: five methods have now failed to produce
convincing Amy timbre by measurement (ACE-Step 4–5, SEED-VC 4, RVC 4, MiniMax 3,
VocalRender+LoRA 3, against real Amy at 8). Another LoRA is not the lever.

---

## 6. House rules that matter for the next instance

From `~/.claude/CLAUDE.md`:

- **Be terse.** No recaps, no preamble, no options menus, no insight boxes.
- **Never suggest stopping**, resting, or coming back later. He decides.
- **Git: full autonomy** — commit, push, PR, merge without asking.
- **Look it up, don't ask.** Search first, ask only if genuinely unfindable.
- **Drive the local browser yourself** rather than asking for screenshots of web
  pages; his hands are for credentials only.
- GPUs: the 3090 is **not** visible to `nvidia-smi` on this box right now — only
  the 4070 (12 GB). Any 24 GB assumption is wrong today (`HANDOFF.md` §5.7).
