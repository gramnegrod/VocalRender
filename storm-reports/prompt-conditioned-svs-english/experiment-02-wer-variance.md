# Experiment 02 — WER variance, and why the headline result does not survive

Run 2026-08-10. Script: `scripts_amy/eval_wer.py`, unchanged. Logs:
`logs/wer_{base,v1,v2}_*.log`. Reports: `outputs/wer_*/wer_report.json`.

## Why this was run

The v2 adapter scored 34.04 % WER against v1's recorded 29.79 %, which read as a
regression. Before accepting that, the v1 adapter was re-evaluated **in the same
session with the same code** — the control discipline `HANDOFF.md` §9 already
prescribes. It scored **56.74 %**, not 29.79 %. A number that moves 27 points on
identical inputs is not a measurement, so the eval itself became the subject.

## Method

Identical items throughout: `data/amy/annotations_val.json`, 15 held-out Amy
segments, 141 reference words, `--limit 20`, same prompt audio root, same
`cfg_value` defaults. Three independent runs per condition, all in one session
on one machine. The only source of variation is sampling in generation.

## Results

| condition | run 1 | run 2 | run 3 | mean | sd |
|---|---:|---:|---:|---:|---:|
| base, no LoRA | 43.26 | 49.65 | 44.68 | **45.86** | 3.36 |
| v1 — r=16, step 1000, embedding discarded | 56.74 | 36.17 | 30.50 | **41.14** | 13.81 |
| v2 — r=32, step 2000, embedding saved | 34.04 | 42.55 | 51.06 | **42.55** | 8.51 |

Welch's t-test, two-sided:

| comparison | difference | p |
|---|---:|---:|
| base vs v1 | +4.73 pts | 0.617 |
| base vs v2 | +3.31 pts | 0.581 |
| v1 vs v2 | −1.41 pts | 0.889 |

**No comparison approaches significance.** All three distributions overlap.

## What this overturns

`HANDOFF.md` §4 records the project's headline finding:

> | Released checkpoint | 44.68 % |
> | + English LoRA @ step 1000 | **29.79 %** |
>
> *"The LoRA moved WER by 15 points…"*

Both figures were **single draws**. This experiment reproduces 44.68 % as one of
three base runs — it is an ordinary sample from a distribution centred on
45.86 %. And 29.79 % sits below the lowest of three v1 runs here (30.50 %), so it
is the favourable tail of a distribution centred on 41.14 %.

The 15-point improvement is an artefact of comparing one draw against another.
The defensible estimate is **~3–5 points, and not statistically resolvable at
n=3**.

This does not show the LoRA is worthless. Both LoRA means sit below base, and
the direction is consistent across conditions. It shows the *effect size was
overstated roughly threefold*, and that no decision should have rested on it.

## What this says about v2 specifically

v1 and v2 are indistinguishable (p=0.89). So on this metric, the embedding fix
and the rank increase bought nothing measurable — though note v2 has the tighter
spread of the two LoRA conditions (sd 8.51 vs 13.81), and base is tighter still
(3.36). That ordering is suggestive rather than established.

The qualitative difference is more robust than the aggregate. v1 emits
`i go back alcua`, `they go bagel to her`, and one empty hypothesis; v2 returns
three exact renditions of *"you go back to her and i go back to her"* and a
clean *"we only said goodbye a hundred times"*. But the base model also produces
recognisable English here (`you went back to what you know so far far far`), so
the dramatic Mandarin-collapse example in `HANDOFF.md` §4 is not representative
of base behaviour on this val set either.

## Why the metric is this noisy

- **141 reference words.** One badly-collapsed segment moves the aggregate by
  ~10 points. Segment 7 is a 3-word reference (`u log blow`); a single failure
  there scores 100–133 %.
- **References are whisper transcriptions of the originals**, so they carry
  their own errors — already flagged in `HANDOFF.md` §4.
- **Generation is stochastic** and nothing pins the seed.
- **Duplicated items**: five of fifteen segments are the same lyric line
  (*"you go back to her and i go back to her"*), so effective diversity is lower
  than n=15 suggests.

To detect a 5-point difference at the observed sd of ~9 would need roughly **50
runs per condition** — infeasible. The fix is not more repeats but a better
instrument.

## Consequences

1. **Do not quote 29.79 % or the 15-point improvement.** Both are retired.
2. **Report WER as mean ± sd over ≥3 seeds**, never a single run. `eval_wer.py`
   should grow a `--runs N` flag that does this by default.
3. **Enlarge the eval set.** 15 segments and 141 words cannot carry the weight
   placed on them. GTSinger's held-out English val (180 segments) is already on
   disk and unused for WER — that is a ~12× increase available today.
4. **Deduplicate the val items**, or weight by unique lyric line.
5. **Pin a generation seed** so a run is reproducible, then vary it deliberately.
6. **Run `eval_melody.py`.** It has still never been run, and melody accuracy is
   the axis the user actually complained about ("not melodic"). Given how weak
   WER is here, it may be the more informative metric.

## Caveat on this experiment

n=3 per condition is itself small — it establishes that the variance is large
and that the prior claims are unsupported, but it does not establish that the
LoRA has no effect. Absence of significance at n=3 with sd≈9 is weak evidence of
absence. The correct reading is: **the effect, if present, is smaller than this
instrument can see.**
