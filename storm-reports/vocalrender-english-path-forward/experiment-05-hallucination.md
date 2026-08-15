# Experiment 05 — Where the hallucination lives, and the setting that fixes half of it

Run 2026-08-14. Triggered by a listener report: *"the first part started out good
then went weird."* The ASR agreed — *"And my tears dry"* correct, *"get some
from, wow, my God"* not.

## E1 — Is it end-of-sequence drift?

Hypothesis from the research pass: VoxCPM has open bugs (#352, #357, #213)
describing hallucination at the END of audio specifically for non-Chinese
languages, and a maintainer (#195) ties it to the stop head converging far
faster than the diffusion loss — which is exactly the signature our training
showed. If true, errors should cluster late and long segments should be worse.

Method: 45 renders (15 held-out items × 3 passes, VocalRender v2 @ cfg 3.0),
transcribed with word timestamps, then a Levenshtein **alignment backtrace** to
attribute each error to a reference-word position.

**Result — position matters, length does not.**

| | error rate |
|---|---:|
| first half of line | 13.2 % (25/189) |
| second half | 22.6 % (53/234) |
| difference | **+9.4 pts, Fisher p = 0.0163** |

By quartile: 12.7 % → 16.7 % → **27.8 %** → 17.3 %. The peak is the third
quarter, not the very end — degradation is mid-to-late, and partially recovers.

Duration versus WER: **r = +0.026, p = 0.86.** No relationship whatsoever.

**Consequence: the chunking fix is dead.** Degradation tracks *relative*
position within a line, not elapsed time or token count. Splitting lines into
shorter chunks would simply produce more bad second-halves. This was the
cheapest proposed mitigation and it is ruled out before any of it was built.

### A method error worth recording

The first version of this analysis split the reference by word count and the
hypothesis by timestamp, then scored the halves independently. That is invalid —
the two split points do not correspond — and it produced 33–50 % "half" error
rates on items whose overall WER was 0 %. Replaced with the alignment backtrace
above. The invalid version briefly appeared to show *no* positional effect.

## E2 — What actually helps: CFG scale

Chasing a 19-point discrepancy between two of our own measurements turned up a
confound: they differed in prompt strategy **and** in `cfg_value` (3.0 vs
`eval_wer.py`'s default 2.0). Classifier-free guidance controls how tightly the
model adheres to its conditioning, so it was the more plausible cause. Tested
directly, all through `eval_wer.py` so the code path is identical.

| cfg | prompt | n | mean WER | sd |
|---:|---|---:|---:|---:|
| 2.0 | same-song | 3 | 42.55 | 8.51 |
| **3.0** | **same-song** | **3** | **26.95** | **2.13** |
| 4.0 | same-song | 3 | 35.70 | 7.81 |
| 5.0 | same-song | 3 | 35.22 | 2.87 |
| 2.0 | fixed | 3 | 35.46 | 3.09 |
| 3.0 | fixed | 3 | 31.44 | 6.03 |

**cfg 3.0 is a minimum, not a monotonic trend** — the curve turns back up at 4
and 5. Pairwise at n=3 is underpowered (cfg2 vs cfg3 p=0.079; cfg4 vs cfg3
p=0.184; cfg5 vs cfg3 p=0.019), but pooling every non-3.0 setting:

> **cfg 3.0: 26.95 ± 2.13  ·  everything else: 37.84 ± 6.92
> ·  −10.89 pts, Welch p = 0.0020**

cfg 3.0 also has the tightest spread of any condition tested, which matters
independently: it is more reproducible, not just better on average.

**Prompt strategy barely matters.** At cfg 3.0, same-song 26.95 vs fixed 31.44;
at cfg 2.0, same-song 42.55 vs fixed 35.46 — inconsistent in direction. The
earlier "fixed prompt halves WER" hypothesis is withdrawn; it was CFG all along.

## Where this leaves the comparison

| system | WER | voice identity |
|---|---:|---:|
| VocalRender v2 @ cfg 2.0 | 42.55 ± 8.51 | 9.0 |
| **VocalRender v2 @ cfg 3.0** | **26.95 ± 2.13** | **9.0** |
| SoulX-Singer | 22.46 ± 2.86 | 2.0–4.0 |

**VocalRender at cfg 3.0 is no longer distinguishable from SoulX on words**
(4.49 pts apart, p = 0.10) while remaining far ahead on voice. The voice judge
scored renders made at cfg 3.0, so the two measurements are consistent with each
other.

If that holds up, the generate-then-convert hybrid recommended in the main
report is unnecessary — the model already on disk does both jobs, and the fix
was one number in a config file.

## Unresolved

- **An 8-point gap between two code paths at nominally identical settings.**
  `analyze_hallucination.py` (subprocess to `infer_..._single.py`) gave 23.17 %
  where `eval_wer.py` gave 31.44 % at the same cfg and prompt. Something else
  differs — `prompt_max_frames` and prompt encoding are the suspects. Until
  that is understood, treat absolute WER values as code-path-specific and only
  compare within a path. **The 23.17 % figure should not be quoted.**
- Whether cfg 3.0 remains optimal on non-Amy voices or longer passages.
- The positional effect is established but its *cause* is not. It is consistent
  with the stop-head imbalance, which remains untested — raising `lambda_stop`
  is still the untried mitigation.
- n=3 per cell throughout. The pooled test is solid; individual pairwise
  comparisons are not.

---

## E3 addendum — the 9.0 was the opening only (CORRECTION)

2026-08-15. The listener reported that renders "sounded good but only at the
very first 2-3 seconds." Tested by splitting renders and judging head vs tail
with the same voice judge.

| render | first 3 s | last 3 s |
|---|---:|---:|
| chunk000 | **9.0** | judge failed ×3 |
| chunk001 | **9.0** | **1.0** |
| chunk007 | judge failed ×3 | **2.0** |

**This corrects Experiment 04.** The whole-clip 9.0 scores were driven by the
opening. The defensible claim is not "VocalRender is indistinguishable from real
Amy" but "**indistinguishable for roughly the first three seconds, then it
decays badly**". An LLM judge scoring an 8-second clip apparently weights the
first impression heavily; a human caught it immediately. Worth remembering: the
judge is useful but it is not a listener.

Two of six calls failed persistently (`n_ok: 0`) even after retries — the known
exception-swallowing behaviour. Four cells is thin, but 9/9 versus 1/2 is not a
marginal difference.

### This reverses the chunking verdict

E1 ruled out chunking because *word* errors track relative position, so shorter
chunks would just produce more bad second-halves. That reasoning still holds
**for words**. But timbre decay is a different failure with a different
signature: it appears to track absolute time since the prompt, which is exactly
what re-anchoring every few seconds fixes. Chunking is back on the table for
voice, having been correctly rejected for words.

The obvious test: render in ~3 s pieces, crossfade, and judge head and tail of
the concatenation. Risks are audible seams and broken musical phrasing across
boundaries.

### Mechanism, unconfirmed

Consistent with prompt-conditioning influence weakening as generation proceeds —
the prompt latents are prepended once, and their effect on attention plausibly
decays with distance. Not measured. An alternative worth ruling out is that the
VAE decode or the F0 track drifts independently of the prompt.
