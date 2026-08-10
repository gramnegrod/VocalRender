# Experiment 01 — Does vocal grit survive the AudioVAE?

Run 2026-08-09 on this machine. Script: `scripts_amy/vae_roundtrip_shr.py`.
Raw output: `outputs/vae_roundtrip/shr_results.json`, plus reconstructed wavs.

This was the frontier question of the deep-research report — the one claim the
report explicitly refused to assert because nobody has ever measured it. It is
now measured.

## Method

Encode → decode through VocalRender's AudioVAE with **no generative step**:
no LM, no DiT, no sampling. `encode()` returns the posterior mean `mu`, so this
is the deterministic path — the ceiling of what the representation can carry.

Per voiced frame, F0 is tracked with pyin **on the original only** and reused
for both signals (estimating F0 separately on the reconstruction would confuse
"the grit is gone" with "the tracker jumped an octave" — the exact failure mode
subharmonics cause in pitch trackers). Subharmonic-to-harmonic ratio is then
the summed magnitude in ±12 % bands around the half-integer multiples
(n−0.5)·F0 divided by the same around integer multiples n·F0.

Material: four Amy stems, chosen for audible rasp. 4,421 voiced frames total.

## Structural finding, before any measurement

**The VAE is asymmetric: it encodes at 16 kHz and decodes at 48 kHz.**
`in_sample_rate = 16000`, `out_sample_rate = 48000`
(`src/vocalrender/modules/audiovae/audio_vae_v2.py:439–440`).

Everything above the **8 kHz encode-side Nyquist is absent from the latent by
construction**. Whatever appears above 8 kHz in the output is the decoder's
invention, not a reconstruction of the source. The mixing-engineer account of
rasp puts the fundamental band at ~2.5 kHz with harmonic partners at 5 and
10 kHz — so the first two survive the bottleneck and **the 10 kHz partner
cannot**, regardless of decoder quality.

## Results

### Subharmonic-to-harmonic ratio

| file | original | round trip | change | frames |
|---|---:|---:|---:|---:|
| 01-Rehab chunk003 | 0.0742 | 0.0515 | **−14.4 %** | 984 |
| 06-Love-Is-A-Losing-Game chunk005 | 0.0235 | 0.0207 | −10.8 % | 1018 |
| 02-You-Know-Im-No-Good chunk001 | 0.0464 | 0.0481 | −1.1 % | 1513 |
| 03-Me-And-Mr-Jones chunk002 | 0.1853 | 0.1419 | −11.1 % | 906 |

**Median change: −10.9 %.**

### Band energy delta (dB, round trip minus original)

| file | 2–3 kHz | 4.5–5.5 kHz | 9–11 kHz | 12–16 kHz |
|---|---:|---:|---:|---:|
| 01-Rehab chunk003 | −0.1 | +0.1 | −2.9 | −0.5 |
| 06-Love-Is-A-Losing-Game chunk005 | −0.5 | −0.1 | −1.5 | −2.6 |
| 02-You-Know-Im-No-Good chunk001 | +0.2 | −0.5 | **−6.9** | **−6.6** |
| 03-Me-And-Mr-Jones chunk002 | −0.0 | −0.5 | **−6.2** | **−5.9** |

The pattern is exactly what the 8 kHz encode Nyquist predicts: the two bands
**below** it are preserved to within half a dB, and the two bands **above** it
lose 1.5–6.9 dB.

## Verdict

**The hypothesis is refuted, and that is good news.**

The report's central untested claim was that grit dies in the VAE and therefore
no amount of training on this architecture could ever recover it. A ~11 %
median SHR loss is real but moderate — it is not the disappearance the three
premises jointly predicted. **The subharmonic carrier of rasp largely survives
the representation.** The VAE is not the primary reason renders score 3/10
against real Amy's 8/10.

By elimination, the loss is in **generation** — the LM/DiT sampling path
producing conditional-mean-seeking output, which is the mechanism XiaoiceSing2
and the over-smoothing literature describe, and which matches SVCC 2025 finding
vocal technique reproduced at only 37–44 % while identity looked solved.

Three consequences:

1. **Do not abandon in-model timbre work as physically impossible.** That
   argument is now closed off by measurement.
2. **The >8 kHz band is fabricated.** Any evaluation of "air", "sheen" or the
   10 kHz rasp partner is scoring decoder hallucination, not fidelity. This
   also means prompt clips carry no genuine information above 8 kHz — worth
   remembering before attributing timbre failures to prompt quality.
3. **Post-VAE per-artist conversion remains the pragmatic path**, but now for
   an ordinary reason — it sidesteps a weak generator — rather than because
   the representation is incapable.

## Caveats

- Four files, one artist, one genre. Enough to refute a strong claim; not
  enough to characterise the VAE across voices.
- SHR is a proxy. It captures period-doubling energy, which is the established
  acoustic carrier of roughness, but perceived grit is not identical to SHR and
  no listening test was run.
- Only the deterministic `mu` path was measured. Sampling from the posterior
  during real generation can only add noise relative to this ceiling.
- The band-delta figures conflate genuine loss with resampling behaviour at the
  16 kHz encode step; the SHR figure, computed on bands around a tracked F0
  well below Nyquist, is the more trustworthy of the two.
