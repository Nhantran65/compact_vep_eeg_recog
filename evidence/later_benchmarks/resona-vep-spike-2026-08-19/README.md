# RESONA-VEP spike — S0 result

Date: 2026-08-19
Plan: [docs/plans/active/paper2-resona-vep-spike.md](../../../docs/plans/active/paper2-resona-vep-spike.md)
Script: [paper2_benchmark/scripts/run_resona_vep_spike_s0.py](../../scripts/run_resona_vep_spike_s0.py)
Report: [report.json](report.json)

## Claim tested

Does a resonate-and-fire (RF) neuron, matched in population size (K=8) and
training budget to a plain LIF neuron, extract frequency identity better on
a signal where the answer is known by construction — an 8 Hz vs 12 Hz sine
plus Gaussian noise, swept over four noise levels? No real EEG is opened by
this stage.

## Gate S0 — PASS

```
cells_total: 20  (5 seeds x 4 noise levels)
cells_rf_beats_lif: 18
threshold_required: 15 (75%)
verdict: pass
```

RF beat LIF on 18 of 20 cells. Both losses were at the hardest noise level
(std=4.0, the "genuinely hard SNR" case predeclared in the plan):

| cell | seed | noise_std | RF acc | LIF acc |
|---|---|---|---|---|
| 3  | 0 | 4.0 | 0.4975 | 0.5450 |
| 11 | 2 | 4.0 | 0.5200 | 0.5250 |

At every other noise level (0.5, 1.0, 2.0), RF beat LIF on all 5 seeds.

## w-spread diagnostic — no instability

```
overall across all 20 cells x 8 neurons:
  min:  3.90 Hz
  max:  20.09 Hz
  mean: 11.99 Hz
  std:  5.25 Hz
```

Learned frequencies stayed spread across the initialized 4-20 Hz band —
no collapse to ~0 Hz and no divergence. This rules out "training
instability" as the explanation and makes the 18/20 result a genuine
signal, not noise from a broken optimizer.

## A numeric-stability fix made before any result was seen

The RF neuron contract locks a *discrete-time forward-Euler* step, which is
unconditionally unstable for an undamped oscillator (state grows every
step for any nonzero learnable frequency `w`). A frequency-independent
damping constant (`b=-1`, checked first) overflowed to ~1e73 within the
512-sample window for the higher-frequency neurons in the population —
before any training even began. This was caught by a direct numerical
check, not by watching a spike/loss go to NaN mid-run.

Fix: `b` is initialized per neuron so every neuron's Euler step has the
*same* target spectral radius (0.995), a closed-form, frequency-dependent
damping init standard for resonator networks. The `w`-initialization band
was also narrowed from an initially-planned 4-40 Hz (chosen to match S1's
preprocessing band) down to **4-20 Hz**, because at fs=256 Hz, 40 Hz sits
close enough to the Euler stability wall (dt*w=0.98) that the damping
required for stability leaves almost no real resonance (Q~1.2). 4-20 Hz
keeps the whole population meaningfully resonant (Q~4-10) while bracketing
the two class frequencies (8, 12 Hz) with margin. Both choices were frozen
before any classification result was observed and are documented in the
script's module docstring.

## Compute note

The training loop unrolls 512 timesteps in a Python-level loop (state at
t+1 depends on t, so it cannot be vectorized away), on tensors as small as
64 x 8. This was run on CPU, not GPU: a direct A/B check on one cell showed
CPU at 165s vs GPU at 298s for identical training (100 epochs) — GPU
kernel-launch overhead dominates at this tensor size, so CPU is the faster
choice here, not a fallback.

```
wall_seconds_total: 4517.8s (~75 min) for all 20 cells, both arms
```

## Decision rule outcome

Per the plan's fixed decision rule: **S0 PASS -> proceed to S1** (minimal
real-data mechanism check on Nakanishi2015 via MOABB, 3 subjects, gated
A-vs-B only). Not yet run as of this report.

```
real_eeg_touched_in_S0: false
throwaway_spike: true
```
