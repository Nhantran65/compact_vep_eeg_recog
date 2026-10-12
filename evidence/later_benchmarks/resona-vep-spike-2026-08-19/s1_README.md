# RESONA-VEP spike — S1 result

Date: 2026-08-19
Plan: [docs/plans/active/paper2-resona-vep-spike.md](../../../docs/plans/active/paper2-resona-vep-spike.md)
Script: [paper2_benchmark/scripts/run_resona_vep_spike_s1.py](../../scripts/run_resona_vep_spike_s1.py)
Report: [s1_report.json](s1_report.json)

## Gate S1 — FAIL (INCONCLUSIVE at n=3, per the locked decision rule)

```
subjects_total: 3
subjects_a_beats_b: 1
threshold_required: 2
verdict: fail
```

| subject | A (RF) train / test | B (LIF) train / test | C (FBCCA, context only) | A beats B |
|---|---|---|---|---|
| 1 | 0.108 / 0.067 | 0.092 / 0.067 | 0.750 | No |
| 2 | 0.108 / 0.117 | 0.108 / 0.117 | 0.617 | No |
| 3 | 0.117 / 0.133 | 0.083 / 0.100 | 0.983 | Yes |

## The literal gate number is not the important finding here

**Train accuracy never rose meaningfully above chance (1/12 = 0.083) for
either arm, on any subject.** Both A and B are essentially failing to fit
even their own training set. Meanwhile the training-free classical baseline
(arm C, FBCCA-style CCA) scores 0.62-0.98 on the same trials. This is not
"RF and LIF are similarly bad at this task" — it is a sign that **this
spiking-network training setup, at the fixed budget locked before either
stage ran, does not learn from real SSVEP EEG at all**, independent of
which neuron model is used. A 1-of-3 tally under those conditions carries
no more information about the resonance mechanism than a coin flip would.

This distinction matters for the same reason S0's w-spread diagnostic
mattered: a failed/tied comparison can mean two very different things
("the mechanism doesn't help" vs. "the harness never worked well enough to
see the mechanism"), and only the second one leaves the actual research
question untested. Recommending against reading this S1 result as "the
resonance mechanism failed on real data."

## Two bugs were caught and fixed before any officially gated number was seen

Both are documented in the script's module docstring; summarized here:

1. **Split bug** (caught via a class-distribution check, before any
   accuracy was observed): the plan's literal split rule — "chronological
   first-70%/last-30% trials per subject", applied pooled across all 180
   trials — collides with how Nakanishi2015 actually records data: all 15
   repetitions of one class contiguously, then the next class (a block
   design, not interleaved). A pooled cut landed on class-block boundaries
   and would have excluded most classes from train or from test entirely.
   **Flagged to the user and confirmed** before proceeding: split is now
   applied per class (first ~70% of each class's own 15 trials -> train),
   preserving the "chronological, not shuffled" intent while keeping every
   class represented in both partitions.

2. **Scale bug** (caught via a diagnostic run before the officially gated
   run): MNE returns EEG in volts (~6e-6 amplitude). The RF/LIF neuron
   contract's `threshold=1.0` was calibrated in S0 against synthetic sines
   of amplitude ~1.0. Raw-volt input never reached threshold — the first
   attempt at this run had both arms collapse to exact chance (0.083) with
   RF's learned `w` barely moving from its initialized values, while FBCCA
   still scored 0.75 on the identical trials, ruling out "the task itself
   is unlearnable" as the explanation. Fixed with per-trial, per-channel
   unit-variance normalization (z-score) after baseline removal — a scale
   calibration, applied identically to both arms, not a result-driven
   hyperparameter choice.

The result above (train accuracy stuck near chance) is *after* both fixes.
Spike rates are healthy post-fix (~0.39, not saturated/silent) and losses
do decrease marginally, but training accuracy still does not clear chance
within the fixed 100-epoch budget on any subject.

## What this does and does not tell us

- Does **not** show the resonance mechanism fails on real SSVEP data —
  neither arm learned enough to be informative about that question.
- Does **not** invalidate S0's result — S0's synthetic task used clean
  amplitude-1 sines the population could actually fit; S1's real EEG,
  averaged-spike-rate-only readout, and 120-trial/subject budget appear to
  be a genuinely harder optimization problem than this training budget
  (100 epochs, Adam lr=1e-3, batch 32) can solve, regardless of neuron
  type.
- **Does** show FBCCA-style correlation against known target frequencies
  is a much easier way to extract this signal than a from-scratch,
  rate-coded SNN readout trained on ~120 trials — expected, and explicitly
  the reason arm C is reported for context only and never gated.

## Follow-up diagnostic (not gated, run after the user agreed to it): can any model fit this data?

A plain `sklearn.LogisticRegression` (default L2, no tuning) trained on the
exact same z-scored, flattened epochs (8 channels x 1063 samples = 8504
features, same 120/60 per-subject split) as arms A and B:

| subject | linear train_acc | linear test_acc | FBCCA (context) | SNN A / B test_acc |
|---|---|---|---|---|
| 1 | 1.000 | 0.933 | 0.750 | 0.067 / 0.067 |
| 2 | 1.000 | 0.350 | 0.617 | 0.117 / 0.117 |
| 3 | 1.000 | 0.883 | 0.983 | 0.133 / 0.100 |

Train accuracy of 1.000 is expected and not informative on its own (8504
features > 120 training rows, trivially separable). The informative number
is **test accuracy: 0.35-0.93, far above chance and above FBCCA on 2 of 3
subjects.** This rules out "the data/task can't be learned from 120 trials"
as an explanation. The class-discriminative signal is clearly present and
recoverable by a generic, untuned linear model on the identical
preprocessing. The bottleneck is specifically the SNN pipeline used for
arms A/B (a 48-dim population-averaged spike-rate readout and a 100-epoch
budget), not the dataset, the split, or the research question.

## Decision rule outcome

Per the locked decision rule: **S0 PASS, S1 FAIL -> INCONCLUSIVE at n=3.
Report. Bring back for a human decision on whether to extend subjects.**

Recorded here, not silently acted on: given both diagnostics above,
extending to more subjects (the decision rule's literal next lever) would
very likely reproduce the same near-chance training accuracy and would not
resolve the actual bottleneck. The linear-baseline check narrows this
further: the fix that would matter is specifically to the SNN readout/
training budget (e.g., a richer spike-based feature than one averaged rate
per neuron over the full 4.15 s window, and/or many more than 100 epochs),
not more data. Changing the training budget/architecture is explicitly
**not authorized by this spike** ("no hyperparameter search ... after
seeing a number") and would need to be a separate, human-authorized
follow-up — the same discipline S0 used for its own Balanced-RF
contingency.

```
real_eeg_touched_in_S0: false
subjects_used_S1: 3
moabb_dataset: Nakanishi2015
tsinghua_benchmark_or_beta_accessed: false
throwaway_spike: true
wall_seconds_total: 893.8
```
