# RESONA-VEP spike — S2 result (Amendment 2)

Date: 2026-08-19
Plan: [docs/plans/active/paper2-resona-vep-spike.md](../../../docs/plans/active/paper2-resona-vep-spike.md), "Amendment 2"
Script: [paper2_benchmark/scripts/run_resona_vep_spike_s2.py](../../scripts/run_resona_vep_spike_s2.py)
Report: [s2_report.json](s2_report.json)

`neural_models_trained: 0`. Every feature bank is frozen; one identical,
untuned `StandardScaler -> LogisticRegression(max_iter=5000)` probe is fit
on top of each. Wall time: **18.9 seconds** for all 3 subjects, 5 arms —
this stage trains no network, so the entire question of "did optimization
work" is removed by construction.

## Full results

| subject | A (RF bank) test | B (LIF bank) test | C (FFT power) test | D (raw linear) test | E (FBCCA, context) |
|---|---|---|---|---|---|
| 1 | 0.167 | 0.133 | 0.367 | 0.933 | 0.750 |
| 2 | 0.050 | 0.067 | 0.150 | 0.283 | 0.617 |
| 3 | 0.267 | 0.083 | 0.683 | 0.883 | 0.983 |

(Train accuracy was ~1.0 for A/B/C/D on nearly every subject — 384 or 8504
features against 120 training rows, expected and not informative on its
own; test accuracy is the number that matters here, same caveat as S1's
linear-baseline diagnostic.)

## Gate G-wire — PASS

```
tolerance: +/- 0.10 absolute, arm D vs. S1's recorded linear-baseline test accuracy
```

| subject | S1 expected | S2 observed (arm D) | diff |
|---|---|---|---|
| 1 | 0.933 | 0.933 | 0.0003 |
| 2 | 0.350 | 0.283 | 0.067 |
| 3 | 0.883 | 0.883 | 0.0003 |

All within tolerance. The S2 pipeline (same preprocessing, same split,
re-implemented probe) reproduces S1's diagnostic closely — subjects 1 and 3
match almost exactly; subject 2 is 0.067 off, still inside the ±0.10 band.
**The wiring is confirmed correct before any other arm is interpreted.**

## Gate G-feat — FAIL

```
threshold: max(A_test, B_test) > 0.20, on at least 2 of 3 subjects
```

| subject | max(A, B) test | clears 0.20? |
|---|---|---|
| 1 | 0.167 | No |
| 2 | 0.067 | No |
| 3 | 0.267 | Yes |

Only 1 of 3 subjects clears the bar; 2 are required. **G-feat fails.**

## Gate G-rq1 — not evaluated (G-feat failed)

Per the locked gate order, G-rq1 (A vs. B) is not computed as a
decision-relevant number when G-feat fails.

## What this means — and why it is a real result, not a third harness failure

This is the first result in the RESONA-VEP line that is not confounded by
optimization: no network was trained, so there is no training-budget,
learning-rate, or convergence explanation available for a weak outcome.
**The frozen RF and LIF spike-rate features themselves do not carry enough
class information for an untuned linear probe to use, on 2 of 3 subjects.**

The comparison to arm C sharpens this further. C (FFT power, same 384
feature count, same probe, no learning either) beats both A and B by a wide
margin on every subject (0.367 vs. 0.167/0.133 on subject 1; 0.150 vs.
0.067/0.050 on subject 2; 0.683 vs. 0.267/0.083 on subject 3). C is
reported as a ceiling for *any* rate/power-coded method, RF or LIF, and A
and B sit well below that ceiling — so the gap is not fully explained by
"discarding phase costs accuracy" (which is what C's own gap to D and E
measures). Spike-rate coding specifically, at matched feature count, loses
more than a continuous power spectrum does. Arm D (raw time series, phase
intact) and arm E (FBCCA, phase intact) both track each other and both sit
well above C, consistent with the same pattern noted in S1 and the S1
linear-baseline diagnostic: phase is where most of the recoverable signal
lives in this data, and both A and B discard it by construction (rate
coding averages spikes over the full 4.15 s window).

## Decision rule outcome

Per Amendment 2's locked decision rule:

```
G-feat FAIL -> Frozen RF features carry no usable class information on
               real SSVEP. Combined with C, this is the first clean
               negative result the line has produced. Report and bring
               back; do not start a fourth variant.
```

This is where S2 stops. Not authorized inside this plan: a fourth variant,
any probe tuning or swap, any change to the frozen bank contracts. The
decision now belongs to a human: whether this clean negative (frozen
features, matched feature count, RF underperforms even a power-spectrum
ceiling on 2/3 subjects) is sufficient to close the RESONA-VEP line, or
whether a genuinely different formulation (e.g. one that preserves phase,
which every arm that did well here — C partially, D and E fully — has in
common) is worth a fresh, separately authorized plan.

```
real_eeg_touched_in_S2: true
subjects_used_S2: 3
moabb_dataset: Nakanishi2015
tsinghua_benchmark_or_beta_accessed: false
neural_models_trained: 0
throwaway_spike: true
wall_seconds_total: 18.9
```
