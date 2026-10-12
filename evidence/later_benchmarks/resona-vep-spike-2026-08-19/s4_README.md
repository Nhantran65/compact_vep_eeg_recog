# RESONA-VEP spike — S4 result (Amendment 4)

Date: 2026-08-20  
Plan: [docs/plans/active/paper2-resona-vep-spike.md](../../../docs/plans/active/paper2-resona-vep-spike.md), "Amendment 4" and "S4"  
Script: [paper2_benchmark/scripts/run_resona_vep_spike_s4.py](../../scripts/run_resona_vep_spike_s4.py)  
Report: [s4_report.json](s4_report.json)

## Decision-rule outcome — all gates PASS

S4 is the first positive, unconfounded evidence in this spike for the
resonance mechanism on real EEG. Per the locked decision rule, this
**authorizes writing a registered protocol only**. It does not authorize
running that protocol and licenses no efficiency or Pareto claim; S4 did not
measure compute cost.

The reset removal is a model change, not a calibration change: the RF arm is
now a resonator with a threshold detector and sparse binary output, without
subtractive reset feedback. This weakens any claim that it is a canonical or
biologically plausible spiking neuron. Balanced RF remains the documented
reset-preserving contingency, but was not used here.

## Blocking preflights, in execution order

### P4 provenance — PASS

Both P3 and the real run received the same imported `resona_banks` module
object at:

```text
paper2_benchmark/scripts/resona_banks.py
SHA256 45e23a44ca242dd8a700fdb3f7384b46fa07ea02bfb00e8a504293bdac51767d
```

The S4 runner contains no reimplementation of RF or LIF bank dynamics.

### P3 instrument unit test — PASS 12/12

The locked `resona_banks.p3_calibration_signals` calibration set and
percentile 95 were used. Clean unit-variance probes used the locked four-second
Nakanishi epoch duration. Every row clears contrast `> 0.80` and measured-peak
error `< 0.25 Hz`.

| Input Hz | Measured peak Hz | Contrast | Error Hz |
| ---: | ---: | ---: | ---: |
| 9.25 | 9.189 | 1.000 | 0.061 |
| 9.75 | 9.716 | 1.000 | 0.034 |
| 10.25 | 10.245 | 1.000 | 0.005 |
| 10.75 | 10.776 | 1.000 | 0.026 |
| 11.25 | 11.309 | 1.000 | 0.059 |
| 11.75 | 11.843 | 1.000 | 0.093 |
| 12.25 | 12.111 | 1.000 | **0.139** |
| 12.75 | 12.649 | 1.000 | 0.101 |
| 13.25 | 13.189 | 1.000 | 0.061 |
| 13.75 | 13.731 | 1.000 | 0.019 |
| 14.25 | 14.277 | 1.000 | 0.027 |
| 14.75 | 14.824 | 1.000 | 0.074 |

This exactly reproduces Amendment 4's expected maximum error 0.139 Hz and
contrast 1.000.

### P2 measured-SNR power rule — CONTINUE

SNR was measured only on each subject's training partition. No subject had
interpolated predicted RF-p95 accuracy below 0.20; the stop rule required at
least two.

| Subject | Measured SNR dB | Predicted RF-p95 accuracy |
| ---: | ---: | ---: |
| 1 | 2.679 | 1.000 |
| 2 | 0.516 | 1.000 |
| 3 | 4.962 | 1.000 |

## Real-data gates and reported context

The identical locked probe was applied to all 384-feature A/B/C arms. C, E,
and the A-minus-C gap are context only and were never gated.

| Subject | A: reset-free RF | B: LIF | C: FFT power | D: raw linear | E: FBCCA | A-C |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **0.400** | 0.067 | 0.367 | 0.933 | 0.750 | +0.033 |
| 2 | **0.150** | 0.083 | 0.150 | 0.283 | 0.617 | +0.000 |
| 3 | **0.717** | 0.083 | 0.683 | 0.883 | 0.983 | +0.033 |

- **G-wire PASS:** arm D is within `+/-0.10` of S1 on all three subjects.
- **G-feat PASS:** `max(A_test, B_test) > 0.20` on subjects 1 and 3; 2/3
  required.
- **G-rq1 PASS:** A beats B on all three subjects; 2/3 required.

The LIF control remains at or below chance-like accuracy while the verified RF
features clear the feature gate on two subjects. RF also closely tracks the
reported-only FFT-power arm here, but this run cannot support an efficiency
claim.

## Threshold diagnostics and scope

Percentile-95 thresholds were calibrated on the training partition only, per
subject, channel, and neuron, then applied unchanged to test. RF threshold
`min / median / max` values were:

| Subject | Min | Median | Max |
| ---: | ---: | ---: | ---: |
| 1 | 18.541 | 29.785 | 45.891 |
| 2 | 18.820 | 27.330 | 37.167 |
| 3 | 19.552 | 30.173 | 39.322 |

The JSON records full RF and LIF summaries for every subject and each of the
eight channels so the in-band alpha risk remains visible. P3 proves clean-sine
frequency resolution; G-feat, not P3, is the real-noise transfer test.

```text
real_eeg_touched_in_S4: true
subjects_used_S4: 3
asserted_feature_count_abc: 384
moabb_dataset: Nakanishi2015
tsinghua_benchmark_or_beta_accessed: false
neural_models_trained: 0
throwaway_spike: true
wall_seconds_total: 36.902
```
