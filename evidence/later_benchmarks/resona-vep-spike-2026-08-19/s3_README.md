# RESONA-VEP spike — S3 result (Amendment 3)

Date: 2026-08-19
Plan: [docs/plans/active/paper2-resona-vep-spike.md](../../../docs/plans/active/paper2-resona-vep-spike.md), "Amendment 3"
Script: [paper2_benchmark/scripts/run_resona_vep_spike_s3.py](../../scripts/run_resona_vep_spike_s3.py)
Report: [s3_report.json](s3_report.json)

## Gate P3 (instrument unit test, blocking, synthetic only) — FAIL

```
threshold: contrast > 0.80 AND |measured peak - f_in| < 0.25 Hz, at all 12 class frequencies
```

11 of 12 probe frequencies fail on peak-frequency accuracy (errors 0.52–2.62
Hz); only 9.25/9.75/10.25/12.75/13.25 etc. cluster onto the *same* argmax
neuron rather than tracking `f_in` individually. Contrast is a perfect 1.0
on every row (`rate_min = 0.0`), so **the failure is entirely in peak
frequency, not in contrast** — a different failure shape than Amendment 3's
own pre-calibration diagnostic (contrast 0.13, all 48 neurons firing).

**Per the locked decision rule, this stops S3 here: no real EEG file was
opened** (`real_eeg_touched_in_S3: false` in the report). P2, G-wire,
G-feat, and G-rq1 were not evaluated.

## This does not look like a broken calibration — verified directly, not assumed

Before trusting a gate this consequential ("no further variant without a
fresh human decision"), the calibration/classification code itself was
checked against Amendment 3's own recorded numbers, since a bug there would
invalidate the P3 verdict too:

```
Reproduced synthetic 12-class SSVEP, 8 channels, 120 train / 60 test, SNR -10 dB,
using this script's own _calibrate_rf / _rf_apply / _calibrate_lif / _lif_apply:

           this run    Amendment 3's recorded row
RF (A)      0.950            0.978
LIF (B)     0.150            0.106
FFT (C)     1.000            0.989
```

All three land within ordinary single-seed sampling variance of the
amendment's 5-seed-averaged numbers. **The calibrated bank, used as a
384-dim feature vector for a trained linear probe, reproduces the
amendment's own validated classification performance almost exactly.**

This means P3's specific test — does any *single* neuron's argmax track
`f_in` to within 0.25 Hz — is failing even though the *aggregate* feature
representation that G-feat/G-rq1 would actually consume is demonstrably
informative. Two different calibration inputs were tried for P3's own
probe stage, to rule out "wrong choice of calibration data" as the
explanation before concluding this is structural:

```
calibration on -10 dB noisy 12-class synthetic (this run's choice): errors 0.18-2.62 Hz, 1/12 pass
calibration on noiseless 12-class synthetic (also tried):           errors 2.42-5.71 Hz, 0/12 pass, worse
```

Noiseless calibration doing *worse* rules out "not enough noise in
calibration" as the fix, and points at something structural in independent
per-neuron calibration itself: a neuron whose own resonance is poorly
matched by any of the 12 training frequencies gets a low 90th-percentile
threshold (little in training ever drove it hard), making it easy to
trigger by *any* moderately strong probe regardless of match quality, while
a neuron that *is* well-matched by a training frequency earns a high
threshold from its own strong training-time response and can lose the
argmax to an "easy" neighbor. A distributed linear probe can route around
this (it can down-weight uninformative neurons); a single-neuron argmax
cannot.

## What this means for the plan

P3 as written and G-feat/G-rq1 as written appear to test two different
properties: P3 requires each neuron to individually behave like a narrow
spectrum-analyser bin; the actual downstream pipeline only requires the
*full 384-feature vector* to be linearly separable, which does not require
any single neuron to be individually frequency-selective. The verification
above shows the second property holds (at least in a matched synthetic
setting) even though the first does not. This is recorded as a finding, not
resolved here — I am not authorized to redefine or bypass a locked,
human-authorized blocking gate based on my own read of it, however well
supported; that is exactly the kind of call the decision rule reserves for
a human. My recommendation, for that decision, is below.

## Decision rule outcome

Per Amendment 3's locked decision rule:

```
P3 FAIL -> STOP. The instrument is still broken. Report the tuning table.
           No further variant without a fresh human decision.
```

This is where S3 stops, exactly as specified. **My recommendation** for the
follow-up decision: the evidence above points at P3's pass criterion (not
the calibration) as the likely mismatch — consider authorizing a revised P3
that checks the property G-feat/G-rq1 actually depend on (e.g., "the frozen
bank's 384-feature vector, fit with the same untuned probe, clears some
synthetic-data accuracy floor" — which this run already shows it does, 0.95
at SNR -10 dB) rather than single-neuron argmax tracking. A structural fix
to the calibration itself (e.g., a single shared threshold per channel
instead of independent per-neuron percentiles) is a second option but is a
larger change and wasn't tried here, since neither is authorized without a
fresh decision.

```
real_eeg_touched_in_S3: false
subjects_used_S3: 0
moabb_dataset: Nakanishi2015
tsinghua_benchmark_or_beta_accessed: false
neural_models_trained: 0
throwaway_spike: true
```
