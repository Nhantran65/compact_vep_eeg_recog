# HARP-SNN Fast Accuracy-Ceiling Gate

## Technical summary

**Terminal decision: `STOP_HARP_CONTINUOUS_NOT_SOTA_CAPABLE`.**

This is a throwaway exploratory screen on the already-spent Nakanishi2015
subjects 1--3 partition. It is not confirmatory evidence and cannot support
a SOTA, population, energy, efficiency, or Pareto claim.

## Blocking checks

- P0 provenance/leakage: **pass**
- P1 synthetic instrument: **pass**
- H1 continuous ceiling: **fail**
- H2 exact spiking twin: **not_evaluated**

## Phase awareness helped, but the absolute ceiling remained far below 0.80

The primary metric is unseen-class accuracy over ten classes, averaged
equally across the nine subject/source-set cells.

| Metric | Observed | Gate |
| --- | ---: | ---: |
| A1 unseen accuracy | 0.393 | >= 0.800 |
| A1 - genuine FBCCA | +0.098 | >= 0.000 |
| A1 - magnitude-only | +0.140 | >= +0.050 |

| Subject | Source set | C0 unseen | C1 unseen | A0 unseen | A1 unseen |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | S-a | 0.260 | 0.240 | 0.280 | 0.320 |
| 1 | S-b | 0.320 | 0.300 | 0.220 | 0.500 |
| 1 | S-c | 0.320 | 0.320 | 0.260 | 0.340 |
| 2 | S-a | 0.200 | 0.220 | 0.380 | 0.560 |
| 2 | S-b | 0.200 | 0.240 | 0.400 | 0.520 |
| 2 | S-c | 0.180 | 0.260 | 0.140 | 0.460 |
| 3 | S-a | 0.280 | 0.320 | 0.160 | 0.220 |
| 3 | S-b | 0.320 | 0.360 | 0.180 | 0.400 |
| 3 | S-c | 0.300 | 0.400 | 0.260 | 0.220 |

## Scope, method, and audit boundary

- Data: Nakanishi2015 subjects 1--3, 12 classes, 10 source-partition
  trials and 5 test-partition trials per class; first 256 samples after
  the MOABB dataset latency offset.
- Phase mapping: the paper's target-index JFPM formula joined to the
  dataset repository's target-axis frequency order; no EEG-derived phase.
- Genuine FBCCA: five official Chebyshev-I subbands with zero-phase
  filtering and published subband weights; the older single-band helper
  is recorded only as lineage and is not used as C1.
- HARP fitting: exactly two source labels per cell, shared scorer across
  all 12 queries, 200 full-batch Adam epochs, seed 42, and source-only
  robust channel scaling.
- Exact trial IDs, per-trial hashes, predictions, score matrices, fitted
  parameters, dependency versions, and bootstrap intervals are in
  `report.json`.

## Limitations and decision consequence

The three subjects and reused test partition only decide whether further
ICASSP investment is justified. H2 is structurally unreachable when H1
fails. Cost diagnostics are likewise omitted after an H1 failure, as locked
before execution. No Tsinghua Benchmark, BETA, QNN, EMD-Fuzzy reproduction,
hardware energy estimate, or frozen primary Paper 2 result was accessed.

## Validation

- Prediction-to-metric recomputation: **pass**
- Report JSON SHA256: `4ffb61a6a6a3987d9e4a8c5b762b77901acaaa94098a616822e2f0c2470e421a`
- Total wall time: 24.6 s.
