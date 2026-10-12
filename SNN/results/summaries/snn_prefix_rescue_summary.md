# SNN Prefix Rescue Summary

Promotion gates:

- `K64`: five-fold mean accuracy >= full-window minus 2 pp = 0.8056.
- `K80`: five-fold mean accuracy >= 0.8000.

## Fold-1 Winners

| Prefix | Best config | Acc | Macro-F1 | Delta vs baseline | Fold-1 gate | Gate margin |
|---:|---|---:|---:|---:|---|---:|
| 64 | `shallow_temporal_lif_distill_k64_a0p25_d0p15` | 0.7911 | 0.7836 | +0.0055 | fail vs 0.8016 | -0.0104 |
| 80 | `shallow_temporal_lif_distill_k80_a0p50_d0p15` | 0.7953 | 0.7844 | +0.0124 | fail vs 0.8000 | -0.0047 |

## Five-Fold Gate Status

| Prefix | Config | Folds | Acc mean | Acc std | Macro-F1 mean | Gate | Margin |
|---:|---|---:|---:|---:|---:|---|---:|
| 64 | `shallow_temporal_lif_distill_k64_a0p25_d0p15` | 5 | 0.7697 | 0.0190 | 0.7620 | fail vs 0.8056 | -0.0359 |
| 80 | `shallow_temporal_lif_distill_k80_a0p50_d0p15` | 5 | 0.7901 | 0.0147 | 0.7803 | fail vs 0.8000 | -0.0099 |

## All Rescue Rows

| Prefix | Config | Folds | Alpha | Dropout | Acc mean | Macro-F1 mean | Spike rate | Folds present |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 64 | `shallow_temporal_lif_distill_k64_a0p25_d0p30` | 1 | 0.25 | 0.30 | 0.7911 | 0.7771 | 0.3355 | 1 |
| 64 | `shallow_temporal_lif_distill_k64_a0p00_d0p30` | 1 | 0.00 | 0.30 | 0.7898 | 0.7783 | 0.2947 | 1 |
| 64 | `shallow_temporal_lif_distill_k64` | 1 | 0.50 | 0.30 | 0.7856 | 0.7707 | 0.3221 | 1 |
| 64 | `shallow_temporal_lif_distill_k64_a0p25_d0p15` | 5 | 0.25 | 0.15 | 0.7697 | 0.7620 | 0.2819 | 1 2 3 4 5 |
| 64 | `shallow_temporal_lif_distill_k64_a0p50_d0p15` | 1 | 0.50 | 0.15 | 0.7690 | 0.7549 | 0.2697 | 1 |
| 80 | `shallow_temporal_lif_distill_k80_a0p50_d0p15` | 5 | 0.50 | 0.15 | 0.7901 | 0.7803 | 0.3020 | 1 2 3 4 5 |
| 80 | `shallow_temporal_lif_distill_k80_a0p25_d0p30` | 1 | 0.25 | 0.30 | 0.7898 | 0.7811 | 0.2758 | 1 |
| 80 | `shallow_temporal_lif_distill_k80_a0p00_d0p30` | 1 | 0.00 | 0.30 | 0.7856 | 0.7783 | 0.2788 | 1 |
| 80 | `shallow_temporal_lif_distill_k80_a0p25_d0p15` | 1 | 0.25 | 0.15 | 0.7856 | 0.7771 | 0.2890 | 1 |
| 80 | `shallow_temporal_lif_distill_k80` | 1 | 0.50 | 0.30 | 0.7828 | 0.7724 | 0.2932 | 1 |

Recommended five-fold configs from fold-1 selection:
- `K64`: `shallow_temporal_lif_distill_k64_a0p25_d0p15`
- `K80`: `shallow_temporal_lif_distill_k80_a0p50_d0p15`
