# SNN V6/V5a Extension Screen

Updated: 2026-06-07

Promotion gates:

```text
V6 supervised: fold-1 >=84.0% acc or >=83.3% macro-F1 -> run 5-fold
V5a probe/fusion: fold-1 >86.5% acc or >85.8% macro-F1 -> run 5-fold
```

Fold-1 screen:

| Branch | Row | Scope | Accuracy | Macro-F1 | Decision |
|---|---|---|---:|---:|---|
| V6b | `shallow_temporal_spike_aware_gated_lif` | supervised fold 1 | `83.68%` | `82.99%` | close, below gate; user-requested 5-fold run |
| V6c | `shallow_temporal_windowed_spike_lif_4bins` | supervised fold 1 | `82.02%` | `81.13%` | below gate |
| V6e | `shallow_temporal_multiscale_gated_lif` | supervised fold 1 | `82.99%` | `82.18%` | below gate |
| V5a-ext | `windowed_spike_count_4 + RBF-SVC` | probe fold 1 | `83.54%` | `82.72%` | below gate |
| V5a-ext | `spike_mean_late_fusion` | probe fold 1 | `85.34%` | `84.60%` | below gate |

User-requested V6b 5-fold run:

| Branch | Scope | Accuracy | Macro-F1 | Balanced acc | Decision |
|---|---|---:|---:|---:|---|
| V6b `shallow_temporal_spike_aware_gated_lif` | supervised 5-fold | `84.41 +/- 1.29%` | `83.73 +/- 1.13%` | `83.67 +/- 1.13%` | promoted at this step; later superseded by V6d |

Decision:

```text
V6b missed the original fold-1 gate by about 0.31 pp macro-F1, but its
user-requested 5-fold result beats V6a.
V6b was promoted as the supervised SNN row at this step, then superseded by V6d.
Keep V5a spike_count + RBF-SVC as the strongest SNN representation row.
Do not promote V6c/V6e or V5a windowed/fusion probes.
```
