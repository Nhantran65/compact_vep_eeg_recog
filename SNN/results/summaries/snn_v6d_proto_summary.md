# SNN V6d Differentiable Prototype Head Summary

Updated: 2026-06-07

Baseline supervised SNN:

```text
V6b shallow_temporal_spike_aware_gated_lif
accuracy = 84.41 +/- 1.29%
macro-F1 = 83.73 +/- 1.13%
balanced acc = 83.67 +/- 1.13%
```

V6d model:

```text
frozen Shallow temporal map [B,40,104]
-> Linear 40->64
-> LIF over 104 steps
-> gated_mean_membrane [64] + spike_count [64]
-> fusion embedding z [64]
-> differentiable RBF/prototype head
```

Fold-1 screen:

| Config | Accuracy | Macro-F1 | Decision |
|---|---:|---:|---|
| `shallow_temporal_spike_aware_proto_lif_ce` | `84.09%` | `83.30%` | pass gate |
| `shallow_temporal_spike_aware_proto_lif_supcon005` | `84.23%` | `83.30%` | pass gate |
| `shallow_temporal_spike_aware_proto_lif_supcon010` | `84.23%` | `83.38%` | pass gate |
| `shallow_temporal_spike_aware_proto_lif_supcon005_sep` | `84.23%` | `83.39%` | pass gate |

Five-fold results:

| Config | Accuracy | Macro-F1 | Balanced acc | Decision |
|---|---:|---:|---:|---|
| `shallow_temporal_spike_aware_proto_lif_ce` | `84.75 +/- 0.66%` | `83.94 +/- 0.51%` | `83.85 +/- 0.41%` | replace V6b as supervised SNN row |
| `shallow_temporal_spike_aware_proto_lif_supcon005` | `84.08 +/- 1.21%` | `83.31 +/- 1.08%` | `83.11 +/- 1.07%` | below V6b/V6d0 |
| `shallow_temporal_spike_aware_proto_lif_supcon010` | `84.22 +/- 0.76%` | `83.55 +/- 0.64%` | `83.42 +/- 0.61%` | below V6b/V6d0 |
| `shallow_temporal_spike_aware_proto_lif_supcon005_sep` | `84.22 +/- 0.84%` | `83.49 +/- 0.72%` | `83.34 +/- 0.66%` | below V6b/V6d0 |

Decision:

```text
V6d0 CE-only improves over V6b by +0.34 pp accuracy and +0.21 pp macro-F1.
Promote V6d0 as the strongest supervised SNN architecture row.
It does not clear the strong-claim gate of >=85.0% acc or >=84.3% macro-F1.
SupCon/prototype-separation variants do not improve over CE-only.
V5a spike_count + RBF-SVC remains the strongest SNN representation row.
```
