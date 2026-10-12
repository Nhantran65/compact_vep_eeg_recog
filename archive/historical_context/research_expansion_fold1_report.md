# QNN/SNN Research Expansion Fold-1 Report

Date: 2026-06-03

Scope: fold 1 only. The documented rule remains: promote to 5-fold only if the
fold-1 gate is cleared against matched baselines or controls.

## Summary

| Track | Best tested variant | Accuracy | Macro-F1 | Balanced acc | Main comparison | Gate |
|---|---:|---:|---:|---:|---|---|
| SNN prefix distillation | K=64 | 0.7856 | 0.7707 | 0.7680 | teacher/full-window fold-1 0.8257 acc; V3 post-hoc K=64 5-fold 0.7708 acc | do not promote |
| SNN adaptive spike encoder | adaptive_signed_delta | 0.4620 | 0.4654 | 0.4732 | fixed signed_delta fold-1 0.4675 acc; delta embedding + RBF 0.6846 acc | do not promote |
| QNN temporal topology | t2_temporal_topology_qdelta | 0.8257 | 0.8175 | 0.8158 | matched t2_temporal_qdelta smooth/cosine fold-1 0.8354 acc | do not promote |
| Quantum kernel probe | snn_temporal_lif_concat QK | 0.7732 | 0.7667 | 0.7666 | same subset LinearSVC/RBF-SVC 0.8326 acc | do not promote |

## Details

SNN prefix distillation:

| Config | Prefix | Accuracy | Macro-F1 | Balanced acc | Best epoch |
|---|---:|---:|---:|---:|---:|
| shallow_temporal_lif_distill_k48 | 48 | 0.7663 | 0.7540 | 0.7538 | 23 |
| shallow_temporal_lif_distill_k64 | 64 | 0.7856 | 0.7707 | 0.7680 | 34 |
| shallow_temporal_lif_distill_k80 | 80 | 0.7828 | 0.7724 | 0.7694 | 12 |

Adaptive encoder:

| Config | Accuracy | Macro-F1 | Balanced acc | Spike rate | Best epoch |
|---|---:|---:|---:|---:|---:|
| spiking_shallow_adaptive_signed_delta | 0.4620 | 0.4654 | 0.4732 | 0.2821 | 8 |

QNN temporal topology:

| Config | Accuracy | Macro-F1 | Balanced acc | Best epoch | Training time sec |
|---|---:|---:|---:|---:|---:|
| t2_temporal_topology_qdelta | 0.8257 | 0.8175 | 0.8158 | 1 | 3084.0 |

Matched baseline:

| Config | Accuracy | Macro-F1 | Balanced acc | Best epoch |
|---|---:|---:|---:|---:|
| t2_temporal_qdelta | 0.8354 | 0.8275 | 0.8279 | 29 |

Quantum kernel probe:

| Source | Model | Accuracy | Macro-F1 | Balanced acc | Train subset |
|---|---|---:|---:|---:|---:|
| shallow80 | quantum_kernel_svc | 0.6667 | 0.6589 | 0.6626 | 600 |
| shallow80 | linearsvc_c1 | 0.8077 | 0.8013 | 0.8033 | 600 |
| shallow80 | rbf_svc_c10_gamma_scale | 0.8077 | 0.8003 | 0.8014 | 600 |
| snn_temporal_lif_concat | quantum_kernel_svc | 0.7732 | 0.7667 | 0.7666 | 600 |
| snn_temporal_lif_concat | linearsvc_c1 | 0.8326 | 0.8267 | 0.8284 | 600 |
| snn_temporal_lif_concat | rbf_svc_c10_gamma_scale | 0.8326 | 0.8246 | 0.8249 | 600 |

## Artifacts

- SNN prefix summary: `SNN/results/summaries/snn_prefix_distill_summary.md`
- SNN prefix JSONs: `SNN/results/v4_prefix_distill/*/fold_1.json`
- Adaptive encoder JSON: `SNN/results/by_config/spiking_shallow_adaptive_signed_delta/fold_1.json`
- QNN topology summary: `QNN/results/research_expansion/topology_smooth_cosine/runs/latest/ablation_summary.csv`
- QNN topology JSON: `QNN/results/research_expansion/topology_smooth_cosine/by_config/t2_temporal_topology_qdelta/t2_temporal_topology_qdelta_fold1_seed42.json`
- Quantum kernel summary: `QNN/results/quantum_kernel_probe/summary.md`
- Quantum kernel JSONs: `QNN/results/quantum_kernel_probe/*/angle_ryrz_ring_l1_pca8/fold_1.json`

## Claim Discipline

None of the four fold-1 experiments clears its documented promotion gate. The
results are useful negative/diagnostic evidence, but they should not be moved to
5-fold paper claims without either changing the hypothesis or adding a clearly
pre-registered second-pass protocol.
