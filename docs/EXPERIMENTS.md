# Compact VEP EEG Experiment Audit and Summary

Survey date: 12/10/2026. Sources: code, notebook output, JSON/CSV/checkpoints on disk, and the MSc supervision brief. This is a project audit and experimental handover; it does not train a new benchmark.

## Conclusions from code and evidence

Preprocessing, classical ML, many DL architectures, raw-input SNN, Shallow+SNN, nonlinear probes, early exit, distillation, and head ablations have been carried out. The two strongest rows are spike-count+RBF (86.02%) and prototype LIF CE (84.75%). They are historical epoch-level results; they do not demonstrate an advantage on unseen subjects.

Points in the PDF that need correction before use in a thesis/paper:

1. "Subject-aware five-fold": `SVM_Classification.ipynb` cell 10 uses row-wise StratifiedKFold, and cell 8 drops the Subject metadata. The rebuilt folds have **57 recordings shared between train and test in every fold**; epochs are 1 second long with 50% overlap. `evidence/subject_disjoint` is the actual subject-level split, belonging to later research.
2. "No additional supervised training of SNN stage": `SNN/src/snn_vep/hybrid_training.py::run_hybrid_probe` loads the checkpoint of a head trained with CE. Only the **probe step** freezes the SNN; it is not a LIF that has never seen labels.
3. "End-to-end": V2/V6 freeze the Shallow backbone. Only the downstream SNN/head is trained jointly; the full raw-EEG backbone is not updated.
4. The legacy DL baselines selected checkpoint/scheduler/early stopping on test. The SNN uses inner validation, but the backbone inherits test-selected weights; hence there is no clean comparison between ML/DL/SNN yet. The hybrid also fits the scaler on the entire outer train set before inner validation.
5. Many configurations were screened on the same fold/test and compared multiple times. Mean±SD alone cannot establish significance or subject-level reliability.
6. The new data audit records **64 expected recordings, 59 usable, 1 missing raw, 4 preprocessing failures, 31 subjects with epochs, 1 without epochs**, unlike the old README which stated 63/63. See `evidence/subject_disjoint/audit.json`. Labels are inherited from the recording folder and are not locked to stimulus markers.

## Repository-wide scope surveyed

| Group | Content | Handling in the new project |
|---|---|---|
| Root MATLAB, CSV, notebook, results | VEP preprocessing and original ML/DL | Keep code, outputs, and restored CSVs |
| SNN | Raw SNN, hybrid, probe, ablation, prefix | Keep core runtime, configs, checkpoints, results |
| QNN | Quantum models, containing the shared Shallow backbone | Take only the Shallow class, 5 checkpoints, Shallow results |
| FoundationModel_EEGPT/LaBraM/REVE/BrainOmni/NeuroLM | Paper 1: transfer and interface alignment | Out of scope; EEGPT SNN screen source kept in archive with original results |
| paper2_benchmark | Matched-readout benchmark on VEP/MSS/infant, subject-disjoint, with QNN | Keep VEP manifest and related reports as evidence; do not bring the multi-task runner |
| labs | MSS 122ch/20 classes, infant 32ch/10 classes, reference, unfinished RSOD | Outside C1/C2 scope; not a completed C3 |
| sleep_* | Sleep staging, ULW/Linear-BIM/BIM-Explain, cVAN, SNN hypotheses | Separate project; not included in Compact VEP |
| research_archive | Closed hypotheses for Paper 2/sleep | Outside runtime; source remains in the old repo |

## Existing pipeline

EDF → EEGLAB → ICA/ICLabel removing Eye/Muscle ≥90% → average reference → 0.1–50 Hz → sliding 1 s epochs, 50% overlap, reject ±150 µV → CSV 14×128 = 1,792 features. The actual code is `preprocessing/Preprocess_Step4.m`, not the ExportCSV name in the old README. The original pipeline contains author-machine paths; the packaged CSVs are sufficient input for the current runtime.

Best hybrid: frozen Shallow temporal map `[B,40,104]` → Linear 40→64 → recurrent LIF → temporal depthwise gate → gated membrane + spike count → fusion → differentiable prototype/RBF head. Best probe: trained temporal LIF checkpoint → spike_count → StandardScaler → RBF-SVC C=10. Scaler/PCA/SVC probes are fit on outer train.

## Main saved results

The table below uses sample SD (ddof=1), in %. See the individual JSON files and `experiment_summary.csv`; do not mix the notebooks' population SD with this table.

| Model / readout | Accuracy | Macro-F1 | Balanced accuracy | Source |
|---|---:|---:|---:|---|
| Random Forest | 44.57 ± 2.03 | 41.29 ± 2.02 | 41.60 ± 1.94 | `results/classical_ml/` |
| SVM raw RBF | 41.86 ± 0.97 | 37.41 ± 1.71 | 38.51 ± 1.24 | `results/classical_ml/` |
| DeepConvNet, script snapshot | 77.13 ± 2.07 | 75.83 ± 2.11 | 76.28 ± 2.53 | `results/deepconvnet/` |
| ShallowConvNet, checkpoint snapshot | 82.14 ± 1.01 | 81.33 ± 0.88 | 81.20 ± 0.81 | `results/shallowconvnet/` |
| Temporal LIF | 82.56 ± 1.57 | 81.82 ± 1.44 | 81.62 ± 1.41 | `SNN/results/v2_hybrid/shallow_temporal_lif_head/` |
| Temporal gated LIF | 83.50 ± 0.89 | 82.83 ± 0.84 | 82.78 ± 0.80 | `.../shallow_temporal_gated_lif/` |
| Spike-aware gated LIF | 84.41 ± 1.29 | 83.73 ± 1.13 | 83.67 ± 1.13 | `.../shallow_temporal_spike_aware_gated_lif/` |
| Prototype LIF, CE | 84.75 ± 0.66 | 83.94 ± 0.51 | 83.85 ± 0.41 | `.../shallow_temporal_spike_aware_proto_lif_ce/` |
| Spike-count + PCA/RBF | 85.71 ± 0.86 | 85.04 ± 0.70 | 84.88 ± 0.70 | `.../shallow_temporal_lif_head_embedding_probe_spike_count/`, head 2 |
| Spike-count + RBF | 86.02 ± 0.93 | 85.39 ± 0.82 | 85.29 ± 0.87 | same folder, head 1 |

The Shallow notebook has its own snapshot of 81.48±1.08%, distinct from the checkpoint snapshot of 82.14±1.01%. The Deep notebook has 77.05±1.91%, different from the script snapshot; notebooks usually compute population SD. Keep sources separate and do not silently substitute numbers. The EEGNet/TCNet/ITNet/ChannelNet/ATCNet/Conformer/Transformer notebooks have executed outputs but are all limited by the test-selected protocol; separate CSVs were not found for every model.

## SNN branches tried

| Stage | Runs / results | Status |
|---|---|---|
| V1 raw SNN | Direct/rate/delta/signed_delta supervised fold 1: 34.58/29.74/46.47/46.75%; best delta embedding+RBF 68.46% | Not promoted to 5-fold |
| V2 controls | Raw/delta/signed_delta × 3 probes, all 5 folds; best signed_delta+PCA/RBF 50.11±0.75% | The delta transform alone does not explain the gain |
| V2 Shallow80 heads | LIF 81.53±0.67%, PLIF 81.59±0.67%, all 5 folds | Temporal map is better |
| V3 post-hoc prefixes | K=8/16/24/32/48/64/80/104, all 5 folds; accuracy 61.60/69.16/71.35/74.09/75.77/77.08/79.13/82.56% | K80 did not reach 80%; not evidence of early raw-input decision |
| V4 prefix-trained / Shallow distill | Fold1 screen and 5-fold rescue; K64 76.97±1.90%, K80 79.01±1.47% | Did not pass gate |
| Adaptive signed-delta | Fold1 supervised 46.20%; diagnostics/probes are in JSON | Not promoted |
| V5a representation probe | Many membrane/spike/concat modes, 3 probes; best spike_count+RBF 86.02% | Best historical representation |
| V5b EEGPT distillation | Student fold1 82.57%, teacher 89.35% | Not promoted, outside core runtime |
| V6a/b gating and fusion | 83.50%, 84.41%, all 5 folds | Positive historical ablations |
| V6c/e windowed/multiscale | Fold1 82.02%, 82.99%; windowed/fusion probes did not beat best | Not promoted |
| V6d prototype | CE 84.75%; SupCon005/010/005+sep: 84.08/84.22/84.22%, all 5 folds | CE best; regularizers did not help |
| EEGPT token SNN | Fold1 60.86% / 59.61% | Negative screen, no C3 |
| Smoke tests | Folders containing smoke | Run checks only, not counted as research results |

Per-fold, per-mode, per-head train/val/test metrics are in `experiment_inventory.csv`; the summary has fold count, mean, and sample SD. The `SNN/results/summaries/*.md` files are historical notes and mention QNN/FM; use this audit to understand the corrected scope.

## Later research evidence and C3

The Paper 2 V2 primary test report (490 claims/results, 490 recomputations) records **no exact-control effect for Current VEP under Holm**; the positive MSS effects belong to a different task. Do not use this result to validate the legacy 86% row. The inference rule was amended after test access but before reading aggregates, so the report itself records an amended inference.

RESONA/HARP/STAR have screens on SSVEP Nakanishi2015, 8 channels/12 classes/3 subjects. HARP stopped at 39.3% unseen, STAR's oracle ceiling is 30.4%; RESONA S4 did not train a neural model and only permits writing a protocol. None of these ran the brief's pipeline on UCI Emotiv or MindBigData.

**The brief's C3 has no data/run evidence yet** for the two named datasets. Sleep experiments do not count as C3 for Compact VEP. Remaining steps: clean subject-disjoint ML/DL/SNN rerun, independent seed set, LOSO, paired significance under an appropriate protocol, preregistered ablations, and a full-pipeline latency/parameter benchmark. Training time/spike rate/head-only timing exist but are insufficient to claim energy efficiency.

## Packaging and validation

Data was restored from the Git commit recorded in `source_manifest.json`; the original repo was not reset. The Shallow backbone was copied verbatim from the old class into `snn_vep/backbone.py`, preserving architecture/state keys; the default checkpoint path moved to `baselines/checkpoints`. Old JSON results keep their historical paths to preserve provenance.

`legacy_input_order.json` records the 59 files in notebook stdout order; the rebuilt folds have metadata and SHA256. There are no original fold hashes to prove byte-for-byte identity with the lost CSVs. Re-running the CPU probe shows the package executes; see `validation.json` for the actual differences from the original scores. Do not attribute deviations to an unmeasured cause.

Full 5-fold CPU validation: RBF reaches **86.05±0.91%** (sample SD), vs. the historical 86.02±0.93%. Folds 2–5 match the confusion matrix; fold 1 differs by 1 sample (+0.1383 percentage points). PCA/RBF fold 1 matches the confusion matrix exactly. 693 metric sets were recomputed from confusion matrices with tolerance 1e-9; the 5 backbones and 10 main head checkpoints load strictly, with correct shapes and surrogate gradients in a training step. This is packaging validation in the recorded environment, not a new subject-disjoint benchmark.
