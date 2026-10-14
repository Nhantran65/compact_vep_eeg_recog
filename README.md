# Compact VEP EEG Recognition

A standalone project for recognizing Apple, Car, Flower, and Human Face from 14-channel, 128 Hz EEG in 1-second windows. It was extracted from EEG-Foundation-Model-Paper on 12/10/2026, following the ML/DL/SNN scope of `docs/MSc__Supervision.pdf`.

**Read first:** [Experiment summary and audit](docs/EXPERIMENTS.md). The 86.02% score is a historical result on epoch-level folds, not accuracy on unseen subjects. The legacy DL backbones used the test set for checkpoint selection. The probe uses an SNN that was trained with labels. The brief contains descriptions that do not match the actual code.

## Components

- `SNN/src/snn_vep/`: standalone SNN code and Shallow backbone; does not import QNN.
- `SNN/configs/`, `SNN/scripts/`: configs and CLIs for training, probing, ablation, early exit, and Shallow-teacher distillation.
- `SNN/results/`, `SNN/artifacts/`: original results, checkpoints, and embeddings; historical JSON is kept unchanged.
- `baselines/checkpoints/`: 5 Shallow backbones corresponding to the historical folds.
- `results/`: evidence for classical ML, DeepConvNet, and ShallowConvNet.
- `CSV-FEATURES/`: 59 CSVs restored from Git, 3,612 epochs; includes Class/Phase/Subject.
- `trainfold/`: historical folds rebuilt from the file order in notebook output, with added metadata and hashes. Do not use for subject-generalization claims.
- `evidence/subject_disjoint/`: manifest and subject-disjoint split from later research, separate from `trainfold`.
- `evidence/later_benchmarks/`: later research reports kept for provenance; not runners for this project.
- `archive/notebooks/`, `archive/baseline_scripts/`: original ML/DL with saved outputs. When running notebooks, set the kernel cwd to the project root before cells that read data; baseline scripts read `trainfold` relative to cwd. The original DeepConvNet used the test set for checkpoint selection and is only for historical reproduction.
- `archive/foundation_extensions/`: source for the negative EEGPT screens; requires an external cache and is not part of the standalone runtime.
- `preprocessing/`: original MATLAB/EEGLAB code. Raw EDF/SET files are not included; running preprocessing from scratch requires the original data, EEGLAB/ICLabel, and adjusting MATLAB paths.

## Install and run

Run commands from the project root. Python >=3.10; the tested environment is detailed in `docs/environment.json`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/inventory_results.py
python scripts/verify_project.py
```

Reproduce the spike-count probe on existing checkpoints, writing new results to `runs/`:

```bash
python SNN/scripts/run_shallow_snn_probe.py \
  --config shallow_temporal_lif_head --folds 1 2 3 4 5 \
  --modes spike_count --device cpu --results-dir runs/probe/results
```

Retrain the supervised head on the packaged backbone:

```bash
python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_spike_aware_proto_lif_ce --folds 1 2 3 4 5 \
  --device auto --results-dir runs/proto/results \
  --checkpoints-dir runs/proto/checkpoints --no-progress
```

Run the historical classical baselines (requires the `classical` extra; see `--help`):

```bash
python -m pip install -e '.[classical]'
python archive/baseline_scripts/run_classical_ml_baselines.py --help
```

If the fold directories are removed to save space, rebuild them with `python scripts/prepare_legacy_folds.py`. The script refuses to overwrite existing folds. Keep all 59 CSVs and `docs/legacy_input_order.json`.

The original CLIs may overwrite results/checkpoints when using defaults: always pass separate `--results-dir` and `--checkpoints-dir` when training anew. Some summarizers write by default to `SNN/results/summaries`; use a fresh inventory to check the original data.

## Continuing the research

A new benchmark on subject-disjoint train/validation/test with train-only preprocessing is required before concluding that SNNs help generalization. The manifest exists, but the heads/backbones in the package are still tied to the historical folds. No runner has been implemented for the subject-disjoint protocol yet, and there are no experiments on the two C3 datasets in the brief. Do not mix the two protocols or reuse epoch-fold backbones for new subject folds.

Original data: Mendeley Data, DOI `10.17632/g9shp2gxhy.2`.
