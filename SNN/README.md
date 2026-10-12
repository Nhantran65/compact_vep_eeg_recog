> Historical notes copied from the original repository. Use ../README.md and ../docs/EXPERIMENTS.md for corrected scope and protocol. EEGPT executable extensions are archived.

# SNN / Spiking Neural Network VEP Experiments

This folder contains the spiking neural network experiment workspace for the
14-channel, 1-second VEP EEG object classification dataset.

Research notes and source links are in [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md).

## Current Goal

The SNN branch is a neuromorphic EEG baseline, not a foundation-model competitor
by default. The first question is whether spike-based temporal dynamics and
spike encodings can produce a serious compact VEP baseline under the same fixed
folds used by the rest of the project.

Target experiments:

1. `Spiking-ShallowConvNet`: convolutional SNN with ShallowConvNet-like temporal
   and spatial convolutions, LIF neurons, surrogate-gradient training.
2. Encoding ablation: `direct`, `rate`, `delta`, `signed_delta`.
3. SNN embedding probe: spike/membrane embeddings followed by
   `StandardScaler + LinearSVC/RBF-SVC/PCA+RBF-SVC`.

Implementation framework: `snnTorch`. SpikingJelly is kept as the v2
replication/full-stack option if the best SNN result becomes worth promoting.

## Final Branch Decision

After V5/V6, report two SNN rows depending on the benchmark framing.

Best accuracy SNN-representation row:

```text
frozen ShallowConvNet temporal feature map [B,40,104]
-> LIF temporal readout
-> spike_count embedding
-> train-only StandardScaler
-> RBF-SVC C=10 gamma=scale
```

Five-fold result:

```text
accuracy       86.02 +/- 0.93%
macro-F1       85.39 +/- 0.82%
balanced acc   85.29 +/- 0.87%
```

Best supervised SNN-hybrid row:

```text
frozen ShallowConvNet temporal feature map [B,40,104]
-> Linear 40->64
-> LIF over 104 steps
-> depthwise temporal gate
-> gated mean membrane
-> spike_count
-> Linear([gated_membrane, spike_count] -> 64) + LayerNorm/GELU/Dropout
-> differentiable RBF/prototype head
```

Five-fold result:

```text
accuracy       84.75 +/- 0.66%
macro-F1       83.94 +/- 0.51%
balanced acc   83.85 +/- 0.41%
```

Interpretation:

- `shallow_temporal_lif_head + spike_count + RBF-SVC` is the strongest SNN
  benchmark row, and is now clearly above the best QNN snapshot
  (`82.61 +/- 0.99%`) while remaining below EEGPT adapter58 RBF-SVC.
- `shallow_temporal_spike_aware_proto_lif_ce` is the strongest supervised SNN
  head, improving over V6b by about `+0.34 pp` accuracy and over the plain
  temporal LIF baseline by about `+2.19 pp`.
- A plug-in EEGPT-token SNN screen over cached EEGPT tokens `[B,4,2048]` did
  not transfer: best fold-1 was `60.86%` acc / `59.73%` macro-F1, so no 5-fold.
- The paper claim should distinguish representation-probe performance from
  end-to-end supervised SNN performance.

## Fold-1 Result Snapshot

Snapshot source: JSON files under `SNN/results/by_config` on 2026-06-02.
All rows use fold 1, train-validation checkpoint selection from the training
fold, and one final evaluation on the fixed test fold.

| Method | Head | Accuracy | Macro-F1 | Balanced Acc | Decision |
|---|---|---:|---:|---:|---|
| `spiking_shallow_direct` | supervised | `34.58%` | `28.27%` | `30.86%` | weak |
| `spiking_shallow_rate` | supervised | `29.74%` | `25.76%` | `28.90%` | weak |
| `spiking_shallow_delta` | supervised | `46.47%` | `45.10%` | `44.85%` | weak |
| `spiking_shallow_signed_delta` | supervised | `46.75%` | `45.53%` | `45.78%` | weak |
| `direct` SNN embedding | RBF-SVC C=10 | `55.05%` | `54.24%` | `54.36%` | weak |
| `rate` SNN embedding | RBF-SVC C=10 | `49.10%` | `47.89%` | `48.01%` | weak |
| `delta` SNN embedding | RBF-SVC C=10 | **`68.46%`** | **`67.83%`** | **`68.06%`** | weak-positive, no 5-fold |
| `signed_delta` SNN embedding | RBF-SVC C=10 | `66.80%` | `66.24%` | `66.58%` | weak-positive, no 5-fold |

Decision:

```text
best fold-1 = delta SNN embedding + RBF-SVC
68.46% acc / 67.83% macro-F1
below the 75-80% promotion band
-> do not run 5-fold for v1
```

Interpretation:

- End-to-end supervised SNN is weak under the current v1 architecture.
- Delta-style encodings are clearly better than direct/rate for this dataset.
- SNN embeddings contain usable nonlinear signal, but not enough to justify
  full 5-fold promotion against ShallowConvNet or QNN.

## V2 Diagnostics And Shallow Hybrid Snapshot

Snapshot source: JSON files under `SNN/results/v2_controls` and
`SNN/results/v2_hybrid` on 2026-06-03. V2 uses fold-specific frozen
ShallowConvNet checkpoints from `QNN/checkpoints/shallow_original_fold{fold}_seed42.pth`.

Classical control results:

| Method | Head | Folds | Accuracy | Macro-F1 | Balanced Acc | Interpretation |
|---|---|---:|---:|---:|---:|---|
| `raw_flat_control` | RBF-SVC C=10 | 5 | `41.42 +/- 1.99%` | `40.19 +/- 2.31%` | `40.19 +/- 2.30%` | matches low raw-feature baseline regime |
| `delta_flat_control` | RBF-SVC C=10 | 5 | `33.80 +/- 1.29%` | `32.36 +/- 1.30%` | `32.26 +/- 1.25%` | delta transform alone is weak |
| `signed_delta_flat_control` | PCA128 + RBF-SVC C=3 | 5 | `50.11 +/- 0.75%` | `48.82 +/- 0.70%` | `48.78 +/- 0.86%` | best raw delta-family control, still far below SNN embedding |

Hybrid results:

| Method | Head | Folds | Accuracy | Macro-F1 | Balanced Acc | Interpretation |
|---|---|---:|---:|---:|---:|---|
| `shallow80_lif_head` | supervised | 5 | `81.53 +/- 0.67%` | `80.69 +/- 0.54%` | `80.56 +/- 0.54%` | viable Shallow+SNN head, below Shallow baseline |
| `shallow80_plif_head` | supervised | 5 | `81.59 +/- 0.67%` | `80.67 +/- 0.56%` | `80.51 +/- 0.57%` | learnable beta/threshold did not materially improve |
| `shallow_temporal_lif_head` | supervised | 5 | **`82.56 +/- 1.57%`** | **`81.82 +/- 1.44%`** | **`81.62 +/- 1.41%`** | previous V2 best; superseded by V5/V6 |

Fold-1 embedding probes:

| Source | Probe head | Accuracy | Macro-F1 | Note |
|---|---|---:|---:|---|
| `shallow80_lif_head` | PCA128 + RBF-SVC C=3 | `81.88%` | `80.85%` | probe close to supervised head |
| `shallow80_plif_head` | PCA128 + RBF-SVC C=3 | `81.88%` | `80.85%` | PLIF probe tied with LIF probe |

Decision:

```text
raw/signed-delta controls are far below delta SNN embedding
-> SNN v1 embedding added representation beyond raw delta transforms

shallow80 LIF/PLIF reached ~81.5% 5-fold
-> hybrid is viable but not stronger than ShallowConvNet

shallow temporal-map LIF reached 82.56% 5-fold
-> keep as the plain temporal LIF baseline for V5/V6 comparisons
```

Interpretation:

- The useful SNN direction is not end-to-end spike encoding from raw EEG.
- The useful direction is frozen ShallowConvNet temporal features followed by a
  spiking temporal readout.
- `shallow_temporal_lif_head` is a serious neuromorphic hybrid baseline, but it
  is still close to existing Shallow/QNN results and should not be framed as a
  decisive improvement without more validation.

## V3 Early-Exit / Latency Snapshot

Snapshot source: JSON files under `SNN/results/v3_early_exit` on 2026-06-03.
This is a post-hoc prefix evaluation of the trained
`shallow_temporal_lif_head` checkpoints. No retraining is used.

The temporal feature map has shape `[40,104]`. Prefix coverage is approximated
as:

```text
coverage_samples = prefix_steps + 24
coverage_ms = coverage_samples / 128 * 1000
```

| Prefix bins | Approx coverage | Acc % | Macro-F1 % | Head ms/sample | Relative head latency | Decision |
|---:|---:|---:|---:|---:|---:|---|
| 8 | `250.0 ms` | `61.60 +/- 2.53` | `61.11 +/- 2.59` | `0.056` | `0.185` | too weak |
| 16 | `312.5 ms` | `69.16 +/- 1.83` | `68.43 +/- 1.86` | `0.049` | `0.163` | too weak |
| 24 | `375.0 ms` | `71.35 +/- 1.72` | `70.54 +/- 1.65` | `0.073` | `0.241` | too weak |
| 32 | `437.5 ms` | `74.09 +/- 2.29` | `73.28 +/- 2.31` | `0.095` | `0.315` | below early gate |
| 48 | `562.5 ms` | `75.77 +/- 2.43` | `74.94 +/- 2.43` | `0.144` | `0.476` | below early gate |
| 64 | `687.5 ms` | `77.08 +/- 1.58` | `76.27 +/- 1.66` | `0.188` | `0.621` | below early gate |
| 80 | `812.5 ms` | `79.13 +/- 1.18` | `78.35 +/- 1.20` | `0.234` | `0.773` | near, but below 80% |
| 104 | `1000.0 ms` | **`82.56 +/- 1.57`** | **`81.82 +/- 1.44`** | `0.303` | `1.000` | full-window result |

Decision:

```text
<=32 bins does not reach 80% accuracy
<=48 bins does not reach 80% accuracy
64 bins is not within 1 pp of full-window accuracy
80 bins is close to 80% but still below it on mean accuracy
-> post-hoc early-exit is negative under the predefined gates
```

Interpretation:

- The trained temporal LIF readout needs most of the VEP window to reach its
  strongest accuracy.
- The useful current claim remains: near-QNN accuracy with much faster
  downstream training.
- Prefix-specific retraining or distillation would be needed before claiming
  early VEP decision.

## V4 Planned Research Extensions

V4 opens two targeted SNN hypotheses. These are not replacements for the final
V2 row unless they clear the promotion gates below.

### V4a Prefix-Specific Distillation

Hypothesis: the negative V3 curve may be a post-hoc limitation. A student SNN
trained only on a short prefix may recover more full-window VEP signal than the
same checkpoint truncated after training.

Protocol:

```text
teacher = fold-specific frozen ShallowConvNet checkpoint
student = ShallowFeatureSNNHead over frozen Shallow temporal map
prefix K = 48, 64, 80 bins
loss = (1-alpha) CE + alpha * T^2 KL(student/T, teacher/T)
defaults: alpha=0.5, temperature=2.0
```

Baselines to beat:

```text
V3 post-hoc K=64: 77.08 +/- 1.58% accuracy
V3 post-hoc K=80: 79.13 +/- 1.18% accuracy
full shallow_temporal_lif_head: 82.56 +/- 1.57% accuracy
```

Promotion gate:

```text
K=64 within 2 pp of the full-window result
or K=80 >= 80% accuracy
```

Commands:

```bash
.venv/bin/python SNN/scripts/run_snn_prefix_distill.py \
  --configs shallow_temporal_lif_distill_k48 shallow_temporal_lif_distill_k64 shallow_temporal_lif_distill_k80 \
  --folds 1 \
  --device auto \
  --no-progress

.venv/bin/python SNN/scripts/summarize_snn_prefix_distill.py
```

Artifacts:

```text
SNN/results/v4_prefix_distill/<config_name>/fold_<k>.json
SNN/artifacts/checkpoints/v4_prefix_distill/<config_name>/fold_<k>.pt
SNN/results/summaries/snn_prefix_distill_summary.csv
SNN/results/summaries/snn_prefix_distill_summary.md
```

Claim discipline: do not claim early VEP decision unless the prefix-specific
student clears the gate. Otherwise report V4a as evidence that the current SNN
still needs most of the VEP window.

### V4b Adaptive Signed-Delta Encoding

Hypothesis: fixed spike encodings may lose useful VEP waveform information. A
small trainable event encoder can learn per-channel gain and positive/negative
thresholds before the existing convolutional SNN.

Protocol:

```text
EEG [B,14,128]
-> trainable gain [14] and positive/negative thresholds [14]
-> soft signed-delta events [B,28,128]
-> SpikingShallowConvNet
```

Baselines to beat:

```text
spiking_shallow_delta fold-1: 46.47% / 45.10% acc/macro-F1
spiking_shallow_signed_delta fold-1: 46.75% / 45.53%
delta SNN embedding + RBF-SVC fold-1: 68.46% / 67.83%
```

Promotion gate:

```text
fold-1 >= 75% accuracy or >= 73% macro-F1 -> run five folds
otherwise keep as an encoding diagnostic
```

Command:

```bash
.venv/bin/python SNN/scripts/run_fold.py \
  --config SNN/configs/spiking_shallow_adaptive_signed_delta.yaml \
  --fold 1 \
  --device auto \
  --no-progress
```

Result path:

```text
SNN/results/by_config/spiking_shallow_adaptive_signed_delta/fold_<k>.json
SNN/artifacts/checkpoints/spiking_shallow_adaptive_signed_delta/fold_<k>.pt
```

Research anchors:

- SCNet adaptive coding for EEG SNN:
  https://www.sciencedirect.com/science/article/pii/S0925231223005933
- Short-length SSVEP knowledge distillation:
  https://www.sciencedirect.com/science/article/pii/S0893608025000127
- SNN knowledge distillation:
  https://ojs.aaai.org/index.php/AAAI/article/view/37175
- SpikeWavformer / wavelet-SNN context:
  https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1652274/abstract

## Comparison Anchors

| Method | Result | Interpretation |
|---|---:|---|
| Classical RBF-SVM raw features | `41.86 +/- 0.87%` accuracy | low classical anchor |
| EEGNet | `52.85%` accuracy | weak compact CNN anchor |
| ShallowConvNet | `82.143 +/- 1.010%` accuracy | strongest conventional CNN anchor |
| Best QNN snapshot | `82.614 +/- 0.994%` accuracy | modest QNN gain over Shallow |
| EEGPT adapter58 RBF-SVC | `87.901 +/- 1.824%` accuracy | strongest foundation-model anchor |

Promotion rule:

```text
fold-1 < 75% acc -> weak SNN evidence; inspect failure, no 5-fold
fold-1 75-80% acc -> continue encoding/readout ablation
fold-1 >= 80% acc or >= 79% macro-F1 -> run 5-fold
5-fold >= 82% acc -> serious neuromorphic baseline against Shallow/QNN
```

## Data Protocol

Project data:

```text
trainfold/fold_{1..5}/train.csv
trainfold/fold_{1..5}/test.csv
1792 features + Class
labels 1..4 -> 0..3
reshape [14,128]
channels = AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
```

SNN supervised training protocol:

```text
train.csv
-> stratified train/validation split
-> fit StandardScaler on train split only
-> transform train/validation/test
-> reshape [B,14,128]
-> spike/current encoder
-> Spiking-ShallowConvNet
-> select checkpoint by validation macro-F1
-> evaluate test fold once
```

The test fold is not used for checkpoint selection.

## Encoding Configs

| Config | Encoder | Input channels | Purpose |
|---|---|---:|---|
| `spiking_shallow_direct` | continuous current | 14 | strongest practical surrogate-gradient baseline |
| `spiking_shallow_rate` | Bernoulli rate from sigmoid amplitude | 14 | classic spike-train conversion control |
| `spiking_shallow_delta` | unsigned change events | 14 | tests event-driven temporal changes |
| `spiking_shallow_signed_delta` | separate positive/negative change events | 28 | preserves direction of VEP waveform changes |

## Commands

Install:

```bash
uv pip install -r SNN/requirements.txt
```

Smoke test before real training:

```bash
.venv/bin/python SNN/scripts/smoke_test.py --fold 1 --device auto
```

Run one supervised fold:

```bash
.venv/bin/python SNN/scripts/run_fold.py \
  --config SNN/configs/spiking_shallow_direct.yaml \
  --fold 1 \
  --device auto
```

Run fold-1 encoding ablation:

```bash
.venv/bin/python SNN/scripts/run_ablation.py \
  --folds 1 \
  --device auto \
  --no-progress
```

Run SNN embedding probe for a trained checkpoint:

```bash
.venv/bin/python SNN/scripts/run_probe.py \
  --config SNN/configs/snn_embedding_probe.yaml \
  --fold 1 \
  --checkpoint SNN/artifacts/checkpoints/spiking_shallow_direct/fold_1.pt \
  --device auto
```

Summarize:

```bash
.venv/bin/python SNN/scripts/summarize_results.py
```

## V2 Commands

Smoke:

```bash
.venv/bin/python SNN/scripts/smoke_snn_v2.py --fold 1 --device auto --no-progress
```

Controls:

```bash
.venv/bin/python SNN/scripts/run_snn_v2_controls.py --folds 1 2 3 4 5
```

Shallow+SNN hybrid:

```bash
.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow80_lif_head shallow80_plif_head \
  --folds 1 \
  --device auto \
  --no-progress
```

Temporal-map hybrid:

```bash
.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_lif_head \
  --folds 1 2 3 4 5 \
  --device auto \
  --no-progress
```

V2 summary:

```bash
.venv/bin/python SNN/scripts/summarize_snn_v2.py
```

## V3 Early-Exit Commands

Smoke:

```bash
.venv/bin/python SNN/scripts/smoke_snn_early_exit.py --fold 1 --device auto
```

Run post-hoc prefix curve:

```bash
.venv/bin/python SNN/scripts/run_snn_early_exit.py \
  --config SNN/configs/shallow_temporal_lif_head.yaml \
  --folds 1 2 3 4 5 \
  --prefix-steps 8 16 24 32 48 64 80 104 \
  --device auto \
  --timing-repeats 5
```

Summarize:

```bash
.venv/bin/python SNN/scripts/summarize_snn_early_exit.py
```

## Artifacts

```text
SNN/results/by_config/<config_name>/fold_<k>.json
SNN/results/v2_controls/<config_name>/fold_<k>.json
SNN/results/v2_hybrid/<config_name>/fold_<k>.json
SNN/results/v3_early_exit/shallow_temporal_lif_head/fold_<k>.json
SNN/results/v4_prefix_distill/<config_name>/fold_<k>.json
SNN/results/summaries/encoding_ablation_fold1.csv
SNN/results/summaries/fivefold_summary.csv
SNN/results/summaries/snn_v2_summary.csv
SNN/results/summaries/snn_v2_summary.md
SNN/results/summaries/snn_early_exit_summary.csv
SNN/results/summaries/snn_prefix_distill_summary.csv
SNN/results/summaries/snn_eegpt_distill_summary.csv
SNN/results/summaries/snn_eegpt_distill_summary.md
SNN/results/summaries/snn_v6_extensions_summary.md
SNN/results/summaries/snn_v6d_proto_summary.md
SNN/results/v5_eegpt_distill/<config_name>/fold_<k>.json
SNN/artifacts/checkpoints/<config_name>/fold_<k>.pt
SNN/artifacts/checkpoints/v2_hybrid/<config_name>/fold_<k>.pt
SNN/artifacts/checkpoints/v4_prefix_distill/<config_name>/fold_<k>.pt
SNN/artifacts/checkpoints/v5_eegpt_distill/<config_name>/fold_<k>.pt
SNN/artifacts/embeddings/<config_name>/fold_<k>_<mode>.npz
```

## Claim Discipline

- Do not claim SNN superiority unless it beats ShallowConvNet/QNN on complete
  5-fold metrics.
- If SNN is below Shallow but uses sparse spikes, report it as an
  energy/latency-oriented neuromorphic baseline, not an accuracy winner.
- If `direct` beats spike-train encodings, the honest interpretation is that
  surrogate-gradient SNN dynamics helped more than hard event conversion.
- If `SNN embedding + RBF-SVC` beats the supervised head, use the same language
  as the foundation-model branches: representation plus classical nonlinear
  head, not end-to-end SNN dominance.
- For the final `shallow_temporal_lif_head`, claim near-QNN accuracy with much
  faster downstream training and a spiking temporal readout over learned EEG
  feature maps.
- Do not claim early VEP decision from the post-hoc prefix curve. Prefix `80`
  is close but remains below the predefined `80%` accuracy gate, and full
  `104` bins are needed for the strongest result.
- For V4a, claim early decoding only if the prefix-trained/distilled student
  clears the documented gate.
- For V4b, claim adaptive encoding only against fixed-encoder controls under the
  same fold protocol.
- For V5a, claim a representation-probe result, not pure end-to-end SNN
  superiority.
- For V6a, claim temporal-gate improvement over the plain temporal LIF baseline.
- For V6b, claim spike-count fusion improves the supervised temporal-gated row.
- For V6d, claim a differentiable prototype/RBF-style supervised SNN head; use
  V6d0 CE-only as the final supervised SNN row.
- For V5b, do not claim EEGPT-teacher distillation improves the SNN student
  under the current logit-margin recipe.

## V5/V6 Five-Fold Results

Snapshot source: JSON files under `SNN/results/v2_hybrid`,
`SNN/results/v5_eegpt_distill`, and `SNN/results/summaries` on 2026-06-04.

Closed diagnostics remain closed: raw encoders, post-hoc early exit,
Shallow-teacher prefix rescue, adaptive signed-delta, and quantum-kernel probes.

Five-fold baseline:

```text
shallow_temporal_lif_head
accuracy = 82.56 +/- 1.57%
macro-F1 = 81.82 +/- 1.44%
```

| Branch | Best result | Gate / Baseline | Decision |
|---|---:|---|---|
| `V5a temporal_lif_embedding_probe` | `86.02 +/- 0.93%` acc / `85.39 +/- 0.82%` macro-F1 with `spike_count + RBF-SVC C=10` | best QNN snapshot `82.61 +/- 0.99%` acc | promote as best SNN representation row |
| `V5b eegpt_teacher_distill_full104` | fold-1 student `82.57%` acc / `81.65%` macro-F1; teacher `89.35%` acc | fold-1 gate `>=84.0%` acc or `>=83.0%` macro-F1 | neutral/close for this recipe |
| `V6a shallow_temporal_gated_lif` | `83.50 +/- 0.89%` acc / `82.83 +/- 0.84%` macro-F1 | plain temporal LIF `82.56 +/- 1.57%` acc | positive; superseded by V6b |
| `V6b shallow_temporal_spike_aware_gated_lif` | `84.41 +/- 1.29%` acc / `83.73 +/- 1.13%` macro-F1 | V6a `83.50 +/- 0.89%` acc | positive; superseded by V6d |
| `V6d0 shallow_temporal_spike_aware_proto_lif_ce` | `84.75 +/- 0.66%` acc / `83.94 +/- 0.51%` macro-F1 | V6b `84.41 +/- 1.29%` acc | promote as best supervised SNN row |

Interpretation:

- V5a is the strongest SNN result. The SNN head is useful as a representation
  learner when paired with a classical nonlinear probe.
- V5b confirms the EEGPT teacher is strong, but logit-margin distillation into
  the current student did not transfer enough accuracy.
- V6a suggests uniform mean membrane pooling is a limitation; V6b shows that
  spike-count fusion helps; V6d shows that an end-to-end prototype/RBF-style
  head is stronger than the linear neural classifier.

Executed commands:

```bash
.venv/bin/python SNN/scripts/run_shallow_snn_probe.py \
  --config shallow_temporal_lif_head \
  --folds 1 2 3 4 5 \
  --modes spike_count mean_membrane final_membrane max_membrane concat \
  --device auto

.venv/bin/python SNN/scripts/run_snn_eegpt_distill.py \
  --config shallow_temporal_lif_eegpt_distill_full104 \
  --folds 1 \
  --device auto \
  --no-progress

.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_gated_lif \
  --folds 1 2 3 4 5 \
  --device auto \
  --no-progress
```

## V6/V5a Extension Screen

Snapshot source: `SNN/results/summaries/snn_v6_extensions_summary.md` on
2026-06-07.

Promotion gates:

```text
V6 supervised: fold-1 >=84.0% acc or >=83.3% macro-F1 -> run 5-fold
V5a probe/fusion: fold-1 >86.5% acc or >85.8% macro-F1 -> run 5-fold
```

| Branch | Row | Fold-1 result | Decision |
|---|---|---:|---|
| `V6b` | `shallow_temporal_spike_aware_gated_lif` | `83.68%` acc / `82.99%` macro-F1 | close, user-requested 5-fold run |
| `V6c` | `shallow_temporal_windowed_spike_lif_4bins` | `82.02%` acc / `81.13%` macro-F1 | below gate |
| `V6e` | `shallow_temporal_multiscale_gated_lif` | `82.99%` acc / `82.18%` macro-F1 | below gate |
| `V5a-ext` | `windowed_spike_count_4 + RBF-SVC` | `83.54%` acc / `82.72%` macro-F1 | below gate |
| `V5a-ext` | `spike_mean_late_fusion` | `85.34%` acc / `84.60%` macro-F1 | below gate |

Decision:

```text
V6b missed the original fold-1 gate by about 0.31 pp macro-F1, but the
user-requested 5-fold run reached 84.41 +/- 1.29% acc / 83.73 +/- 1.13%
macro-F1. V6b was promoted at this step, then superseded by V6d.
Keep V5a spike_count + RBF-SVC as the strongest SNN representation row.
Do not promote V6c/V6e or V5a windowed/fusion probes.
```

Executed extension commands:

```bash
.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_spike_aware_gated_lif shallow_temporal_windowed_spike_lif_4bins \
  --folds 1 \
  --device auto \
  --no-progress

.venv/bin/python SNN/scripts/run_shallow_snn_probe.py \
  --config shallow_temporal_lif_head \
  --folds 1 \
  --modes windowed_spike_count_4 spike_mean_late_fusion \
  --device auto

.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_multiscale_gated_lif \
  --folds 1 \
  --device auto \
  --no-progress

.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_spike_aware_gated_lif \
  --folds 2 3 4 5 \
  --device auto \
  --no-progress
```

## V6d Differentiable Prototype Head

Snapshot source: `SNN/results/summaries/snn_v6d_proto_summary.md` on
2026-06-07.

| Branch | Best result | Decision |
|---|---:|---|
| `V6d0 proto_lif_ce` | `84.75 +/- 0.66%` acc / `83.94 +/- 0.51%` macro-F1 | promote as best supervised SNN row |
| `V6d1 proto_lif_supcon005` | `84.08 +/- 1.21%` acc / `83.31 +/- 1.08%` macro-F1 | below V6b/V6d0 |
| `V6d2 proto_lif_supcon010` | `84.22 +/- 0.76%` acc / `83.55 +/- 0.64%` macro-F1 | below V6b/V6d0 |
| `V6d3 proto_lif_supcon005_sep` | `84.22 +/- 0.84%` acc / `83.49 +/- 0.72%` macro-F1 | below V6b/V6d0 |

Decision:

```text
V6d0 CE-only replaces V6b as the supervised SNN architecture row.
It does not clear the strong-claim gate of >=85.0% acc or >=84.3% macro-F1.
SupCon and prototype-separation regularization did not improve this dataset.
V5a spike_count + RBF-SVC remains the strongest SNN representation row.
```

Executed V6d commands:

```bash
.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_spike_aware_proto_lif_ce shallow_temporal_spike_aware_proto_lif_supcon005 shallow_temporal_spike_aware_proto_lif_supcon010 shallow_temporal_spike_aware_proto_lif_supcon005_sep \
  --folds 1 \
  --device auto \
  --no-progress

.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_spike_aware_proto_lif_ce shallow_temporal_spike_aware_proto_lif_supcon005 shallow_temporal_spike_aware_proto_lif_supcon010 shallow_temporal_spike_aware_proto_lif_supcon005_sep \
  --folds 2 3 4 5 \
  --device auto \
  --no-progress
```

## EEGPT Token SNN Plug-in Screen

Snapshot source: `SNN/results/summaries/snn_fm_token_summary.md` on
2026-06-08. This uses cached EEGPT adapter58 token artifacts only; no EEGPT
encoder rerun.

```text
EEGPT cached tokens [B,4,2048]
-> Linear 2048->64
-> LIF over 4 token steps
-> gated mean membrane + spike_count fusion
-> linear or differentiable prototype head
```

| Branch | Fold-1 result | Gate | Decision |
|---|---:|---|---|
| `eegpt_token_spike_aware_gated_lif` | `60.86%` acc / `59.73%` macro-F1 | `>=84.0%` acc or `>=83.5%` macro-F1 | no 5-fold |
| `eegpt_token_spike_aware_proto_lif_ce` | `59.61%` acc / `58.64%` macro-F1 | same | no 5-fold |

Decision:

```text
Do not promote EEGPT-token SNN as a foundation-model plug-in head.
The likely issue is interface shape: EEGPT cache exposes only 4 token steps,
while the successful Shallow SNN path uses a dense VEP temporal map [B,40,104].
This supports a narrower claim: the SNN head is strong when the frozen feature
map preserves enough temporal dynamics, but it is not universally plug-and-play
over arbitrary foundation-model token caches.
```

Executed commands:

```bash
.venv/bin/python SNN/scripts/run_eegpt_token_snn.py \
  --configs eegpt_token_spike_aware_gated_lif eegpt_token_spike_aware_proto_lif_ce \
  --folds 1 \
  --device auto \
  --no-progress

.venv/bin/python SNN/scripts/summarize_snn_fm_token.py
```
