# Paper Contributions - VEP EEG Benchmark

Last updated: 2026-06-08

The paper's goal should be framed as:

```text
A controlled benchmark for compact low-channel VEP EEG, using the same fixed
folds to compare conventional EEG networks, QNN/QML, SNN, and EEG foundation
models; the focus is interface alignment and transfer diagnostics, not just
leaderboard accuracy.
```

## 1. Main Thesis

The strongest thesis:

```text
In compact 14-channel VEP EEG, transfer effectiveness depends more on correctly
aligning the model's pretrained input contract than on whether the model has a
dynamic-channel interface or a larger size.
```

In short:

```text
Interface alignment beats nominal channel flexibility.
```

Shared dataset/protocol:

```text
Task: 4-class visual/VEP EEG classification
Classes: Apple, Car, Flower, Human Face
Device: Emotiv EPOC-X, 14 channels, 128 Hz
Epoch: 1 second, [14,128]
Samples: 3612 epochs
Evaluation: fixed trainfold/fold_{1..5}
Rule: fit scaler/PCA/probe preprocessing on train fold only
```

Metric policy for the main benchmark:

- **Accuracy**: overall performance on the entire test fold.
- **Macro-F1**: balances precision/recall across the 4 classes, so majority classes do not dominate.
- **Balanced Accuracy**: mean per-class recall, measuring the ability to recognize each class more fairly than accuracy.
- **Cohen's Kappa**: agreement above chance level after correcting for the true and predicted class distributions.

## 2. Main Results

| Model | Scope | Accuracy | Macro-F1 | Balanced Acc | Cohen's Kappa | Meaning |
|---|---|---:|---:|---:|---:|---|
| EEGPT adapter58 + RBF-SVC | 5-fold | `87.90 +/- 1.82%` | `87.45 +/- 2.21%` | `87.23 +/- 2.50%` | `83.64 +/- 2.51%` | strongest foundation-model row |
| SNN temporal LIF `spike_count` + RBF-SVC | 5-fold | `86.02 +/- 0.93%` | `85.39 +/- 0.82%` | `85.29 +/- 0.87%` | `81.12 +/- 1.25%` | strongest SNN representation row |
| SNN differentiable prototype LIF | 5-fold | `84.75 +/- 0.66%` | `83.94 +/- 0.51%` | `83.85 +/- 0.41%` | `79.40 +/- 0.85%` | strongest supervised SNN row |
| QNN `smooth_cosine/t2_temporal_qdelta` | 5-fold | `82.61 +/- 0.99%` | `81.83 +/- 0.92%` | `81.75 +/- 0.86%` | `76.52 +/- 1.33%` | modest QNN gain, not quantum-advantage evidence |
| ShallowConvNet `shallow_original` | 5-fold | `82.14 +/- 1.01%` | `81.33 +/- 0.88%` | `81.20 +/- 0.81%` | `75.88 +/- 1.34%` | strong local EEG baseline |
| LaBraM V2a stretch_3000 + PCA/RBF | 5-fold | `74.97 +/- 0.86%` | `74.29 +/- 1.12%` | `74.04 +/- 1.24%` | `66.14 +/- 1.22%` | strong secondary FM after temporal alignment |
| BrainOmni V6 adapter | fold 1 only | `62.10%` | `61.43%` | `61.69%` | `49.10%` | weak-positive dynamic/flexible transfer |
| REVE V6 adapter | fold 1 only | `61.27%` | `59.96%` | `59.81%` | `47.56%` | weak-positive dynamic-channel transfer |
| NeuroLM V1/V2 best | fold 1 only | `53.25%` | `52.22%` | `52.07%` | `36.74%` | named-channel token interface not enough |
| Raw RBF-SVM | 5-fold baseline | `41.86 +/- 0.87%` | `37.41 +/- 1.53%` | `38.51 +/- 1.11%` | `~19.51%` | low classical baseline |

Kappa is computed post hoc from the confusion matrix:

```text
kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
```

For 5-fold rows with fold-level confusion matrices, Kappa is mean +/- sample std
over 5 folds. For BrainOmni/REVE/NeuroLM, these are fold-1 results, so there is
no std. Raw RBF-SVM only has an averaged confusion matrix in the old artifact,
so its Kappa is estimated from the averaged confusion matrix; rerun the SVM
baseline separately to get fold-wise Kappa mean +/- std for this row.

## 3. Transfer-Time / Efficiency Finding

Timing should be added, but it must be clearly labelled as a **recorded
wall-clock transfer component**, not a standardized inference/energy benchmark.
The branches do not log the same kind of time: SNN/QNN mainly log downstream
head training, while foundation models have adapter training, embedding
extraction/caching, and probe fitting.

| Group | Representative row | Accuracy | Recorded transfer time | Interpretation |
|---|---|---:|---:|---|
| SNN representation | temporal LIF `spike_count` + RBF-SVC | `86.02 +/- 0.93%` | RBF fit `~0.11 s/fold`; temporal LIF checkpoint training `~1.75 min/fold` if counted from scratch | fastest strong downstream representation readout |
| SNN supervised | differentiable prototype LIF | `84.75 +/- 0.66%` | `~2.31 min/fold` | fastest strong supervised neuromorphic head |
| QNN/QML | `t2_temporal_qdelta` label smoothing + cosine | `82.61 +/- 0.99%` | `~84.4 min/fold` | much slower downstream head; modest accuracy gain |
| Foundation model | EEGPT adapter58 + RBF-SVC | `87.90 +/- 1.82%` | adapter train `~4.74 min/fold`; RBF fit `~5.84 s/fold`; embedding extraction not normalized here | strongest accuracy, but timing depends on cached embedding pipeline |

Claim to use:

```text
SNN provides the best accuracy-time trade-off among non-foundation downstream
heads: it reaches the QNN/foundation-model accuracy regime with minute-scale
training, while QNN recipes require tens of minutes per fold.
```

Claim to avoid:

```text
SNN is absolutely fastest end-to-end across all model families.
```

Reason to avoid: foundation-model timing needs to separate adapter training,
frozen backbone encoding, cached embedding reuse, and probe fitting. A more
rigorous runtime claim requires a standardized timing benchmark that reruns the
full pipeline on the same hardware.

## 4. Main Contributions

### Contribution 1: Controlled Cross-Family Benchmark

The paper can claim:

```text
We build a controlled benchmark for compact 14-channel VEP EEG classification,
evaluating conventional EEG networks, QNN/QML, SNN, and EEG foundation models
on the same fixed folds with train-only preprocessing.
```

Value:

- Same split and metrics across many model families.
- Includes both positive and negative results.
- Separates three questions: accuracy, transfer suitability, and model-family claims.

### Contribution 2: Interface-Contract Taxonomy

Instead of classifying models only by architecture, the project classifies them
by pretrained input contract:

| Interface contract | Models | Finding |
|---|---|---|
| Fixed hard channel contract | EEGPT | very strong when the 14->58 channel + sampling contract is fixed correctly |
| Fixed canonical montage/temporal contract | LaBraM | weak if the temporal interface is wrong; becomes a secondary baseline when aligned correctly |
| Dynamic channel + electrode positions | REVE, BrainOmni | convenient interface but only weak-positive on this compact VEP |
| Named-channel token interface | NeuroLM | named tokens are not enough to rescue short 1s Emotiv14 VEP |
| Dataset-trained local frontend | Shallow/QNN/SNN | strong local baseline, not pretrained FM transfer |

Main claim:

```text
Dynamic-channel support is an interface convenience, not sufficient evidence of
representation suitability for compact VEP EEG.
```

### Contribution 3: Frozen-Probe-First Transfer Protocol

A reusable protocol for new EEG datasets:

```text
fixed folds
-> train-only scaler
-> minimal correct model interface
-> frozen embeddings
-> LinearSVC / RBF-SVC / PCA+RBF
-> only then train adapter or fine-tune
```

Because the dataset is small, neural heads/fine-tuning overfit easily. Frozen
probes help decide early which branches are worth running 5-fold or with an
adapter.

Evidence:

- EEGPT frozen/adapter embeddings + RBF-SVC is the strongest FM path.
- LaBraM is only strong after fixing the temporal interface.
- REVE/BrainOmni/NeuroLM did not reach promotion despite adapter/interface variants.
- SNN V5a follows the same logic: SNN temporal dynamics produce a good
  representation, and RBF-SVC reads it better than a supervised linear head.

### Contribution 4: Conditional Adapter Gains

Adapters do not always help. They are only worthwhile when they fix a specific mismatch.

| Branch | Intervention | Outcome |
|---|---|---|
| EEGPT | `adapter58` 14->58 + resample | large gain, final `87.90%` |
| LaBraM | temporal `stretch_3000` | main rescue to `74.97%` |
| LaBraM channel adapters | extra channel/interpolation adapters | mostly neutral |
| REVE/BrainOmni adapters | position/embedding adapters | weak-positive only |
| NeuroLM adapters | raw-token adapters | neutral or harmful |
| EEGPT-QDelta | post-RBF residual/QDelta | neutral, about `+0.083 pp` |

Claim:

```text
Adapters are useful when they correct a concrete pretrained-interface mismatch;
they do not reliably rescue weak representations.
```

### Contribution 5: SNN as Efficient Temporal Representation Learning

SNN branch now has two distinct contributions:

| SNN row | Meaning | Result |
|---|---|---:|
| V5a temporal LIF `spike_count` + RBF-SVC | SNN representation probe | `86.02 +/- 0.93%` acc |
| V6d differentiable prototype LIF | supervised SNN architecture | `84.75 +/- 0.66%` acc |
| Plain temporal LIF | previous baseline | `82.56 +/- 1.57%` acc |

Interpretation:

- V5a shows that LIF spike dynamics encode a strong VEP representation when read
  by a nonlinear classical probe.
- V6a shows that uniform mean membrane readout was a bottleneck; V6b improves
  this with spike-count fusion; V6d improves the supervised row further with a
  differentiable prototype/RBF-style head.
- Raw SNN encoders remain weak; useful SNN path is over learned Shallow temporal
  features, not raw EEG spike coding.
- A direct EEGPT-token plug-in screen failed (`60.86%` fold-1 best), so the SNN
  head should be framed as requiring a rich temporal feature map, not arbitrary
  foundation-model tokens.
- V6d SupCon/prototype-separation variants were neutral, but the CE-only
  differentiable prototype head improved over V6b.

Claim discipline:

```text
V5a is a representation-probe result, not pure end-to-end SNN superiority.
V6d is the cleaner supervised SNN-hybrid architecture result.
```

### Contribution 6: Controlled Negative Results

Negative/neutral results are part of the contribution because they rule out overclaims:

- Dynamic-channel REVE/BrainOmni do not automatically transfer strongly.
- NeuroLM named-channel tokens do not fit the compact 1s VEP protocol.
- QNN gains around Shallow are modest; there is no quantum advantage.
- Quantum kernel on SNN embeddings loses to the matched RBF-SVC control.
- EEGPT-QDelta is nearly neutral.
- Raw SNN/event encoders are weak.
- SNN prefix/early-exit has not passed its gate, so no early VEP decision is claimed.
- The current EEGPT-teacher distillation into SNN did not pass its gate despite a strong teacher.
- V6c/V6e supervised extensions and V5a windowed/fusion probes did not beat the
  predefined fold-1 gates.
- V6d SupCon/prototype-separation regularization did not improve on CE-only.
- The EEGPT-token SNN plug-in head did not pass the fold-1 gate; do not claim the
  SNN head is a universal foundation-model readout.

## 5. Suggested Contribution Paragraph

The contributions section of the paper could be written as follows:

```text
We present a controlled benchmark for compact low-channel VEP EEG
classification across conventional EEG networks, QNN/QML, SNN, and EEG
foundation models under fixed folds and train-only preprocessing. We organize
EEG model transfer by pretrained input-interface contracts and show that
interface alignment is a first-order factor: fixed-contract models such as
EEGPT and LaBraM become strong after targeted channel/temporal alignment,
whereas dynamic-channel or named-token interfaces alone do not guarantee
transfer. We use frozen representation probing as a low-cost suitability test
before adapter training, showing that adapters help when they correct concrete
contract mismatch but often fail to rescue weak representations. We also report
recorded transfer-time components, showing that SNN temporal readouts provide a
strong accuracy-time trade-off relative to much slower QNN downstream heads.
Finally, we provide cross-family diagnostics, including strong SNN temporal
representation probes and controlled negative results for raw SNN encoders,
quantum kernels, and weak foundation-model transfer paths.
```

## 6. Claims To Use

- EEGPT adapter58 + RBF-SVC is the strongest foundation-model baseline.
- V5a SNN temporal LIF `spike_count` + RBF-SVC is the strongest SNN representation
  row.
- V6d differentiable prototype LIF is the strongest supervised SNN row.
- The EEGPT-token SNN plug-in screen is a negative control: best fold-1 is only
  `60.86%` acc / `59.73%` macro-F1, so it was not run 5-fold.
- The V6/V5a extension screen is mostly negative/neutral; V6d CE-only is the latest
  supervised SNN improvement.
- Interface alignment is the decisive factor in compact VEP transfer.
- Dynamic-channel support does not guarantee a suitable representation.
- Frozen probing should precede expensive adapter training/fine-tuning.
- Adapter gains are conditional, not a default.

## 7. Claims To Avoid

- Do not claim dynamic-channel models are weak in general; only say that in this
  compact Emotiv14 VEP benchmark they did not transfer strongly.
- Do not claim V5a is an end-to-end supervised SNN; it is an SNN representation +
  classical nonlinear probe.
- Do not claim quantum advantage without beating matched classical controls.
- Do not claim EEGPT-QDelta gives a meaningful improvement.
- Do not claim SNN early VEP decision; prefix gates were not met.
- Do not claim the V6-style SNN head is plug-and-play on every foundation-model
  token cache; the EEGPT token screen did not pass its gate.
- Do not claim QNN gains are robust without multi-seed validation.

## 8. Practical Takeaway

For a new EEG dataset, the reusable part is not `adapter58`, `stretch_3000`, or
any specific hyperparameter. The reusable part is the protocol:

```text
define dataset contract
-> fixed folds
-> train-only preprocessing
-> explicit pretrained input contract
-> minimal interface alignment
-> frozen embedding probes
-> adapter only when mismatch is concrete
-> matched controls for QNN/SNN/fine-tuning claims
```

This is a contribution that is meaningful beyond the current dataset.
