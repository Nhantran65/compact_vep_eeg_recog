# Literature Review: SNNs For VEP EEG

Last updated: 2026-06-07

This note tracks the research basis for applying spiking neural networks to the
project's low-channel VEP EEG classification task:

```text
Emotiv14 VEP EEG [14,128]
-> spike/current encoding or frozen Shallow temporal feature map
-> convolutional SNN / LIF temporal readout / SNN embedding
-> supervised head or classical SVC probe
```

## Final Branch Decision

Use `snnTorch` for this branch. After V5/V6, the final report should separate
the best representation-probe row from the best supervised SNN-hybrid row.

Best SNN representation row:

```text
EEG [B,1,14,128]
-> frozen ShallowConvNet conv1/conv2/BatchNorm/square
-> temporal feature map [B,40,104]
-> LIF temporal readout
-> spike_count embedding
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
EEG [B,1,14,128]
-> frozen ShallowConvNet conv1/conv2/BatchNorm/square
-> temporal feature map [B,40,104]
-> LIF temporal readout
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

The representation-probe row is clearly above the best QNN snapshot
(`82.61 +/- 0.99%`) and below EEGPT adapter58 RBF-SVC. The supervised
prototype-head row improves over the plain temporal LIF baseline
(`82.56 +/- 1.57%`) and is the cleaner end-to-end SNN-hybrid architecture
claim.

Framework reasoning:

- The project needs a clean, reproducible EEG experiment more than a full-stack
  neuromorphic deployment stack.
- `snnTorch` is PyTorch-native and directly supports surrogate-gradient
  training, which is the right default for small supervised EEG datasets.
- SpikingJelly has stronger full-stack infrastructure, CUDA acceleration,
  ANN2SNN, and deployment tooling. It should be the v2 replication option if
  the SNN result becomes strong enough to promote.

## Why SNN Is Plausible For This Dataset

Dataset facts:

```text
4-class VEP object classification
14 Emotiv channels
128 Hz
1 second epochs
3612 epochs
fixed 5 folds
```

SNNs are plausible because VEP is time-locked and transient. The model can use
membrane dynamics and spike counts over the 128-sample response window, while
encoding ablations test whether continuous EEG should be used as direct current
or converted into event-like spike trains.

However, the expected contribution is not "SNN beats EEGPT" by default.
The realistic first contribution is a controlled neuromorphic baseline against
ShallowConvNet and QNN under the same folds.

## Empirical Findings

V1 raw/end-to-end SNN:

| Method | Scope | Result | Interpretation |
|---|---|---:|---|
| `spiking_shallow_direct` | fold 1 | `34.58% / 28.27%` acc/macro-F1 | weak |
| `spiking_shallow_delta` | fold 1 | `46.47% / 45.10%` | weak |
| `spiking_shallow_signed_delta` | fold 1 | `46.75% / 45.53%` | weak |
| `delta SNN embedding + RBF-SVC` | fold 1 | `68.46% / 67.83%` | weak-positive representation signal |

V2 controls and hybrids:

| Method | Folds | Result | Interpretation |
|---|---:|---:|---|
| `delta_flat_control + RBF-SVC` | 5 | `33.80 +/- 1.29%` acc | delta transform alone is weak |
| `signed_delta_flat_control + PCA/RBF` | 5 | `50.11 +/- 0.75%` acc | best raw delta-family control |
| `shallow80_lif_head` | 5 | `81.53 +/- 0.67%` acc | viable, but below Shallow |
| `shallow80_plif_head` | 5 | `81.59 +/- 0.67%` acc | learnable beta/threshold did not materially help |
| `shallow_temporal_lif_head` | 5 | `82.56 +/- 1.57%` acc | previous V2 best; V5/V6 baseline |

V3 post-hoc early-exit:

| Prefix bins | Approx coverage | Accuracy | Interpretation |
|---:|---:|---:|---|
| 32 | `437.5 ms` | `74.09 +/- 2.29%` | below early gate |
| 48 | `562.5 ms` | `75.77 +/- 2.43%` | below early gate |
| 64 | `687.5 ms` | `77.08 +/- 1.58%` | not within 1 pp of full window |
| 80 | `812.5 ms` | `79.13 +/- 1.18%` | near, but below 80% |
| 104 | `1000.0 ms` | `82.56 +/- 1.57%` | full-window result |

Conclusion:

```text
post-hoc early-exit is negative under the predefined gates
-> do not claim early VEP decision
-> keep the efficiency claim around training time and temporal readout
```

## V4 Research Extensions

V4 keeps the final V2 result as the current reported SNN row, but opens two
targeted hypotheses that are more informative than adding another generic SNN
head.

### Prefix-Specific Distillation

The V3 early-exit curve is post-hoc: it truncates checkpoints trained on the full
temporal feature map. Short-window VEP decoding may need a student trained
specifically on the available prefix. The planned V4a experiment follows the
short-length VEP/SSVEP distillation literature and general SNN knowledge
distillation work:

```text
teacher = frozen full-window ShallowConvNet
student = shallow_temporal_lif_head with prefix K
K = 48, 64, 80
loss = CE + logit distillation
```

The result should be interpreted as a latency/early-decision experiment, not as
a replacement for the full-window SNN unless it clears the predefined gate:

```text
K=64 within 2 pp of full-window accuracy
or K=80 >= 80% accuracy
```

If the gate is not cleared, the conclusion is still useful: under this compact
VEP protocol the current SNN temporal readout needs most of the 1-second window.

### Adaptive Signed-Delta Encoding

The V1 fixed encodings showed that raw end-to-end SNN is weak and that
delta-style SNN embeddings contain some signal. V4b tests whether the weakness
comes from fixed thresholds rather than the SNN architecture:

```text
EEG [14,128]
-> trainable channel gain and positive/negative thresholds
-> soft signed-delta events [28,128]
-> SpikingShallowConvNet
```

This is motivated by adaptive coding in convolutional SNNs for EEG motor imagery
and by the broader observation that EEG-to-spike conversion can dominate SNN
performance. It should be compared directly against `spiking_shallow_delta`,
`spiking_shallow_signed_delta`, and the `delta SNN embedding + RBF-SVC` fold-1
row. Do not run five folds unless fold 1 reaches at least `75%` accuracy or
`73%` macro-F1.

## Literature Anchors

| Work | Relevance To This Branch |
|---|---|
| Choi 2024 biomedical SNN survey | Summarizes SNN architectures, training approaches, and EEG/ECG/EMG biomedical signal use cases. |
| snnTorch surrogate-gradient documentation | Practical basis for training spiking layers despite non-differentiable spike functions. |
| SpikingJelly framework paper | Full-stack PyTorch SNN ecosystem; useful for future replication or energy/deployment extensions. |
| CSNN for EEG anticipatory brain potentials | Direct evidence that convolutional SNNs can classify EEG temporal events and that delta-modulated spike trains are a meaningful ablation. |
| SCNet for EEG motor imagery | Supports convolutional SNN design for EEG decoding and motivates CNN-like feature extraction plus spiking dynamics. |
| SpikeWavformer / SNN EEG wavelet-spiking attention | Recent direction combining EEG signal transforms with spiking sequence models; relevant as a later, not v1, extension. |
| RMKD short-length SSVEP distillation | Supports prefix-specific teacher/student training when short VEP windows lose information. |
| SNN knowledge distillation studies | Motivate logit-level distillation from ANN/full-window teachers to SNN students. |

## Encoding Hypotheses

The main experimental risk is not only the SNN architecture. It is the conversion
from continuous EEG into the signal consumed by spiking neurons.

Planned encodings:

| Encoding | Hypothesis | Risk |
|---|---|---|
| `direct` | Continuous standardized EEG current keeps amplitude information and is best for surrogate-gradient training. | Less neuromorphic because input is not a hard spike train. |
| `rate` | Spike rate approximates amplitude and gives a classic SNN input. | A 1-second, 128-step epoch may be too short for rate coding. |
| `delta` | VEP class signal may live in transient changes, so event-like change spikes may be compact. | Unsigned events lose waveform direction. |
| `signed_delta` | Separating positive and negative changes preserves waveform direction and event sparsity. | Doubles input channels and may be too sparse if threshold is high. |

Interpretation rule:

```text
direct wins -> SNN dynamics helped, but hard spike conversion did not
delta/signed_delta wins -> event-like VEP coding is useful
rate wins -> amplitude-to-rate coding is sufficient for this epoch length
all weak -> SNN is not competitive under this low-data VEP protocol
```

## Architecture Rationale

The v1 model should stay close to ShallowConvNet because that is the strongest
conventional baseline in the project. The planned SNN uses:

```text
[B,C,T]
-> temporal Conv2d over time
-> spatial Conv2d over channels
-> BatchNorm
-> LIF neuron unrolled across time
-> spike/membrane readout
-> 4-class classifier
```

This isolates the research question:

```text
same EEG inductive bias as ShallowConvNet
+ spiking dynamics
+ encoding ablation
```

It avoids starting with a spiking transformer, reservoir, or foundation model,
which would add complexity before proving the basic SNN signal.

## Why The Temporal Hybrid Worked

The successful V2 design does not run SNN directly on raw EEG. It uses
ShallowConvNet as a frozen EEG feature frontend and places the LIF readout before
Shallow's pooling/log/flatten compression:

```text
Shallow/QNN path:
  [B,40,104] -> AvgPool/log/flatten -> [B,80] -> head

SNN temporal path:
  [B,40,104] -> LIF unroll over 104 bins -> head
```

This preserves temporal resolution for the spiking readout. The `shallow80`
variants compress time to `40 filters x 2 pooled windows`, which gives the LIF
head only two effective timesteps. The temporal-map variant gives LIF 104
timesteps and produced the strongest V2 SNN result.

## Probe Rationale

The foundation-model branches show that frozen/fixed representations plus
`StandardScaler + RBF-SVC` can be stronger than neural heads on this small
dataset. Therefore SNN should also be tested as a representation learner:

```text
trained SNN
-> spike_count / mean_membrane / final_membrane / concat embedding
-> StandardScaler
-> LinearSVC / RBF-SVC / PCA+RBF-SVC
```

If the probe beats the SNN classifier head, report the result as a representation
plus classical nonlinear head, not as pure end-to-end SNN superiority.

## Research Extensions After V1

These are optional extensions after the current SNN branch:

- SpikingJelly replication of the best `snnTorch` result.
- ANN-to-SNN conversion from the trained ShallowConvNet baseline.
- Teacher distillation into `shallow_temporal_lif_head` from Shallow/QNN/EEGPT
  logits if the goal is to push beyond the current 82-83% regime.
- Energy proxy metrics: spike rate, synaptic operations proxy, latency to
  correct classification.
- Prefix-specific retraining at `K=64` or `K=80` only if the project wants to
  revisit latency after the negative post-hoc early-exit curve.
- Adaptive signed-delta encoding with trainable per-channel gains and
  thresholds before opening larger raw-SNN architectures.
- Wavelet or filter-bank front-end before SNN if raw temporal encoding is weak.

Do not prioritize more end-to-end raw SNN runs unless there is a new interface
or training hypothesis. The evidence says raw SNN is weak and Shallow-temporal
features are the useful interface.

## V5/V6 Five-Fold Results After V4

The V4 Shallow-teacher prefix distillation/rescue and adaptive signed-delta
encoder have now been treated as closed diagnostics. The next SNN work tested
three targeted hypotheses without reopening raw encoders, post-hoc early-exit,
Shallow-teacher prefix rescue, adaptive signed-delta, or quantum-kernel probes.

| Branch | Hypothesis | Result | Decision |
|---|---|---:|---|
| `V5a temporal_lif_embedding_probe` | The best temporal LIF head may be a better representation than its supervised classifier shows. | 5-fold `86.02 +/- 0.93%` acc / `85.39 +/- 0.82%` macro-F1 | best SNN representation row |
| `V5b eegpt_teacher_distill_full104` | A stronger EEGPT adapter58 RBF-SVC teacher may transfer class-boundary geometry into the fast SNN head. | fold-1 student `82.57%` acc / `81.65%` macro-F1; teacher `89.35%` acc | below gate; neutral/close current recipe |
| `V6a shallow_temporal_gated_lif` | A VEP-aware temporal gate may outperform uniform mean membrane readout over 104 bins. | 5-fold `83.50 +/- 0.89%` acc / `82.83 +/- 0.84%` macro-F1 | positive; superseded by V6b |
| `V6b shallow_temporal_spike_aware_gated_lif` | Gated membrane readout may need explicit spike-count evidence inside the supervised head. | 5-fold `84.41 +/- 1.29%` acc / `83.73 +/- 1.13%` macro-F1 | positive; superseded by V6d |
| `V6d shallow_temporal_spike_aware_proto_lif_ce` | The supervised SNN head may need a differentiable RBF/prototype decision surface. | 5-fold `84.75 +/- 0.66%` acc / `83.94 +/- 0.51%` macro-F1 | best supervised SNN row |

Five-fold baseline:

```text
shallow_temporal_lif_head = 82.56 +/- 1.57% acc / 81.82 +/- 1.44% macro-F1
```

This keeps the research question focused: can SNN move beyond being a fast
QNN-adjacent head by improving representation probing, using a stronger EEGPT
teacher, or learning when to read VEP temporal dynamics?

Current answer from five folds:

```text
representation probing and temporal gating are positive;
EEGPT logit-margin distillation is not yet positive for the current student.
```

## V6/V5a Extension Screen

The follow-up screen tested whether the two positive SNN routes could be pushed
further: stronger supervised temporal readouts after V6a and cheap V5a
representation/probe fusions. The screen used fold 1 only and promoted to
five folds only if the predefined gate was met.

| Branch | Variant | Fold-1 result | Decision |
|---|---|---:|---|
| `V6b` | spike-aware gated LIF, fusing gated membrane with spike count | fold-1 `83.68%` acc / `82.99%` macro-F1; 5-fold `84.41 +/- 1.29%` acc / `83.73 +/- 1.13%` macro-F1 | promoted at this step; superseded by V6d |
| `V6c` | windowed spike LIF, 4 temporal bins | `82.02%` acc / `81.13%` macro-F1 | below gate |
| `V6e` | multiscale gated LIF, kernels 3/9/17 | `82.99%` acc / `82.18%` macro-F1 | below gate |
| `V5a-ext` | windowed spike-count probe, 4 bins | `83.54%` acc / `82.72%` macro-F1 | below gate |
| `V5a-ext` | spike-count and mean-membrane late fusion | `85.34%` acc / `84.60%` macro-F1 | below gate |

Current answer from the extension screen:

```text
V6b missed the original fold-1 gate slightly, but the user-requested 5-fold run
beats V6a and became the supervised SNN row before V6d superseded it.
Naive V5a windowed/fusion probes do not beat the existing V5a spike_count RBF
probe.
```

## V6d Differentiable Prototype Head

V6d tests the hypothesis that V6b was limited by a linear neural classifier
rather than by the LIF/gated spike representation. The successful variant keeps
the supervised PyTorch model end-to-end and replaces the final linear classifier
with a differentiable RBF/prototype head.

| Variant | Five-fold result | Decision |
|---|---:|---|
| `V6d0 CE-only` | `84.75 +/- 0.66%` acc / `83.94 +/- 0.51%` macro-F1 | best supervised SNN row |
| `V6d1 SupCon 0.05` | `84.08 +/- 1.21%` acc / `83.31 +/- 1.08%` macro-F1 | below V6d0 |
| `V6d2 SupCon 0.10` | `84.22 +/- 0.76%` acc / `83.55 +/- 0.64%` macro-F1 | below V6d0 |
| `V6d3 SupCon 0.05 + proto-sep` | `84.22 +/- 0.84%` acc / `83.49 +/- 0.72%` macro-F1 | below V6d0 |

Current answer:

```text
Differentiable prototype/RBF-style decision geometry helps the supervised SNN
row. SupCon and prototype-separation regularization did not help in this
low-data VEP setting.
```

## Reference Links

- Biomedical SNN survey: https://pmc.ncbi.nlm.nih.gov/articles/PMC11362400/
- snnTorch surrogate gradients: https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html
- SpikingJelly framework: https://pmc.ncbi.nlm.nih.gov/articles/PMC10558124/
- CSNN EEG anticipatory potentials: https://arxiv.org/abs/2208.06900
- SCNet EEG motor imagery: https://www.sciencedirect.com/science/article/pii/S0925231223005933
- SNN EEG wavelet/spiking attention: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1652274/full
- Short-length SSVEP knowledge distillation: https://www.sciencedirect.com/science/article/pii/S0893608025000127
- SNN knowledge distillation: https://ojs.aaai.org/index.php/AAAI/article/view/37175
