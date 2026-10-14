# Open SNN Research Directions for the VEP EEG Benchmark

Last updated: 2026-06-08

Tool note: the user requested the `life-science-research` plugin, but that
plugin was not available in the current session and did not appear in the list
of installable plugins. This document uses web search/paper sources, local
documents in `SNN/`, `SNN/README.md`, `SNN/LITERATURE_REVIEW.md`, and the
existing result summaries/JSON.

The goal of this file is to avoid repeating SNN directions that have already
been explored. The directions below are grouped into:

```text
1. Done / closed
2. V5/V6 directions already run 5-fold
3. Directions still genuinely open after fold 1
4. Proposed experiment order
```

## 1. Current SNN Status

Main conclusion of the SNN branch:

```text
Raw EEG SNN is weak.
Useful SNN path = frozen ShallowConvNet temporal feature map + spiking temporal readout.
```

Current best SNN representation:

```text
frozen ShallowConvNet temporal feature map [B,40,104]
-> LIF temporal readout
-> spike_count embedding
-> RBF-SVC C=10 gamma=scale

accuracy       86.02 +/- 0.93%
macro-F1       85.39 +/- 0.82%
balanced acc   85.29 +/- 0.87%
```

Current best supervised SNN:

```text
frozen ShallowConvNet temporal feature map [B,40,104]
-> spike-aware gated LIF temporal readout
-> gated membrane + spike_count fusion
-> differentiable RBF/prototype head

accuracy       84.75 +/- 0.66%
macro-F1       83.94 +/- 0.51%
balanced acc   83.85 +/- 0.41%
```

Compared with QNN:

```text
Best QNN snapshot:
  t2_temporal_qdelta + label smoothing
  accuracy = 82.614% +/- 0.994%
  train time = about 84.4 min/fold

The SNN representation row already beats the QNN snapshot in accuracy. The
supervised gated SNN also beats plain temporal LIF and sits above the QNN
snapshot, but its claim must be kept separate from the representation-probe row.
```

Therefore research should focus on:

```text
spiking temporal readout / distillation / embedding probe / efficiency,
not generic raw SNN.
```

V5/V6 5-fold update on 2026-06-04:

| Branch | Result | Gate / Baseline | Decision |
|---|---:|---|---|
| `V5a temporal_lif_embedding_probe` | `86.02 +/- 0.93%` acc / `85.39 +/- 0.82%` macro-F1 (`spike_count + RBF-SVC C=10`) | best QNN snapshot `82.61 +/- 0.99%` acc | promote as best SNN representation row |
| `V5b eegpt_teacher_distill_full104` | student `82.57%` acc / `81.65%` macro-F1; teacher `89.35%` acc | `>=84.0%` acc or `>=83.0%` macro-F1 | neutral/close current recipe |
| `V6a shallow_temporal_gated_lif` | `83.50 +/- 0.89%` acc / `82.83 +/- 0.84%` macro-F1 | plain temporal LIF `82.56 +/- 1.57%` acc | positive; superseded by V6b |
| `V6b shallow_temporal_spike_aware_gated_lif` | `84.41 +/- 1.29%` acc / `83.73 +/- 1.13%` macro-F1 | V6a `83.50 +/- 0.89%` acc | positive; superseded by V6d |
| `V6d0 shallow_temporal_spike_aware_proto_lif_ce` | `84.75 +/- 0.66%` acc / `83.94 +/- 0.51%` macro-F1 | V6b `84.41 +/- 1.29%` acc | promote as best supervised SNN row |

V6/V5a extension fold-1 screen on 2026-06-07:

| Branch | Result | Gate | Decision |
|---|---:|---|---|
| `V6b shallow_temporal_spike_aware_gated_lif` | fold-1 `83.68%` acc / `82.99%` macro-F1; 5-fold `84.41 +/- 1.29%` acc / `83.73 +/- 1.13%` macro-F1 | `>=84.0%` acc or `>=83.3%` macro-F1 | fold-1 close; 5-fold user-requested run promoted |
| `V6c shallow_temporal_windowed_spike_lif_4bins` | `82.02%` acc / `81.13%` macro-F1 | same V6 gate | not promoted |
| `V6e shallow_temporal_multiscale_gated_lif` | `82.99%` acc / `82.18%` macro-F1 | same V6 gate | not promoted |
| `V5a-ext windowed_spike_count_4 + RBF-SVC` | `83.54%` acc / `82.72%` macro-F1 | `>86.5%` acc or `>85.8%` macro-F1 | not promoted |
| `V5a-ext spike_mean_late_fusion` | `85.34%` acc / `84.60%` macro-F1 | same V5a gate | close, not promoted |

Conclusion: V6b was run 5-fold on request after the screen and beat V6a, so it
replaces V6a as the supervised SNN row. Keep V5a `spike_count + RBF-SVC` as the
strongest SNN representation row. V6c/V6e and V5a windowed/fusion probes are
not promoted.

V6d differentiable prototype-head update on 2026-06-07:

| Branch | Result | Gate / Baseline | Decision |
|---|---:|---|---|
| `V6d0 proto_lif_ce` | `84.75 +/- 0.66%` acc / `83.94 +/- 0.51%` macro-F1 | V6b `84.41 +/- 1.29%` acc / `83.73 +/- 1.13%` macro-F1 | promote as best supervised SNN row |
| `V6d1 proto_lif_supcon005` | `84.08 +/- 1.21%` acc / `83.31 +/- 1.08%` macro-F1 | V6b baseline | below V6b/V6d0 |
| `V6d2 proto_lif_supcon010` | `84.22 +/- 0.76%` acc / `83.55 +/- 0.64%` macro-F1 | V6b baseline | below V6b/V6d0 |
| `V6d3 proto_lif_supcon005_sep` | `84.22 +/- 0.84%` acc / `83.49 +/- 0.72%` macro-F1 | V6b baseline | below V6b/V6d0 |

Conclusion: the differentiable prototype/RBF-style head lets the supervised SNN
beat V6b, but SupCon/prototype separation does not help in this setting.

EEGPT-token SNN plug-in screen on 2026-06-08:

| Branch | Fold-1 result | Gate | Decision |
|---|---:|---|---|
| `eegpt_token_spike_aware_gated_lif` | `60.86%` acc / `59.73%` macro-F1 | `>=84.0%` acc or `>=83.5%` macro-F1 | not run 5-fold |
| `eegpt_token_spike_aware_proto_lif_ce` | `59.61%` acc / `58.64%` macro-F1 | same | not run 5-fold |

Conclusion: the V6-style SNN head is not plug-and-play on cached EEGPT tokens
`[B,4,2048]`. The current strong SNN direction still needs a frozen feature map
that preserves many temporal steps, such as the Shallow temporal map `[B,40,104]`.

## 2. Done / Closed

The following directions already have results in the README, literature review,
or result files. They should not be listed as active directions again without a
new hypothesis.

| Direction | Status | Conclusion |
|---|---|---|
| Raw supervised SNN `direct/rate/delta/signed_delta` | Done fold 1 | Weak; best fixed signed/delta supervised only about 46-47% |
| Raw SNN embedding probe | Done fold 1 | Weak-positive; best `delta SNN embedding + RBF-SVC` = `68.46% / 67.83%`, not promoted |
| Raw delta-family classical controls | Done 5-fold | Delta transform alone is weak; signed-delta flat control about 50.11% |
| Shallow80 LIF/PLIF hybrid | Done 5-fold | Viable at about 81.5%, but below temporal-map LIF |
| Shallow temporal-map LIF | Done 5-fold | Previous V2 best / plain baseline: `82.56% +/- 1.57%` |
| Post-hoc early-exit | Done 5-fold | Negative; K80 only `79.13%`, below gate |
| Shallow-teacher prefix distillation | Done fold 1 | Slight improvement but did not pass gate |
| Prefix rescue K64/K80 | Done selected 5-fold | K64 = `76.97%`, K80 = `79.01%`; both fail gates |
| Adaptive signed-delta raw encoder | Done fold 1 | `46.20% / 46.54%`, did not reach gate; closed |
| Quantum kernel on SNN embedding | Done fold 1 subset | Quantum kernel loses to matched RBF-SVC control |
| V6/V5a extension screen | Done fold 1 + V6b 5-fold | V6b beats V6a on 5-fold; windowed/fusion probes do not beat V5a |
| V6d differentiable prototype head | Done 5-fold | CE-only beats V6b; SupCon variants do not improve |
| EEGPT-token SNN plug-in head | Done fold 1 | Best only `60.86% / 59.73%`; did not pass gate, no 5-fold |

In particular:

```text
Do not repeat Shallow-teacher prefix distillation by merely tweaking
alpha/dropout. The rescue sweep showed K64/K80 do not pass the gate.
```

```text
Do not open a larger raw SNN encoder if it relies only on the current
fixed/adaptive delta coding. Evidence says the raw SNN path is weak.
```

## 3. Open Direction 1: Temporal-hybrid SNN embedding probe

### Hypothesis

Foundation-model branches show that a frozen/fixed representation + classical
RBF-SVC is often stronger than a neural head on small datasets. The raw SNN
embedding probe has been done and is weak, but the best SNN hybrid
`shallow_temporal_lif_head` has not been fully evaluated as a representation
learner over 5 folds.

### Pipeline

```text
train shallow_temporal_lif_head on fold k
-> reload best checkpoint
-> extract embeddings:
     spike_count
     mean_membrane
     final_membrane
     max_membrane
     concat
-> train-only StandardScaler
-> heads:
     LinearSVC C=1
     RBF-SVC C=10 gamma=scale
     PCA128 + RBF-SVC C=3 gamma=scale
-> evaluate fixed test fold
```

### Why this is not already done

The README already has:

```text
shallow80_lif_head embedding probe fold 1
shallow80_plif_head embedding probe fold 1
raw SNN embedding probes fold 1
```

But there is no full probe matrix yet for:

```text
shallow_temporal_lif_head embeddings
```

The quantum kernel probe used `snn_temporal_lif_concat` on a fold-1 subset, and
the matched RBF-SVC reached about `83.26%`, so a full probe is worth checking.

### Controls

- Supervised `shallow_temporal_lif_head`.
- Shallow80 LIF/PLIF embedding probe.
- Same train-only embedding scaler.
- Same fold split and checkpoint selection.

### Promotion gate

```text
fold-1 >= 83.0% acc or >=82.0% macro-F1 -> run 5-fold
5-fold > 82.61% acc -> serious SNN-vs-QNN result
5-fold near 82.5% but faster/sparser -> efficiency result
```

Five-fold result:

```text
best = spike_count embedding + RBF-SVC C=10 gamma=scale
accuracy = 86.02 +/- 0.93%
macro-F1 = 85.39 +/- 0.82%
balanced acc = 85.29 +/- 0.87%
decision = best SNN representation row
```

### Paper value

If positive:

```text
SNN as representation learner + classical nonlinear probe is stronger than a
pure supervised SNN head.
```

If not:

```text
Supervised temporal LIF readout is already the useful SNN endpoint.
```

## 4. Open Direction 2: EEGPT-teacher distillation into SNN

### Hypothesis

Shallow-teacher prefix distillation failed its gates. A possible reason is that
the teacher sits only around 82%. EEGPT adapter58 + RBF-SVC reaches:

```text
87.901% +/- 1.824% accuracy
87.450% +/- 2.214% macro-F1
```

The remaining open direction is therefore to use a stronger teacher to distill
class geometry into a small, fast SNN student.

### Pipeline

```text
Teacher:
  EEGPT adapter58 frozen embeddings
  -> RBF-SVC decision_function margins
  -> optional calibration / temperature scaling

Student:
  frozen Shallow temporal map [B,40,104]
  -> LIF temporal readout
  -> classifier

Loss:
  (1 - alpha) * CE(y, student_logits)
  + alpha * T^2 * KL(student_logits/T, teacher_margins/T)
```

### Variants

```text
V5a full-window EEGPT distillation:
  prefix_steps = 104
  goal = push SNN above current QNN/SNN regime

V5b prefix EEGPT distillation:
  prefix_steps = 64, 80
  goal = revisit early decision only with stronger teacher

V5c feature-level alignment:
  project SNN embedding -> EEGPT PCA/embedding space
  add MSE or contrastive loss on train fold only
```

### Controls

- Current `shallow_temporal_lif_head`.
- Shallow-teacher prefix distillation/rescue rows.
- Non-spiking MLP/TCN student on same Shallow temporal map.
- EEGPT teacher is an upper teacher anchor, not a speed-matched baseline.

### Promotion gate

```text
Full-window:
  fold-1 >= 84% acc or >=83% macro-F1 -> run 5-fold
  5-fold > QNN best 82.61% with low training time -> promote

Prefix:
  K64 within 2 pp of full-window SNN
  or K80 >= 80% 5-fold
```

Fold-1 result for the implemented full-window recipe:

```text
teacher = EEGPT adapter58 RBF-SVC C=10
teacher accuracy = 89.35%
student accuracy = 82.57%
student macro-F1 = 81.65%
best epoch = 74
decision = below gate; close or redesign distillation before 5-fold
```

### Paper value

```text
Distilling a strong EEG foundation-model decision boundary into a compact
spiking temporal readout may combine EEGPT representation quality with SNN
training/inference efficiency.
```

This is the direction with the highest paper value if it works.

## 5. Open Direction 3: VEP-aware temporal gated LIF readout

### Hypothesis

The current `ShallowFeatureSNNHead` is very simple:

```text
for each timestep:
  Linear(40 -> hidden)
  LIF
readout = mean_membrane / final / max / spike_count
```

It does not learn explicit timing importance. VEP is a time-locked response, so
a lightweight temporal gate/attention could learn more discriminative bins than
mean pooling.

### Architecture

```text
Shallow temporal map [B,40,104]
-> Linear 40 -> H
-> LIF over 104 steps
-> collect membrane/spike sequence [B,H,104]
-> temporal gate:
     depthwise Conv1d(H -> H, kernel=5 or 9)
     sigmoid gate per timestep
-> weighted mean membrane/spike readout
-> Linear H -> 4
```

Config names:

```text
shallow_temporal_gated_lif
shallow_temporal_lif_temporal_attention
```

### Controls

- Current `mean_membrane` LIF readout.
- Same architecture without LIF.
- Same hidden_dim/dropout/training recipe.

### Promotion gate

```text
fold-1 beats current temporal LIF by >=0.5 pp acc or macro-F1 -> run 5-fold
5-fold >=83% acc or lower variance than current SNN -> promote
```

Five-fold result:

```text
baseline shallow_temporal_lif_head = 82.56 +/- 1.57% acc / 81.82 +/- 1.44% macro-F1
shallow_temporal_gated_lif = 83.50 +/- 0.89% acc / 82.83 +/- 0.84% macro-F1
delta = +0.94 pp acc / +1.02 pp macro-F1
decision = positive V6a result; superseded by V6b spike-aware gated LIF
```

## 6. Open Direction 4: Multi-scale / wavelet-inspired spiking temporal readout

### Hypothesis

The VEP signal has both waveform morphology and time-frequency structure. The
current LIF readout sees each timestep through a single Linear layer.
Multi-scale temporal branches could capture slow/fast components better.

### Architecture

```text
Shallow temporal map [B,40,104]
-> parallel temporal branches:
     depthwise Conv1d kernel 3
     depthwise Conv1d kernel 7
     depthwise Conv1d kernel 15
     optional Haar DWT low/high components
-> concatenate/project to H
-> LIF readout
-> classifier
```

Config names:

```text
shallow_temporal_multiscale_lif
shallow_temporal_wavelet_lif
```

### Controls

- Non-spiking multi-scale temporal head.
- Current temporal LIF with same hidden dim.
- RBF probe on extracted multi-scale embeddings.

### Promotion gate

```text
fold-1 >=83% acc or > current SNN macro-F1 by >=0.5 pp -> run 5-fold
```

## 7. Open Direction 5: Topology-preserving SNN before Shallow spatial collapse

### Hypothesis

The current best SNN uses the Shallow feature map after spatial convolution:

```text
Shallow conv1/conv2/BN/square -> [B,40,104]
```

Channel topology has already been collapsed. A new SNN could read an earlier map:

```text
after Shallow temporal conv only -> [B,40,14,104]
```

Then use a channel graph/gate to exploit Emotiv14 geometry.

### Architecture

```text
raw EEG [B,14,128]
-> frozen Shallow temporal conv1 -> [B,40,14,104]
-> channel graph message passing or gated channel pooling over 14 electrodes
-> LIF over time
-> global filter/channel readout
-> classifier
```

Simpler first version:

```text
conv1 map [B,40,14,104]
-> grouped 1x1 channel projection
-> LIF over time
-> attention/gated pooling over 14 channels
```

### Controls

- Current Shallow spatial conv + temporal LIF.
- Non-spiking graph/attention head on same `conv1` map.
- Raw graph SNN only if frozen-conv1 route shows signal.

### Promotion gate

```text
fold-1 >=83% acc -> run 5-fold
or fold-1 improves Apple/Car recalls meaningfully without hurting macro-F1
```

## 8. Open Direction 6: Shallow-initialized / ANN-to-SNN readout

### Hypothesis

Current temporal LIF head is trained from scratch on frozen Shallow maps. Since
ShallowConvNet already has a trained classifier and a pooling/log feature path,
it can be used to initialize or distill the SNN readout.

### Variants

```text
Variant A: Shallow pooling mimic
  [B,40,104]
  -> LIF integrate
  -> readout approximates Shallow AvgPool/log features [B,80]
  -> reuse or initialize from Shallow 80->4 classifier

Variant B: two-stage training
  stage 1: match Shallow 80D features/logits
  stage 2: supervised CE fine-tune

Variant C: ANN head control
  same architecture without spike activation
```

### Promotion gate

```text
fold-1 matches current SNN with fewer epochs or lower spike rate -> efficiency result
fold-1 beats current SNN by >=0.5 pp -> run 5-fold
```

## 9. Open Direction 7: Efficiency / latency diagnostic

### Hypothesis

SNN may not beat EEGPT accuracy, but it may be the best efficiency baseline
within the Shallow/QNN/SNN group. Training time, spike rate, and some head
latency already exist, but they need to be standardized for a rigorous claim.

### Metrics

```text
Accuracy / macro-F1 / balanced acc
Train time per fold
Head inference ms/sample
Spike rate per layer/head
Synaptic operations proxy:
  spike_count * fanout
Dense operation proxy:
  MACs for non-spiking control
Prefix latency curve:
  bins 32, 48, 64, 80, 104
```

### Controls

- ShallowConvNet full head.
- QNN best row.
- SNN supervised temporal head.
- SNN embedding + RBF, if promoted.
- Non-spiking MLP/TCN readout on same temporal map.

### Paper value

```text
SNN is not the strongest foundation-model result, but offers a compact
neuromorphic temporal readout with near-QNN accuracy and much lower downstream
training cost.
```

## 10. Recommended active experiment order after V6d prototype screen

Proposed practical order:

```text
1. Promote V5a temporal-hybrid SNN embedding probe in the benchmark table
   Five-fold reached 86.02% / 85.39%.

2. Promote V6d0 spike-aware prototype LIF as the supervised SNN row
   Five-fold improved V6b by about +0.34 pp accuracy and plain temporal LIF by
   about +2.19 pp accuracy.

3. Close V5b current EEGPT-teacher logit-margin distillation recipe
   Teacher was strong, but student stayed below the promotion gate.

4. Close current SupCon/prototype-separation variants unless a new hypothesis is introduced
   V6d0 CE-only was better than the regularized variants.

5. Topology-preserving conv1-map SNN
   More ambitious; preserves 14-channel geometry before spatial collapse.

6. Shallow-initialized / ANN-to-SNN readout
   Useful if goal is speed/efficiency and training stability.

7. Unified energy/latency diagnostic
   Run once 2-3 serious SNN candidates are available.
```

Do not include these as active priorities:

```text
raw fixed encoders
raw SNN embedding probes
Shallow-teacher prefix distillation/rescue
adaptive_signed_delta raw encoder
post-hoc early-exit
quantum kernel on SNN embedding
V6/V5a extension variants by small hyperparameter tweaks only
V6d SupCon/prototype-separation variants by small lambda tweaks only
```

## 11. Commands For V5/V6 Reproduction

V5a temporal-hybrid embedding probe:

```bash
.venv/bin/python SNN/scripts/run_shallow_snn_probe.py \
  --config shallow_temporal_lif_head \
  --folds 1 2 3 4 5 \
  --modes spike_count mean_membrane final_membrane max_membrane concat \
  --device auto
```

V5b EEGPT-teacher distillation:

```bash
.venv/bin/python SNN/scripts/run_snn_eegpt_distill.py \
  --config shallow_temporal_lif_eegpt_distill_full104 \
  --folds 1 \
  --device auto \
  --no-progress
```

V6a gated temporal LIF:

```bash
.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_gated_lif \
  --folds 1 2 3 4 5 \
  --device auto \
  --no-progress
```

V6/V5a extension fold-1 screen:

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

V6d differentiable prototype head:

```bash
.venv/bin/python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_spike_aware_proto_lif_ce shallow_temporal_spike_aware_proto_lif_supcon005 shallow_temporal_spike_aware_proto_lif_supcon010 shallow_temporal_spike_aware_proto_lif_supcon005_sep \
  --folds 1 2 3 4 5 \
  --device auto \
  --no-progress
```

## 12. Concrete new branch names

Suggested naming:

```text
SNN V5a: temporal_lif_embedding_probe
SNN V5b: eegpt_teacher_distill_full104
SNN V5c: eegpt_teacher_distill_k80
SNN V6a: shallow_temporal_gated_lif
SNN V6b: shallow_temporal_spike_aware_gated_lif
SNN V6c: shallow_temporal_windowed_spike_lif
SNN V6d: shallow_temporal_spike_aware_proto_lif
SNN V6e: shallow_temporal_multiscale_gated_lif
SNN V7a: shallow_conv1_graph_lif
SNN V8a: shallow_initialized_lif
SNN V9: snn_efficiency_benchmark
```

## 13. Paper claim discipline

Use:

- SNN is a promising efficient temporal-readout branch because it matches QNN
  accuracy regime with far lower downstream training cost.
- Current evidence supports SNN over learned Shallow temporal features, not raw
  EEG spike encoding.
- Distillation from a strong EEGPT teacher is the most plausible route to make
  SNN more than a QNN-speed alternative.
- Raw SNN encoders, including adaptive signed-delta, are currently negative or
  weak-positive diagnostics.
- Energy/latency claims need explicit profiling, not just training time.
- V5a is a representation-probe result, while V6d is the cleaner supervised
  SNN-hybrid architecture result.
- Current V5b logit-margin distillation is negative despite a strong EEGPT
  teacher; do not cite it as a successful distillation result.

Avoid:

- Do not claim SNN beats QNN unless full 5-fold result beats QNN under same
  fixed folds.
- Do not claim early VEP decision unless prefix gates are met.
- Do not claim raw spike encoding works unless adaptive/raw encoders beat fixed
  controls and clear fold-1 promotion gate.
- Do not claim neuromorphic energy advantage without spike-rate/SOP/latency
  metrics.

## 14. Sources used

- SCNet adaptive coding for EEG SNN:
  https://www.sciencedirect.com/science/article/pii/S0925231223005933
- RMKD short-length SSVEP knowledge distillation:
  https://www.sciencedirect.com/science/article/pii/S0893608025000127
- SpikeWavformer / spiking wavelet self-attention for EEG:
  https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1652274/full
- CSNN for EEG anticipatory brain potentials:
  https://www.nature.com/articles/s41598-024-59469-7
- EEGSN graph SNN for EEG, withdrawn arXiv record; use only cautiously:
  https://arxiv.org/abs/2304.07655
- snnTorch surrogate gradient docs:
  https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html
