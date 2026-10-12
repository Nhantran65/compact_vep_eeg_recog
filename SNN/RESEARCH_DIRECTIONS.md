# Hướng nghiên cứu SNN còn mở cho VEP EEG benchmark

Last updated: 2026-06-08

Ghi chú tool: người dùng yêu cầu plugin `life-science-research`, nhưng plugin
này không khả dụng trong session hiện tại và cũng không xuất hiện trong danh
sách plugin có thể cài. Tài liệu này dùng web search/paper sources, tài liệu
local trong `SNN/`, `SNN/README.md`, `SNN/LITERATURE_REVIEW.md`, và các result
summary/JSON hiện có.

Mục tiêu của file này là tránh lặp lại các hướng SNN đã làm. Các hướng bên dưới
được chia thành:

```text
1. Đã làm / đã đóng
2. Hướng V5/V6 đã chạy 5-fold
3. Hướng còn mở thật sự sau fold 1
4. Thứ tự thử nghiệm đề xuất
```

## 1. Trạng thái SNN hiện tại

Kết luận chính của nhánh SNN:

```text
Raw EEG SNN yếu.
Useful SNN path = frozen ShallowConvNet temporal feature map + spiking temporal readout.
```

Best SNN representation hiện tại:

```text
frozen ShallowConvNet temporal feature map [B,40,104]
-> LIF temporal readout
-> spike_count embedding
-> RBF-SVC C=10 gamma=scale

accuracy       86.02 +/- 0.93%
macro-F1       85.39 +/- 0.82%
balanced acc   85.29 +/- 0.87%
```

Best supervised SNN hiện tại:

```text
frozen ShallowConvNet temporal feature map [B,40,104]
-> spike-aware gated LIF temporal readout
-> gated membrane + spike_count fusion
-> differentiable RBF/prototype head

accuracy       84.75 +/- 0.66%
macro-F1       83.94 +/- 0.51%
balanced acc   83.85 +/- 0.41%
```

So với QNN:

```text
Best QNN snapshot:
  t2_temporal_qdelta + label smoothing
  accuracy = 82.614% +/- 0.994%
  train time = about 84.4 min/fold

SNN representation row đã vượt QNN snapshot về accuracy. Supervised gated SNN
cũng vượt plain temporal LIF và nằm trên QNN snapshot, nhưng cần claim tách
riêng khỏi representation-probe row.
```

Vì vậy hướng nghiên cứu nên tập trung vào:

```text
spiking temporal readout / distillation / embedding probe / efficiency,
không phải raw SNN generic.
```

V5/V6 5-fold update ngày 2026-06-04:

| Branch | Result | Gate / Baseline | Decision |
|---|---:|---|---|
| `V5a temporal_lif_embedding_probe` | `86.02 +/- 0.93%` acc / `85.39 +/- 0.82%` macro-F1 (`spike_count + RBF-SVC C=10`) | best QNN snapshot `82.61 +/- 0.99%` acc | promote as best SNN representation row |
| `V5b eegpt_teacher_distill_full104` | student `82.57%` acc / `81.65%` macro-F1; teacher `89.35%` acc | `>=84.0%` acc hoặc `>=83.0%` macro-F1 | neutral/close current recipe |
| `V6a shallow_temporal_gated_lif` | `83.50 +/- 0.89%` acc / `82.83 +/- 0.84%` macro-F1 | plain temporal LIF `82.56 +/- 1.57%` acc | positive; superseded by V6b |
| `V6b shallow_temporal_spike_aware_gated_lif` | `84.41 +/- 1.29%` acc / `83.73 +/- 1.13%` macro-F1 | V6a `83.50 +/- 0.89%` acc | positive; superseded by V6d |
| `V6d0 shallow_temporal_spike_aware_proto_lif_ce` | `84.75 +/- 0.66%` acc / `83.94 +/- 0.51%` macro-F1 | V6b `84.41 +/- 1.29%` acc | promote as best supervised SNN row |

V6/V5a extension fold-1 screen ngày 2026-06-07:

| Branch | Result | Gate | Decision |
|---|---:|---|---|
| `V6b shallow_temporal_spike_aware_gated_lif` | fold-1 `83.68%` acc / `82.99%` macro-F1; 5-fold `84.41 +/- 1.29%` acc / `83.73 +/- 1.13%` macro-F1 | `>=84.0%` acc hoặc `>=83.3%` macro-F1 | fold-1 close; 5-fold user-requested run promoted |
| `V6c shallow_temporal_windowed_spike_lif_4bins` | `82.02%` acc / `81.13%` macro-F1 | same V6 gate | không promote |
| `V6e shallow_temporal_multiscale_gated_lif` | `82.99%` acc / `82.18%` macro-F1 | same V6 gate | không promote |
| `V5a-ext windowed_spike_count_4 + RBF-SVC` | `83.54%` acc / `82.72%` macro-F1 | `>86.5%` acc hoặc `>85.8%` macro-F1 | không promote |
| `V5a-ext spike_mean_late_fusion` | `85.34%` acc / `84.60%` macro-F1 | same V5a gate | close, không promote |

Kết luận: V6b được chạy 5-fold theo yêu cầu sau screen và vượt V6a, nên thay
V6a làm supervised SNN row. Giữ V5a `spike_count + RBF-SVC` làm strongest SNN
representation row. V6c/V6e và V5a windowed/fusion probes không promote.

V6d differentiable prototype-head update ngày 2026-06-07:

| Branch | Result | Gate / Baseline | Decision |
|---|---:|---|---|
| `V6d0 proto_lif_ce` | `84.75 +/- 0.66%` acc / `83.94 +/- 0.51%` macro-F1 | V6b `84.41 +/- 1.29%` acc / `83.73 +/- 1.13%` macro-F1 | promote as best supervised SNN row |
| `V6d1 proto_lif_supcon005` | `84.08 +/- 1.21%` acc / `83.31 +/- 1.08%` macro-F1 | V6b baseline | below V6b/V6d0 |
| `V6d2 proto_lif_supcon010` | `84.22 +/- 0.76%` acc / `83.55 +/- 0.64%` macro-F1 | V6b baseline | below V6b/V6d0 |
| `V6d3 proto_lif_supcon005_sep` | `84.22 +/- 0.84%` acc / `83.49 +/- 0.72%` macro-F1 | V6b baseline | below V6b/V6d0 |

Kết luận: differentiable prototype/RBF-style head giúp supervised SNN vượt
V6b, nhưng SupCon/prototype separation chưa giúp trong setting này.

EEGPT-token SNN plug-in screen ngày 2026-06-08:

| Branch | Fold-1 result | Gate | Decision |
|---|---:|---|---|
| `eegpt_token_spike_aware_gated_lif` | `60.86%` acc / `59.73%` macro-F1 | `>=84.0%` acc hoặc `>=83.5%` macro-F1 | không chạy 5-fold |
| `eegpt_token_spike_aware_proto_lif_ce` | `59.61%` acc / `58.64%` macro-F1 | same | không chạy 5-fold |

Kết luận: V6-style SNN head không plug-and-play tốt trên cached EEGPT tokens
`[B,4,2048]`. Hướng SNN mạnh hiện tại vẫn cần frozen feature map giữ nhiều bước
temporal dynamics, như Shallow temporal map `[B,40,104]`.

## 2. Đã làm / đã đóng

Các hướng sau đã có kết quả trong README, literature review hoặc result files.
Không nên đưa vào danh sách active directions nữa nếu không có hypothesis mới.

| Hướng | Trạng thái | Kết luận |
|---|---|---|
| Raw supervised SNN `direct/rate/delta/signed_delta` | Done fold 1 | Yếu; best fixed signed/delta supervised chỉ khoảng 46-47% |
| Raw SNN embedding probe | Done fold 1 | Weak-positive; best `delta SNN embedding + RBF-SVC` = `68.46% / 67.83%`, không promote |
| Raw delta-family classical controls | Done 5-fold | Delta transform alone yếu; signed-delta flat control khoảng 50.11% |
| Shallow80 LIF/PLIF hybrid | Done 5-fold | Viable khoảng 81.5%, nhưng dưới temporal-map LIF |
| Shallow temporal-map LIF | Done 5-fold | Previous V2 best / plain baseline: `82.56% +/- 1.57%` |
| Post-hoc early-exit | Done 5-fold | Negative; K80 chỉ `79.13%`, chưa đủ gate |
| Shallow-teacher prefix distillation | Done fold 1 | Có nhích lên nhưng chưa qua gate |
| Prefix rescue K64/K80 | Done selected 5-fold | K64 = `76.97%`, K80 = `79.01%`; cả hai fail gates |
| Adaptive signed-delta raw encoder | Done fold 1 | `46.20% / 46.54%`, không đạt gate; đóng |
| Quantum kernel on SNN embedding | Done fold 1 subset | Quantum kernel thua matched RBF-SVC control |
| V6/V5a extension screen | Done fold 1 + V6b 5-fold | V6b vượt V6a trên 5-fold; windowed/fusion probes không vượt V5a |
| V6d differentiable prototype head | Done 5-fold | CE-only vượt V6b; SupCon variants không cải thiện |
| EEGPT-token SNN plug-in head | Done fold 1 | Best chỉ `60.86% / 59.73%`; không qua gate, không 5-fold |

Đặc biệt:

```text
Không nên lặp lại Shallow-teacher prefix distillation chỉ bằng cách đổi nhẹ
alpha/dropout. Rescue sweep đã cho thấy K64/K80 không qua gate.
```

```text
Không nên mở thêm raw SNN encoder lớn hơn nếu chỉ dựa trên fixed/adaptive
delta coding hiện tại. Evidence nói raw SNN path yếu.
```

## 3. Hướng mở 1: Temporal-hybrid SNN embedding probe

### Hypothesis

Foundation-model branches cho thấy frozen/fixed representation + classical
RBF-SVC thường mạnh hơn neural head trên dataset nhỏ. Raw SNN embedding probe đã
làm và yếu, nhưng SNN hybrid tốt nhất `shallow_temporal_lif_head` chưa được
đánh giá đầy đủ như một representation learner trên 5 folds.

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

README đã có:

```text
shallow80_lif_head embedding probe fold 1
shallow80_plif_head embedding probe fold 1
raw SNN embedding probes fold 1
```

Nhưng chưa có full probe matrix cho:

```text
shallow_temporal_lif_head embeddings
```

Quantum kernel probe đã dùng `snn_temporal_lif_concat` trên subset fold 1, và
matched RBF-SVC đạt khoảng `83.26%`, nên full probe đáng kiểm tra.

### Controls

- Supervised `shallow_temporal_lif_head`.
- Shallow80 LIF/PLIF embedding probe.
- Same train-only embedding scaler.
- Same fold split and checkpoint selection.

### Promotion gate

```text
fold-1 >= 83.0% acc hoặc >=82.0% macro-F1 -> run 5-fold
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

Nếu tốt:

```text
SNN as representation learner + classical nonlinear probe is stronger than a
pure supervised SNN head.
```

Nếu không:

```text
Supervised temporal LIF readout is already the useful SNN endpoint.
```

## 4. Hướng mở 2: EEGPT-teacher distillation into SNN

### Hypothesis

Shallow-teacher prefix distillation đã fail gates. Lý do có thể là teacher chỉ ở
vùng 82%. EEGPT adapter58 + RBF-SVC đạt:

```text
87.901% +/- 1.824% accuracy
87.450% +/- 2.214% macro-F1
```

Do đó hướng còn mở là dùng teacher mạnh hơn để distill class geometry vào SNN
student nhỏ và nhanh.

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
  fold-1 >= 84% acc hoặc >=83% macro-F1 -> run 5-fold
  5-fold > QNN best 82.61% with low training time -> promote

Prefix:
  K64 within 2 pp of full-window SNN
  hoặc K80 >= 80% 5-fold
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

Đây là hướng paper-value cao nhất nếu chạy được.

## 5. Hướng mở 3: VEP-aware temporal gated LIF readout

### Hypothesis

`ShallowFeatureSNNHead` hiện tại rất đơn giản:

```text
for each timestep:
  Linear(40 -> hidden)
  LIF
readout = mean_membrane / final / max / spike_count
```

Nó chưa học explicit timing importance. VEP là time-locked response, nên một
temporal gate/attention nhẹ có thể học đoạn bins discriminative hơn mean pooling.

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
fold-1 beats current temporal LIF by >=0.5 pp acc hoặc macro-F1 -> run 5-fold
5-fold >=83% acc hoặc lower variance than current SNN -> promote
```

Five-fold result:

```text
baseline shallow_temporal_lif_head = 82.56 +/- 1.57% acc / 81.82 +/- 1.44% macro-F1
shallow_temporal_gated_lif = 83.50 +/- 0.89% acc / 82.83 +/- 0.84% macro-F1
delta = +0.94 pp acc / +1.02 pp macro-F1
decision = positive V6a result; superseded by V6b spike-aware gated LIF
```

## 6. Hướng mở 4: Multi-scale / wavelet-inspired spiking temporal readout

### Hypothesis

VEP signal có cả waveform morphology và time-frequency structure. Current LIF
readout nhìn từng timestep qua một Linear layer. Multi-scale temporal branches
có thể bắt slow/fast components tốt hơn.

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
fold-1 >=83% acc hoặc > current SNN macro-F1 by >=0.5 pp -> run 5-fold
```

## 7. Hướng mở 5: Topology-preserving SNN before Shallow spatial collapse

### Hypothesis

Current best SNN dùng Shallow feature map sau spatial convolution:

```text
Shallow conv1/conv2/BN/square -> [B,40,104]
```

Channel topology đã bị collapse. Một SNN mới có thể đọc map sớm hơn:

```text
after Shallow temporal conv only -> [B,40,14,104]
```

Sau đó dùng channel graph/gate để exploit Emotiv14 geometry.

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

## 8. Hướng mở 6: Shallow-initialized / ANN-to-SNN readout

### Hypothesis

Current temporal LIF head is trained from scratch on frozen Shallow maps. Vì
ShallowConvNet đã có trained classifier và pooling/log feature path, có thể dùng
nó để initialize hoặc distill SNN readout.

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

## 9. Hướng mở 7: Efficiency / latency diagnostic

### Hypothesis

SNN có thể không vượt EEGPT accuracy, nhưng có thể là best efficiency baseline
trong nhóm Shallow/QNN/SNN. Hiện tại đã có training time, spike rate và một số
head latency, nhưng cần chuẩn hóa để claim nghiêm túc.

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

Đề xuất thứ tự thực dụng:

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
