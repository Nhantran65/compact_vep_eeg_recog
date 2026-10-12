# Paper Contributions - VEP EEG Benchmark

Last updated: 2026-06-08

Mục tiêu paper nên được frame là:

```text
Một benchmark có kiểm soát cho compact low-channel VEP EEG, dùng cùng fixed
folds để so sánh conventional EEG networks, QNN/QML, SNN và EEG foundation
models; trọng tâm là interface alignment và transfer diagnostics, không chỉ là
leaderboard accuracy.
```

## 1. Luận điểm chính

Luận điểm mạnh nhất:

```text
Trong compact 14-channel VEP EEG, hiệu quả transfer phụ thuộc nhiều vào việc
align đúng pretrained input contract của model hơn là model có dynamic-channel
interface hay kích thước lớn hơn.
```

Nói ngắn hơn:

```text
Interface alignment beats nominal channel flexibility.
```

Dataset/protocol chung:

```text
Task: 4-class visual/VEP EEG classification
Classes: Apple, Car, Flower, Human Face
Device: Emotiv EPOC-X, 14 channels, 128 Hz
Epoch: 1 second, [14,128]
Samples: 3612 epochs
Evaluation: fixed trainfold/fold_{1..5}
Rule: fit scaler/PCA/probe preprocessing on train fold only
```

Metric policy cho benchmark chính:

- **Accuracy**: hiệu năng tổng thể trên toàn bộ test fold.
- **Macro-F1**: cân bằng precision/recall giữa 4 class, không để class đông chi phối.
- **Balanced Accuracy**: trung bình recall theo class, đo khả năng nhận ra từng class công bằng hơn accuracy.
- **Cohen's Kappa**: mức vượt chance-level sau khi hiệu chỉnh phân bố class thật và phân bố dự đoán.

## 2. Kết quả chính

| Model | Scope | Accuracy | Macro-F1 | Balanced Acc | Cohen's Kappa | Ý nghĩa |
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

Kappa được tính post-hoc từ confusion matrix:

```text
kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
```

Với các row 5-fold có fold-level confusion matrix, Kappa là mean +/- sample std
trên 5 folds. Với BrainOmni/REVE/NeuroLM, đây là fold-1 result nên không có std.
Raw RBF-SVM chỉ còn confusion matrix trung bình trong artifact cũ, nên Kappa là
ước lượng từ averaged confusion matrix; cần rerun riêng SVM baseline nếu muốn
Kappa fold-wise mean +/- std cho row này.

## 3. Transfer-Time / Efficiency Finding

Nên bổ sung thời gian, nhưng phải ghi rõ đây là **recorded wall-clock transfer
component**, không phải một benchmark inference/energy chuẩn hóa. Các nhánh
không log cùng một loại thời gian: SNN/QNN chủ yếu là downstream head training,
foundation models có adapter training, embedding extraction/cache và probe fit.

| Group | Representative row | Accuracy | Recorded transfer time | Cách hiểu |
|---|---|---:|---:|---|
| SNN representation | temporal LIF `spike_count` + RBF-SVC | `86.02 +/- 0.93%` | RBF fit `~0.11 s/fold`; temporal LIF checkpoint training `~1.75 min/fold` nếu tính từ scratch | fastest strong downstream representation readout |
| SNN supervised | differentiable prototype LIF | `84.75 +/- 0.66%` | `~2.31 min/fold` | fastest strong supervised neuromorphic head |
| QNN/QML | `t2_temporal_qdelta` label smoothing + cosine | `82.61 +/- 0.99%` | `~84.4 min/fold` | much slower downstream head; modest accuracy gain |
| Foundation model | EEGPT adapter58 + RBF-SVC | `87.90 +/- 1.82%` | adapter train `~4.74 min/fold`; RBF fit `~5.84 s/fold`; embedding extraction not normalized here | strongest accuracy, but timing depends on cached embedding pipeline |

Claim nên dùng:

```text
SNN provides the best accuracy-time trade-off among non-foundation downstream
heads: it reaches the QNN/foundation-model accuracy regime with minute-scale
training, while QNN recipes require tens of minutes per fold.
```

Claim nên tránh:

```text
SNN is absolutely fastest end-to-end across all model families.
```

Lý do tránh: foundation-model timing cần tách rõ adapter training, frozen
backbone encoding, cached embedding reuse và probe fitting. Nếu muốn claim
runtime nghiêm túc hơn, cần một benchmark thời gian chuẩn hóa chạy lại toàn bộ
pipeline trên cùng hardware.

## 4. Main Contributions

### Contribution 1: Controlled Cross-Family Benchmark

Paper có thể claim:

```text
Chúng tôi xây dựng một benchmark có kiểm soát cho compact 14-channel VEP EEG
classification, đánh giá conventional EEG networks, QNN/QML, SNN và EEG
foundation models trên cùng fixed folds và train-only preprocessing.
```

Giá trị:

- Cùng split và metric cho nhiều họ model.
- Có cả positive và negative results.
- Tách được ba câu hỏi: accuracy, transfer suitability, và model-family claim.

### Contribution 2: Interface-Contract Taxonomy

Thay vì chỉ phân loại model theo kiến trúc, project phân loại theo pretrained
input contract:

| Interface contract | Models | Finding |
|---|---|---|
| Fixed hard channel contract | EEGPT | rất mạnh khi sửa đúng 14->58 channel + sampling contract |
| Fixed canonical montage/temporal contract | LaBraM | yếu nếu temporal interface sai, thành secondary baseline khi align đúng |
| Dynamic channel + electrode positions | REVE, BrainOmni | tiện về interface nhưng chỉ weak-positive trên compact VEP này |
| Named-channel token interface | NeuroLM | named tokens không đủ để rescue short 1s Emotiv14 VEP |
| Dataset-trained local frontend | Shallow/QNN/SNN | mạnh local baseline, không phải pretrained FM transfer |

Claim chính:

```text
Dynamic-channel support is an interface convenience, not sufficient evidence of
representation suitability for compact VEP EEG.
```

### Contribution 3: Frozen-Probe-First Transfer Protocol

Protocol reusable cho dataset EEG mới:

```text
fixed folds
-> train-only scaler
-> minimal correct model interface
-> frozen embeddings
-> LinearSVC / RBF-SVC / PCA+RBF
-> only then train adapter or fine-tune
```

Vì dataset nhỏ, neural head/fine-tuning dễ overfit. Frozen probes giúp quyết
định sớm branch nào đáng chạy 5-fold hoặc adapter.

Bằng chứng:

- EEGPT frozen/adapter embeddings + RBF-SVC là strongest FM path.
- LaBraM chỉ mạnh sau khi sửa temporal interface.
- REVE/BrainOmni/NeuroLM không đạt promotion dù có adapter/interface variants.
- SNN V5a cũng theo logic này: SNN temporal dynamics tạo representation tốt,
  RBF-SVC đọc tốt hơn supervised linear head.

### Contribution 4: Conditional Adapter Gains

Adapter không luôn giúp. Nó chỉ đáng giá khi sửa một mismatch cụ thể.

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

Negative/neutral results là một phần contribution vì chúng loại bỏ overclaim:

- Dynamic-channel REVE/BrainOmni không tự động transfer mạnh.
- NeuroLM named-channel tokens không phù hợp với compact 1s VEP protocol.
- QNN gain quanh Shallow là modest; chưa có quantum advantage.
- Quantum kernel trên SNN embedding thua matched RBF-SVC control.
- EEGPT-QDelta gần như neutral.
- Raw SNN/event encoders yếu.
- SNN prefix/early-exit chưa qua gate, nên không claim early VEP decision.
- EEGPT-teacher distillation vào SNN hiện tại không qua gate dù teacher mạnh.
- V6c/V6e supervised extensions và V5a windowed/fusion probes chưa vượt
  predefined fold-1 gates.
- V6d SupCon/prototype-separation regularization không cải thiện CE-only.
- EEGPT-token SNN plug-in head không qua fold-1 gate; không claim SNN head là
  universal foundation-model readout.

## 5. Suggested Contribution Paragraph

Có thể viết phần contributions trong paper như sau:

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

## 6. Claims Nên Dùng

- EEGPT adapter58 + RBF-SVC là strongest foundation-model baseline.
- V5a SNN temporal LIF `spike_count` + RBF-SVC là strongest SNN representation
  row.
- V6d differentiable prototype LIF là strongest supervised SNN row.
- EEGPT-token SNN plug-in screen là negative control: best fold-1 chỉ `60.86%`
  acc / `59.73%` macro-F1, nên không chạy 5-fold.
- V6/V5a extension screen mostly negative/neutral; V6d CE-only là supervised
  SNN improvement mới nhất.
- Interface alignment là yếu tố quyết định trong compact VEP transfer.
- Dynamic-channel support không đảm bảo representation phù hợp.
- Frozen probing nên đi trước expensive adapter/fine-tuning.
- Adapter gains là conditional, không phải mặc định.

## 7. Claims Nên Tránh

- Không claim dynamic-channel models nói chung là yếu; chỉ nói trong benchmark
  compact Emotiv14 VEP này chúng chưa transfer mạnh.
- Không claim V5a là end-to-end supervised SNN; đó là SNN representation +
  classical nonlinear probe.
- Không claim quantum advantage nếu không beat matched classical controls.
- Không claim EEGPT-QDelta cải thiện meaningful.
- Không claim SNN early VEP decision; prefix gates chưa đạt.
- Không claim V6-style SNN head plug-and-play trên mọi Foundation Model token
  cache; EEGPT token screen không qua gate.
- Không claim QNN gain robust nếu chưa có multi-seed validation.

## 8. Practical Takeaway

Với một EEG dataset mới, phần reusable không phải là `adapter58`,
`stretch_3000`, hay một hyperparameter cụ thể. Phần reusable là protocol:

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

Đây là đóng góp có ý nghĩa ngoài dataset hiện tại.
