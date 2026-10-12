# Kiểm toán và tổng hợp thí nghiệm Compact VEP EEG

Ngày khảo sát: 12/10/2026. Nguồn: code, notebook output, JSON/CSV/checkpoint trên đĩa và MSc supervision brief. Đây là kiểm toán project và bàn giao thực nghiệm; không huấn luyện một benchmark mới.

## Kết luận từ mã và bằng chứng

Đã thực hiện preprocessing, classical ML, nhiều DL architecture, SNN raw-input, Shallow+SNN, probe phi tuyến, early exit, distillation và ablation head. Hai hàng mạnh nhất là spike-count+RBF (86,02%) và prototype LIF CE (84,75%). Chúng là kết quả lịch sử theo epoch; chưa chứng minh ưu thế trên subject chưa thấy.

Các điểm trong PDF cần sửa trước khi dùng làm thesis/paper:

1. “Subject-aware five-fold”: `SVM_Classification.ipynb` cell 10 dùng StratifiedKFold theo hàng, cell 8 loại metadata Subject. Fold dựng lại có **57 recording chung giữa train và test ở mỗi fold**; epoch dài 1 giây, overlap 50%. `evidence/subject_disjoint` mới là bộ chia theo subject, thuộc nghiên cứu sau này.
2. “No additional supervised training of SNN stage”: `SNN/src/snn_vep/hybrid_training.py::run_hybrid_probe` nạp checkpoint của head đã train CE. Chỉ **bước probe** đóng băng SNN; không phải LIF chưa từng học nhãn.
3. “End-to-end”: V2/V6 freeze backbone Shallow. Chỉ downstream SNN/head train jointly; không cập nhật toàn bộ raw-EEG backbone.
4. DL baseline cũ chọn checkpoint/scheduler/early stopping trên test. SNN dùng inner validation, nhưng backbone kế thừa test-selected weights; do đó chưa có so sánh sạch giữa ML/DL/SNN. Hybrid còn fit scaler toàn bộ outer train trước inner validation.
5. Nhiều cấu hình đã được screen trên cùng fold/test và so sánh nhiều lần. Mean±SD không thể tự chứng minh significance hoặc độ tin cậy theo subject.
6. Audit dữ liệu mới ghi **64 recording dự kiến, 59 khả dụng, 1 thiếu raw, 4 preprocessing thất bại, 31 subject có epoch, 1 không có epoch**, khác README cũ ghi 63/63. Xem `evidence/subject_disjoint/audit.json`. Labels kế thừa folder recording, không được khóa theo marker kích thích.

## Phạm vi toàn repository đã khảo sát

| Nhóm | Nội dung | Xử lý trong project mới |
|---|---|---|
| Root MATLAB, CSV, notebook, results | VEP preprocessing và ML/DL gốc | Giữ mã, output và CSV phục hồi |
| SNN | Raw SNN, hybrid, probe, ablation, prefix | Giữ runtime cốt lõi, config, checkpoint, kết quả |
| QNN | Quantum models, có chứa backbone Shallow dùng chung | Chỉ lấy class Shallow, 5 checkpoint, kết quả Shallow |
| FoundationModel_EEGPT/LaBraM/REVE/BrainOmni/NeuroLM | Paper 1: transfer và interface alignment | Ngoài phạm vi; EEGPT SNN screen giữ source ở archive và kết quả gốc |
| paper2_benchmark | Benchmark matched readouts trên VEP/MSS/infant, subject-disjoint, có QNN | Giữ manifest VEP và báo cáo liên quan làm bằng chứng; không mang cả runner đa task |
| labs | MSS 122ch/20 classes, infant 32ch/10 classes, reference, RSOD chưa hoàn tất | Ngoài scope C1/C2; không phải C3 đã hoàn thành |
| sleep_* | Sleep staging, ULW/Linear-BIM/BIM-Explain, cVAN, SNN hypotheses | Dự án khác; không đưa vào Compact VEP |
| research_archive | Closed hypotheses Paper 2/sleep | Ngoài runtime; source vẫn giữ ở repo cũ |

## Pipeline đã có

EDF → EEGLAB → ICA/ICLabel bỏ Eye/Muscle ≥90% → average reference → 0,1–50Hz → sliding epoch 1s, overlap 50%, reject ±150µV → CSV 14×128=1.792 feature. Mã thực tế là `preprocessing/Preprocess_Step4.m`, không phải tên ExportCSV trong README cũ. Pipeline gốc có đường dẫn máy tác giả; CSV đóng gói là input đủ cho runtime hiện tại.

Best hybrid: frozen Shallow temporal map `[B,40,104]` → Linear 40→64 → recurrent LIF → temporal depthwise gate → gated membrane + spike count → fusion → prototype/RBF differentiable head. Best probe: checkpoint temporal LIF đã train → spike_count → StandardScaler → RBF-SVC C=10. Scaler/PCA/SVC probe fit outer train.

## Kết quả chính đã lưu

Bảng dưới dùng SD mẫu (ddof=1), đơn vị %. Xem từng JSON và `experiment_summary.csv`; không trộn SD population của notebook với bảng này.

| Model / readout | Accuracy | Macro-F1 | Balanced accuracy | Nguồn |
|---|---:|---:|---:|---|
| Random Forest | 44,57 ± 2,03 | 41,29 ± 2,02 | 41,60 ± 1,94 | `results/classical_ml/` |
| SVM raw RBF | 41,86 ± 0,97 | 37,41 ± 1,71 | 38,51 ± 1,24 | `results/classical_ml/` |
| DeepConvNet, script snapshot | 77,13 ± 2,07 | 75,83 ± 2,11 | 76,28 ± 2,53 | `results/deepconvnet/` |
| ShallowConvNet, checkpoint snapshot | 82,14 ± 1,01 | 81,33 ± 0,88 | 81,20 ± 0,81 | `results/shallowconvnet/` |
| Temporal LIF | 82,56 ± 1,57 | 81,82 ± 1,44 | 81,62 ± 1,41 | `SNN/results/v2_hybrid/shallow_temporal_lif_head/` |
| Temporal gated LIF | 83,50 ± 0,89 | 82,83 ± 0,84 | 82,78 ± 0,80 | `.../shallow_temporal_gated_lif/` |
| Spike-aware gated LIF | 84,41 ± 1,29 | 83,73 ± 1,13 | 83,67 ± 1,13 | `.../shallow_temporal_spike_aware_gated_lif/` |
| Prototype LIF, CE | 84,75 ± 0,66 | 83,94 ± 0,51 | 83,85 ± 0,41 | `.../shallow_temporal_spike_aware_proto_lif_ce/` |
| Spike-count + PCA/RBF | 85,71 ± 0,86 | 85,04 ± 0,70 | 84,88 ± 0,70 | `.../shallow_temporal_lif_head_embedding_probe_spike_count/`, head 2 |
| Spike-count + RBF | 86,02 ± 0,93 | 85,39 ± 0,82 | 85,29 ± 0,87 | cùng folder, head 1 |

Notebook Shallow có snapshot riêng 81,48±1,08%, không phải checkpoint snapshot 82,14±1,01%. Notebook Deep có 77,05±1,91%, khác script snapshot; notebook thường tính SD population. Giữ riêng các nguồn, không âm thầm thay số. Các notebook EEGNet/TCNet/ITNet/ChannelNet/ATCNet/Conformer/Transformer có output đã chạy nhưng đều bị giới hạn test-selected protocol; chưa thấy đủ CSV riêng cho mọi model.

## Các nhánh SNN đã thử

| Giai đoạn | Đã chạy / kết quả | Trạng thái |
|---|---|---|
| V1 raw SNN | Direct/rate/delta/signed_delta supervised fold 1: 34,58/29,74/46,47/46,75%; best delta embedding+RBF 68,46% | Không promote 5-fold |
| V2 controls | Raw/delta/signed_delta × 3 probe, đủ 5 fold; best signed_delta+PCA/RBF 50,11±0,75% | Transform delta tự nó không giải thích gain |
| V2 Shallow80 heads | LIF 81,53±0,67%, PLIF 81,59±0,67%, đủ 5 fold | Temporal map tốt hơn |
| V3 post-hoc prefixes | K=8/16/24/32/48/64/80/104, đủ 5 fold; accuracy 61,60/69,16/71,35/74,09/75,77/77,08/79,13/82,56% | K80 chưa đạt 80%; không phải chứng minh early raw-input decision |
| V4 prefix-trained / Shallow distill | Screen fold1 và rescue 5 fold; K64 76,97±1,90%, K80 79,01±1,47% | Không qua gate |
| Adaptive signed-delta | Fold1 supervised 46,20%; các diagnostic/probe có trong JSON | Không promote |
| V5a representation probe | Nhiều mode membrane/spike/concat, 3 probe; best spike_count+RBF 86,02% | Best historical representation |
| V5b EEGPT distillation | Student fold1 82,57%, teacher 89,35% | Không promote, ngoài runtime core |
| V6a/b gating và fusion | 83,50%, 84,41%, đủ 5 fold | Positive historical ablations |
| V6c/e windowed/multiscale | Fold1 82,02%, 82,99%; windowed/fusion probes không vượt best | Không promote |
| V6d prototype | CE 84,75%; SupCon005/010/005+sep: 84,08/84,22/84,22%, đều 5 fold | CE tốt nhất; regularizer không cải thiện |
| EEGPT token SNN | Fold1 60,86% / 59,61% | Negative screen, không có C3 |
| Smoke tests | Các folder chứa smoke | Chỉ kiểm tra chạy, không tính thành research results |

Chi tiết mỗi fold, mode, head, train/val/test metric nằm trong `experiment_inventory.csv`; summary có số fold, mean và sample SD. Các file `SNN/results/summaries/*.md` là ghi chú lịch sử và có nhắc QNN/FM; dùng kiểm toán này để hiểu phạm vi đã hiệu chỉnh.

## Bằng chứng nghiên cứu sau này và C3

Báo cáo Paper 2 V2 primary test (490 claims/results, 490 recomputations) ghi **Current VEP không có exact-control effect qua Holm**; positive MSS effects là task khác. Không lấy kết quả này để xác nhận hàng legacy 86%. Inference rule được sửa sau test access, trước đọc aggregate, nên báo cáo tự ghi amended inference.

RESONA/HARP/STAR có screen trên SSVEP Nakanishi2015, 8 kênh/12 classes/3 subject. HARP dừng ở unseen 39,3%, STAR oracle ceiling 30,4%; RESONA S4 không train neural model và chỉ cho phép viết protocol. Không phải chạy pipeline của brief trên UCI Emotiv hoặc MindBigData.

**C3 của brief chưa có dữ liệu/run evidence** cho hai dataset được nêu. Sleep experiments không được tính là C3 của Compact VEP. Các bước còn thiếu: clean subject-disjoint ML/DL/SNN rerun, seed set độc lập, LOSO, paired significance theo protocol phù hợp, ablation preregistered và latency/parameter benchmark toàn pipeline. Training time/spike rate/head-only timing đã có nhưng chưa đủ để khẳng định energy efficiency.

## Đóng gói và validation

Dữ liệu được phục hồi từ Git commit ghi trong `source_manifest.json`; repo gốc không bị reset. Backbone Shallow được chép đúng class cũ sang `snn_vep/backbone.py`, giữ nguyên kiến trúc/state keys; default checkpoint path chuyển sang `baselines/checkpoints`. Kết quả JSON cũ giữ nguyên đường dẫn lịch sử để không sửa provenance.

`legacy_input_order.json` ghi 59 file theo stdout notebook; fold dựng lại có metadata và SHA256. Không có hash fold gốc để chứng minh byte-for-byte giống CSV đã mất. CPU probe chạy lại cho thấy package thực thi được; xem `validation.json` để biết chênh lệch thực tế với score gốc. Không tự quy độ lệch cho một nguyên nhân chưa đo.

Validation CPU đủ 5 fold: RBF đạt **86,05±0,91%** (SD mẫu), so với lịch sử 86,02±0,93%. Fold 2–5 khớp confusion matrix; fold1 lệch 1 mẫu (+0,1383 điểm phần trăm). PCA/RBF fold1 khớp cả confusion matrix. 693 bộ metric đã tính lại từ confusion matrix với tolerance 1e-9; 5 backbone và 10 checkpoint head chính nạp strict thành công, đúng shape và surrogate-gradient bước train. Đây là validation đóng gói trên môi trường ghi lại, không phải benchmark subject-disjoint mới.
