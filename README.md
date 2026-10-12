# Compact VEP EEG Recognition

Project độc lập cho nhận dạng Apple, Car, Flower, Human Face từ EEG 14 kênh, 128 Hz, cửa sổ 1 giây. Được tách từ EEG-Foundation-Model-Paper ngày 12/10/2026, theo phạm vi ML/DL/SNN của `docs/MSc__Supervision.pdf`.

**Đọc trước:** [Tổng hợp thí nghiệm và kiểm toán](docs/EXPERIMENTS.md). Điểm 86,02% là kết quả lịch sử trên fold theo epoch, không phải độ chính xác trên subject chưa thấy. Backbone DL cũ dùng test để chọn checkpoint. Probe dùng SNN đã huấn luyện bằng nhãn. Bản brief có các mô tả chưa khớp mã thực tế.

## Thành phần

- `SNN/src/snn_vep/`: mã SNN và backbone Shallow độc lập, không import QNN.
- `SNN/configs/`, `SNN/scripts/`: cấu hình và CLI train, probe, ablation, early exit, Shallow-teacher distillation.
- `SNN/results/`, `SNN/artifacts/`: kết quả gốc, checkpoint và embedding; JSON lịch sử giữ nguyên.
- `baselines/checkpoints/`: 5 backbone Shallow tương ứng các fold lịch sử.
- `results/`: bằng chứng classical ML, DeepConvNet và ShallowConvNet.
- `CSV-FEATURES/`: 59 CSV khôi phục từ Git, 3.612 epoch; có Class/Phase/Subject.
- `trainfold/`: fold lịch sử dựng lại từ thứ tự file trong output notebook, có thêm metadata và hash. Không dùng cho tuyên bố subject generalization.
- `evidence/subject_disjoint/`: manifest và bộ chia subject-disjoint của nghiên cứu sau này, riêng với `trainfold`.
- `evidence/later_benchmarks/`: báo cáo nghiên cứu sau này để truy nguồn, không phải runner của project này.
- `archive/notebooks/`, `archive/baseline_scripts/`: ML/DL gốc và output đã lưu. Khi chạy notebook, đặt cwd kernel về root project trước các cell đọc dữ liệu; baseline scripts đọc `trainfold` tương đối với cwd. DeepConvNet gốc dùng test để chọn checkpoint, chỉ phục vụ tái lập lịch sử.
- `archive/foundation_extensions/`: source của các screen EEGPT âm tính, cần cache ngoài và không thuộc runtime độc lập.
- `preprocessing/`: MATLAB/EEGLAB gốc. Raw EDF/SET không kèm; nếu chạy preprocessing từ đầu cần dữ liệu gốc, EEGLAB/ICLabel và chỉnh đường dẫn MATLAB.

## Cài và chạy

Chạy lệnh từ thư mục root của project. Python >=3.10; môi trường kiểm tra chi tiết ở `docs/environment.json`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/inventory_results.py
python scripts/verify_project.py
```

Tái lập spike-count probe trên checkpoint đã có, ghi kết quả mới vào `runs/`:

```bash
python SNN/scripts/run_shallow_snn_probe.py \
  --config shallow_temporal_lif_head --folds 1 2 3 4 5 \
  --modes spike_count --device cpu --results-dir runs/probe/results
```

Huấn luyện lại head supervised trên backbone đã đóng gói:

```bash
python SNN/scripts/run_shallow_snn_hybrid.py \
  --configs shallow_temporal_spike_aware_proto_lif_ce --folds 1 2 3 4 5 \
  --device auto --results-dir runs/proto/results \
  --checkpoints-dir runs/proto/checkpoints --no-progress
```

Chạy baseline classical lịch sử (cần extra `classical`; xem `--help`):

```bash
python -m pip install -e '.[classical]'
python archive/baseline_scripts/run_classical_ml_baselines.py --help
```

Nếu bỏ thư mục fold để tiết kiệm chỗ, dựng lại bằng `python scripts/prepare_legacy_folds.py`. Script từ chối ghi đè fold đang có. Giữ cả 59 CSV và `docs/legacy_input_order.json`.

CLI gốc có thể ghi đè kết quả/checkpoint nếu dùng mặc định: luôn chỉ định `--results-dir` và `--checkpoints-dir` riêng khi train mới. Một số summarizer có output mặc định trong `SNN/results/summaries`; dùng inventory mới để kiểm tra dữ liệu gốc.

## Tiếp tục nghiên cứu

Cần chạy benchmark mới trên subject-disjoint train/validation/test và train-only preprocessing trước khi kết luận SNN giúp tổng quát hóa. Bộ manifest đã có nhưng head/backbone trong package vẫn gắn với fold lịch sử. Chưa triển khai runner mới cho protocol subject-disjoint, chưa có thí nghiệm trên hai dataset C3 trong brief. Không trộn hai protocol hoặc tái sử dụng backbone epoch-fold cho subject-fold mới.

Dữ liệu gốc: Mendeley Data, DOI `10.17632/g9shp2gxhy.2`.
