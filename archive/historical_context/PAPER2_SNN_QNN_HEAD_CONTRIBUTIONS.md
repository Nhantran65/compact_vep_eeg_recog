# Paper 2 Contributions - Efficient EEG Downstream Heads: SNN vs QNN

Last updated: 2026-06-27

Source split: `PAPER_CONTRIBUTIONS.md`

## Core Direction

- Paper này nên được frame là benchmark về **downstream heads/readouts under a matched EEG frontend**, không phải full model-family comparison từ raw EEG. QNN branch hiện dùng ShallowConvNet feature/logit rồi thêm QResFusion/QDelta/temporal QDelta. SNN branch mạnh nhất cũng dùng frozen ShallowConvNet temporal feature map rồi thêm LIF temporal readout, spike-count probe, hoặc differentiable prototype head. Vì vậy câu hỏi đúng là loại downstream temporal/readout module nào thêm giá trị tốt nhất khi frontend đã được kiểm soát.

- Thesis ngắn gọn nên là: **spiking temporal readouts provide a better accuracy-time trade-off than quantum-classical residual heads under matched EEG frontend controls**. Claim này hẹp hơn "SNN beats QNN", nhưng defensible hơn. Nó cho phép so sánh công bằng giữa QNN, SNN, và classical controls trên cùng folds, cùng preprocessing, cùng ShallowConvNet feature source. Điểm mạnh của paper sẽ nằm ở matched controls và efficiency analysis, không phải leaderboard đơn thuần.

## Current Contributions

- **Contribution 1: Controlled head-level comparison.** QNN và SNN được đặt sau cùng một family frontend, nên paper kiểm tra phần downstream adaptation thay vì lẫn lộn frontend khác nhau. QNN gồm QResFusion, logit-side QDelta, temporal QDelta, và matched MLP controls. SNN gồm temporal LIF readout, spike-count representation probe, spike-aware gated head, và differentiable prototype/RBF-style head. Đây là setup sạch để hỏi quantum/spiking head có thật sự thêm thông tin không.

- **Contribution 2: SNN temporal representation is strong.** Row mạnh nhất hiện tại là frozen Shallow temporal feature map `[B,40,104] -> LIF -> spike_count -> RBF-SVC`, đạt `86.02 +/- 0.93%` accuracy. Row này nên được gọi là representation-probe result, không phải pure end-to-end SNN superiority. Supervised SNN row sạch hơn là differentiable prototype LIF, đạt `84.75 +/- 0.66%`. Hai row này nên được trình bày tách biệt để tránh reviewer nói paper nhập nhằng giữa probe và architecture.

- **Contribution 3: QNN gains are modest and not quantum-advantage evidence.** QNN best row hiện tại khoảng `82.61 +/- 0.99%`, chỉ nhỉnh hơn ShallowConvNet modestly và chậm hơn nhiều. Quantum kernel trên SNN/Shallow embedding cũng thua matched RBF-SVC control. Vì vậy QNN branch có giá trị như controlled negative/diagnostic evidence: quantum heads cần beat matched classical controls trước khi claim quantum advantage. Đây là claim kỷ luật và có khả năng được reviewer tôn trọng hơn overclaim.

- **Contribution 4: Accuracy-time trade-off.** SNN downstream heads đạt vùng accuracy cao với minute-scale training, trong khi QNN recipes có thể mất tens of minutes per fold. Timing hiện tại phải được gọi là recorded transfer component, không phải standardized end-to-end runtime benchmark. Nếu chuẩn hóa lại timing, paper có thể đóng góp một trade-off analysis rõ: accuracy, training time, inference time, parameter count, and hardware dependency.

## Scope After Split

- Paper này không nên dùng foundation models như contribution chính. EEGPT/LaBraM chỉ có thể xuất hiện như external upper baselines nếu cần, nhưng không nên kéo interface-alignment story vào đây. Trọng tâm nên là efficient downstream adaptation after a controlled EEG frontend. Nếu lẫn foundation-model transfer vào, paper sẽ mất câu hỏi trung tâm.

- Scope hiện tại còn hẹp nếu chỉ chạy trên một compact VEP dataset. Một top-conference version nên chứng minh kết luận SNN/QNN/head-level trade-off lặp lại trên nhiều task EEG. Nếu thêm 2 datasets và giữ cùng protocol, paper sẽ chuyển từ "single-dataset ablation" thành "multi-dataset study of efficient EEG downstream heads".

## What To Add

- Cần chạy thêm ít nhất 2 public datasets, tổng cộng 3 datasets. Nên chọn task khác nhau để kiểm tra generality: compact VEP hiện tại, motor imagery, và ERP/P300 hoặc cognitive/emotion EEG. Với mỗi dataset, cần fixed folds hoặc subject-wise split rõ, cùng train-only scaler, cùng metric policy, và cùng promotion gates. Không nên đổi split ngẫu nhiên theo từng model.

- Cần thêm ít nhất một frontend thứ hai ngoài ShallowConvNet. Nếu chỉ dùng ShallowConvNet, reviewer có thể nói SNN thắng vì nó hợp với feature map cụ thể này. Một EEGNet hoặc compact temporal CNN frontend sẽ giúp kiểm tra kết luận có phụ thuộc frontend không. Protocol nên là `frontend frozen feature map -> downstream head`, với frontend checkpoints được train cùng folds.

- Cần matched classical controls mạnh hơn. Với mỗi dataset/frontend, nên có linear head, MLP head, temporal MLP/TCN hoặc GRU nhỏ, RBF-SVC/PCA-RBF, MLP-QDelta, QNN-QDelta, SNN LIF/prototype. QNN chỉ được coi là có signal nếu beat matched MLP controls; SNN chỉ được coi là có signal nếu beat non-spiking temporal controls. Điều này biến paper thành controlled science thay vì model zoo.

- Cần chuẩn hóa runtime và resource reporting. Report train time, inference time per sample, parameter count, GPU/CPU used, quantum simulator backend, and memory if possible. QNN timing phải tách quantum circuit simulation cost khỏi PyTorch overhead; SNN timing phải tách feature extraction/cache khỏi head training. Nếu dùng cached frontend features, phải report cả cached-probe timing và from-scratch timing.

- Cần thêm analysis để giải thích vì sao SNN thắng. Nên có spike_count vs membrane vs final/max membrane ablation, temporal prefix/early-window behavior, prototype separability, confusion/error overlap giữa Shallow/QNN/SNN, và calibration/uncertainty nếu có. Với QNN, nên phân tích khi nào QDelta thật sự đổi decision boundary và bao nhiêu sample được sửa/hại. Các analysis này giúp paper vượt khỏi một bảng accuracy.

## Claim Discipline

- Claim nên dùng: under matched frontend controls, SNN temporal readouts show a stronger accuracy-time trade-off than QNN residual heads on the evaluated EEG tasks. The current best SNN result is a representation-probe row, while the differentiable prototype LIF is the cleaner supervised architecture row. QNN results are modest and should be interpreted as controlled quantum-classical head ablations, not quantum advantage.

- Claim nên tránh: SNN is better than QNN for EEG classification in general, SNN is end-to-end superior, or QNN is ineffective as a family. Hiện tại evidence chỉ nói về downstream heads after a learned EEG frontend. Nếu thêm datasets cho kết quả không đồng nhất, framing vẫn có thể giữ: spiking temporal dynamics are a strong and efficient downstream adaptation mechanism, while quantum residual heads require stronger evidence against classical controls.

