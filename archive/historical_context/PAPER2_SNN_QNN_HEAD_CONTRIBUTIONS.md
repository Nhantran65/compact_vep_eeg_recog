# Paper 2 Contributions - Efficient EEG Downstream Heads: SNN vs QNN

Last updated: 2026-06-27

Source split: `PAPER_CONTRIBUTIONS.md`

## Core Direction

- This paper should be framed as a benchmark of **downstream heads/readouts under a matched EEG frontend**, not a full model-family comparison from raw EEG. The QNN branch currently uses ShallowConvNet features/logits and adds QResFusion/QDelta/temporal QDelta. The strongest SNN branch also uses the frozen ShallowConvNet temporal feature map and adds a LIF temporal readout, a spike-count probe, or a differentiable prototype head. The right question is therefore which kind of downstream temporal/readout module adds the most value once the frontend is controlled.

- The short thesis should be: **spiking temporal readouts provide a better accuracy-time trade-off than quantum-classical residual heads under matched EEG frontend controls**. This claim is narrower than "SNN beats QNN", but more defensible. It allows a fair comparison between QNN, SNN, and classical controls on the same folds, the same preprocessing, and the same ShallowConvNet feature source. The paper's strength will lie in matched controls and efficiency analysis, not in a plain leaderboard.

## Current Contributions

- **Contribution 1: Controlled head-level comparison.** QNN and SNN are placed after the same frontend family, so the paper examines downstream adaptation instead of conflating different frontends. QNN includes QResFusion, logit-side QDelta, temporal QDelta, and matched MLP controls. SNN includes the temporal LIF readout, spike-count representation probe, spike-aware gated head, and differentiable prototype/RBF-style head. This is a clean setup for asking whether a quantum/spiking head really adds information.

- **Contribution 2: SNN temporal representation is strong.** The current strongest row is the frozen Shallow temporal feature map `[B,40,104] -> LIF -> spike_count -> RBF-SVC`, reaching `86.02 +/- 0.93%` accuracy. This row should be called a representation-probe result, not pure end-to-end SNN superiority. The cleaner supervised SNN row is the differentiable prototype LIF, reaching `84.75 +/- 0.66%`. These two rows should be presented separately so reviewers cannot say the paper conflates probe and architecture.

- **Contribution 3: QNN gains are modest and not quantum-advantage evidence.** The current best QNN row is about `82.61 +/- 0.99%`, only modestly above ShallowConvNet and much slower. A quantum kernel on SNN/Shallow embeddings also loses to the matched RBF-SVC control. The QNN branch is therefore valuable as controlled negative/diagnostic evidence: quantum heads must beat matched classical controls before claiming quantum advantage. This is a disciplined claim that reviewers are more likely to respect than an overclaim.

- **Contribution 4: Accuracy-time trade-off.** SNN downstream heads reach a high-accuracy regime with minute-scale training, while QNN recipes can take tens of minutes per fold. Current timing must be called a recorded transfer component, not a standardized end-to-end runtime benchmark. If timing is re-standardized, the paper can contribute a clear trade-off analysis: accuracy, training time, inference time, parameter count, and hardware dependency.

## Scope After Split

- This paper should not use foundation models as a main contribution. EEGPT/LaBraM may appear only as external upper baselines if needed, but the interface-alignment story should not be pulled in. The focus should be efficient downstream adaptation after a controlled EEG frontend. Mixing in foundation-model transfer would make the paper lose its central question.

- The current scope is still narrow if run only on one compact VEP dataset. A top-conference version should show that the SNN/QNN/head-level trade-off conclusion replicates across multiple EEG tasks. Adding 2 datasets under the same protocol turns the paper from a "single-dataset ablation" into a "multi-dataset study of efficient EEG downstream heads".

## What To Add

- Run at least 2 more public datasets, for 3 datasets in total. Choose different tasks to test generality: the current compact VEP, motor imagery, and ERP/P300 or cognitive/emotion EEG. Each dataset needs fixed folds or a clear subject-wise split, the same train-only scaler, the same metric policy, and the same promotion gates. Do not change splits randomly per model.

- Add at least one second frontend besides ShallowConvNet. With only ShallowConvNet, reviewers may argue that SNN wins because it suits this particular feature map. An EEGNet or compact temporal CNN frontend would test whether the conclusion depends on the frontend. The protocol should be `frontend frozen feature map -> downstream head`, with frontend checkpoints trained on the same folds.

- Add stronger matched classical controls. For each dataset/frontend, include a linear head, MLP head, small temporal MLP/TCN or GRU, RBF-SVC/PCA-RBF, MLP-QDelta, QNN-QDelta, and SNN LIF/prototype. QNN counts as having signal only if it beats matched MLP controls; SNN counts as having signal only if it beats non-spiking temporal controls. This turns the paper into controlled science rather than a model zoo.

- Standardize runtime and resource reporting. Report train time, inference time per sample, parameter count, GPU/CPU used, quantum simulator backend, and memory if possible. QNN timing must separate quantum circuit simulation cost from PyTorch overhead; SNN timing must separate feature extraction/caching from head training. If cached frontend features are used, report both cached-probe timing and from-scratch timing.

- Add analysis explaining why SNN wins. Include spike_count vs membrane vs final/max membrane ablation, temporal prefix/early-window behavior, prototype separability, confusion/error overlap between Shallow/QNN/SNN, and calibration/uncertainty if available. For QNN, analyze when QDelta actually changes the decision boundary and how many samples are fixed/harmed. These analyses help the paper go beyond an accuracy table.

## Claim Discipline

- Claims to use: under matched frontend controls, SNN temporal readouts show a stronger accuracy-time trade-off than QNN residual heads on the evaluated EEG tasks. The current best SNN result is a representation-probe row, while the differentiable prototype LIF is the cleaner supervised architecture row. QNN results are modest and should be interpreted as controlled quantum-classical head ablations, not quantum advantage.

- Claims to avoid: SNN is better than QNN for EEG classification in general, SNN is end-to-end superior, or QNN is ineffective as a family. Current evidence only speaks to downstream heads after a learned EEG frontend. If added datasets give inconsistent results, the framing can still hold: spiking temporal dynamics are a strong and efficient downstream adaptation mechanism, while quantum residual heads require stronger evidence against classical controls.
