from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .configs import config_name, load_config
from .constants import DEFAULT_CHECKPOINTS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_TRAINFOLD_DIR, N_CHANNELS, N_TIMES
from .data import encode_labels, load_fold_arrays
from .hybrid_models import ShallowFeatureSNNHead
from .metrics import classification_metrics
from .shallow_features import ShallowFeatureExtractor, shallow_checkpoint_path
from .utils import ensure_parent, resolve_device, set_seed


def prefix_distill_checkpoint_path(
    checkpoints_dir: str | Path,
    config: dict[str, Any],
    fold: int,
) -> Path:
    return Path(checkpoints_dir) / "v4_prefix_distill" / config_name(config) / f"fold_{fold}.pt"


def _standardized_fold_arrays(trainfold_dir: str | Path, fold: int):
    arrays = load_fold_arrays(trainfold_dir, fold)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(arrays.X_train).astype(np.float32)
    x_test = scaler.transform(arrays.X_test).astype(np.float32)
    x_train = x_train.reshape(-1, N_CHANNELS, N_TIMES)
    x_test = x_test.reshape(-1, N_CHANNELS, N_TIMES)
    y_train = encode_labels(arrays.y_train)
    y_test = encode_labels(arrays.y_test)
    return x_train, y_train, x_test, y_test, scaler


def _extract_temporal_features_and_teacher_logits(
    extractor: ShallowFeatureExtractor,
    x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    extractor.to(device)
    extractor.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    feature_chunks = []
    logit_chunks = []
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            features = extractor.extract_temporal_map(inputs)
            logits = extractor.backbone(extractor._as_shallow_input(inputs))
            feature_chunks.append(features.detach().cpu().numpy())
            logit_chunks.append(logits.detach().cpu().numpy())
    return (
        np.concatenate(feature_chunks, axis=0).astype(np.float32),
        np.concatenate(logit_chunks, axis=0).astype(np.float32),
    )


def _prepare_distill_loaders(
    *,
    fold: int,
    trainfold_dir: str | Path,
    shallow_checkpoint_template: str,
    batch_size: int,
    feature_batch_size: int,
    val_size: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    x_train_raw, y_train, x_test_raw, y_test, _ = _standardized_fold_arrays(trainfold_dir, fold)
    checkpoint = shallow_checkpoint_path(fold, template=shallow_checkpoint_template)
    extractor = ShallowFeatureExtractor(checkpoint)
    x_train_features, train_teacher_logits = _extract_temporal_features_and_teacher_logits(
        extractor,
        x_train_raw,
        batch_size=feature_batch_size,
        device=device,
    )
    x_test_features, test_teacher_logits = _extract_temporal_features_and_teacher_logits(
        extractor,
        x_test_raw,
        batch_size=feature_batch_size,
        device=device,
    )

    indices = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=y_train,
    )

    def make_loader(features, labels, logits, shuffle: bool):
        return DataLoader(
            TensorDataset(
                torch.from_numpy(features),
                torch.from_numpy(labels).long(),
                torch.from_numpy(logits),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_teacher_pred = np.argmax(train_teacher_logits, axis=1)
    test_teacher_pred = np.argmax(test_teacher_logits, axis=1)
    return {
        "train_loader": make_loader(
            x_train_features[train_idx],
            y_train[train_idx],
            train_teacher_logits[train_idx],
            True,
        ),
        "val_loader": make_loader(
            x_train_features[val_idx],
            y_train[val_idx],
            train_teacher_logits[val_idx],
            False,
        ),
        "full_train_loader": make_loader(x_train_features, y_train, train_teacher_logits, False),
        "test_loader": make_loader(x_test_features, y_test, test_teacher_logits, False),
        "metadata": {
            "fold": fold,
            "feature_mode": "temporal_map",
            "checkpoint_template": shallow_checkpoint_template,
            "checkpoint_path": str(checkpoint),
            "scaler_fit": "full_train_fold",
            "train_size": int(len(train_idx)),
            "val_size_count": int(len(val_idx)),
            "full_train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "feature_shape": list(x_train_features.shape[1:]),
            "teacher_train_metrics": classification_metrics(y_train, train_teacher_pred),
            "teacher_test_metrics": classification_metrics(y_test, test_teacher_pred),
        },
    }


def _distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    alpha: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss = F.cross_entropy(student_logits, labels)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    loss = (1.0 - alpha) * ce_loss + alpha * kd_loss
    return loss, ce_loss, kd_loss


def _loop_batches(
    model: ShallowFeatureSNNHead,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    prefix_steps: int,
    alpha: float,
    temperature: float,
    max_batches: int | None = None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    ce_losses = []
    kd_losses = []
    y_true = []
    y_pred = []
    spike_rates = []

    for batch_idx, (features, labels, teacher_logits) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        features = features.to(device)
        labels = labels.to(device)
        teacher_logits = teacher_logits.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits, aux = model(features, return_aux=True, prefix_steps=prefix_steps)
            loss, ce_loss, kd_loss = _distillation_loss(
                logits,
                labels,
                teacher_logits,
                alpha=alpha,
                temperature=temperature,
            )
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        losses.append(float(loss.item()) * labels.size(0))
        ce_losses.append(float(ce_loss.item()) * labels.size(0))
        kd_losses.append(float(kd_loss.item()) * labels.size(0))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        spike_rates.append(float(aux["spike_rate"].detach().cpu().item()))

    if not y_true:
        raise RuntimeError("No batches were processed")
    metrics = classification_metrics(np.array(y_true), np.array(y_pred))
    denom = float(len(y_true))
    metrics["loss"] = float(np.sum(losses) / denom)
    metrics["ce_loss"] = float(np.sum(ce_losses) / denom)
    metrics["kd_loss"] = float(np.sum(kd_losses) / denom)
    metrics["spike_rate"] = float(np.mean(spike_rates))
    return metrics


def _evaluate(
    model: ShallowFeatureSNNHead,
    loader: DataLoader,
    *,
    device: torch.device,
    prefix_steps: int,
    alpha: float,
    temperature: float,
    max_batches: int | None = None,
) -> dict[str, Any]:
    with torch.no_grad():
        return _loop_batches(
            model,
            loader,
            device=device,
            optimizer=None,
            prefix_steps=prefix_steps,
            alpha=alpha,
            temperature=temperature,
            max_batches=max_batches,
        )


def train_prefix_distill_fold(
    config_or_path: dict[str, Any] | str | Path,
    *,
    fold: int,
    trainfold_dir: str | Path = DEFAULT_TRAINFOLD_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    checkpoints_dir: str | Path = DEFAULT_CHECKPOINTS_DIR,
    shallow_checkpoint_template: str = "baselines/checkpoints/shallow_original_fold{fold}_seed42.pth",
    device: str = "auto",
    progress: bool = True,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    prefix_steps_override: int | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else copy.deepcopy(config_or_path)
    training = config.get("training", {})
    distill = config.get("distillation", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    torch_device = resolve_device(device)
    batch_size = int(batch_size_override or training.get("batch_size", 64))
    epochs = int(epochs_override or training.get("epochs", 120))
    patience = int(training.get("patience", 20))
    prefix_steps = int(prefix_steps_override or distill.get("prefix_steps", 64))
    alpha = float(distill.get("alpha", 0.5))
    temperature = float(distill.get("temperature", 2.0))
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("distillation alpha must be in [0,1]")
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")

    loaders = _prepare_distill_loaders(
        fold=fold,
        trainfold_dir=trainfold_dir,
        shallow_checkpoint_template=shallow_checkpoint_template,
        batch_size=batch_size,
        feature_batch_size=int(training.get("feature_batch_size", 256)),
        val_size=float(training.get("val_size", 0.20)),
        seed=seed,
        device=torch_device,
    )
    feature_shape = loaders["metadata"]["feature_shape"]
    if len(feature_shape) != 2 or prefix_steps < 1 or prefix_steps > int(feature_shape[1]):
        raise ValueError(f"Invalid prefix_steps={prefix_steps} for feature shape {feature_shape}")

    model = ShallowFeatureSNNHead.from_config(config).to(torch_device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training.get("lr", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(training.get("scheduler_factor", 0.5)),
        patience=int(training.get("scheduler_patience", 8)),
    )

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_macro_f1 = -1.0
    best_val_metrics = None
    epochs_without_improvement = 0
    history = []
    start = time.time()

    iterator = range(1, epochs + 1)
    if progress:
        iterator = tqdm(iterator, desc=f"{config_name(config)} fold {fold}")
    for epoch in iterator:
        train_metrics = _loop_batches(
            model,
            loaders["train_loader"],
            device=torch_device,
            optimizer=optimizer,
            prefix_steps=prefix_steps,
            alpha=alpha,
            temperature=temperature,
            max_batches=max_train_batches,
        )
        val_metrics = _evaluate(
            model,
            loaders["val_loader"],
            device=torch_device,
            prefix_steps=prefix_steps,
            alpha=alpha,
            temperature=temperature,
            max_batches=max_eval_batches,
        )
        scheduler.step(val_metrics["macro_f1"])
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(val_f1=f"{val_metrics['macro_f1']:.3f}", best=best_epoch)
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    train_metrics = _evaluate(
        model,
        loaders["train_loader"],
        device=torch_device,
        prefix_steps=prefix_steps,
        alpha=alpha,
        temperature=temperature,
        max_batches=max_eval_batches,
    )
    test_metrics = _evaluate(
        model,
        loaders["test_loader"],
        device=torch_device,
        prefix_steps=prefix_steps,
        alpha=alpha,
        temperature=temperature,
        max_batches=max_eval_batches,
    )
    seconds = time.time() - start

    ckpt_path = prefix_distill_checkpoint_path(checkpoints_dir, config, fold)
    ensure_parent(ckpt_path)
    torch.save(
        {
            "config": config,
            "fold": fold,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "metadata": loaders["metadata"],
            "prefix_steps": prefix_steps,
            "alpha": alpha,
            "temperature": temperature,
        },
        ckpt_path,
    )
    result = {
        "name": config_name(config),
        "fold": fold,
        "config": config,
        "data": loaders["metadata"],
        "prefix_steps": prefix_steps,
        "alpha": alpha,
        "temperature": temperature,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "seconds": seconds,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }
    out_path = Path(results_dir) / "v4_prefix_distill" / config_name(config) / f"fold_{fold}.json"
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
