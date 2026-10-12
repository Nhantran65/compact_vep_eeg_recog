from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from .configs import config_name, load_config
from .constants import DEFAULT_CHECKPOINTS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_TRAINFOLD_DIR
from .data import prepare_fold_loaders
from .metrics import classification_metrics
from .models import build_model
from .utils import ensure_parent, resolve_device, set_seed


def result_path(results_dir: str | Path, config: dict[str, Any], fold: int) -> Path:
    return Path(results_dir) / "by_config" / config_name(config) / f"fold_{fold}.json"


def checkpoint_path(checkpoints_dir: str | Path, config: dict[str, Any], fold: int) -> Path:
    return Path(checkpoints_dir) / config_name(config) / f"fold_{fold}.pt"


def _loop_batches(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    spike_rates: list[float] = []

    for batch_idx, (inputs, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs = inputs.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits, aux = model(inputs, return_aux=True)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        losses.append(float(loss.item()) * labels.size(0))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        spike_rates.append(float(aux["spike_rate"].detach().cpu().item()))

    if not y_true:
        raise RuntimeError("No batches were processed")
    metrics = classification_metrics(np.array(y_true), np.array(y_pred))
    metrics["loss"] = float(np.sum(losses) / len(y_true))
    metrics["spike_rate"] = float(np.mean(spike_rates))
    return metrics


def evaluate_model(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int | None = None,
) -> dict:
    with torch.no_grad():
        return _loop_batches(
            model,
            loader,
            device=device,
            criterion=criterion,
            optimizer=None,
            max_batches=max_batches,
        )


def train_one_fold(
    config_or_path: dict[str, Any] | str | Path,
    *,
    fold: int,
    trainfold_dir: str | Path = DEFAULT_TRAINFOLD_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    checkpoints_dir: str | Path = DEFAULT_CHECKPOINTS_DIR,
    device: str = "auto",
    progress: bool = True,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else copy.deepcopy(config_or_path)
    training = config.get("training", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    torch_device = resolve_device(device)

    batch_size = int(batch_size_override or training.get("batch_size", 64))
    epochs = int(epochs_override or training.get("epochs", 120))
    patience = int(training.get("patience", 20))

    loaders = prepare_fold_loaders(
        trainfold_dir,
        fold,
        batch_size=batch_size,
        val_size=float(training.get("val_size", 0.20)),
        seed=seed,
    )
    model = build_model(config).to(torch_device)
    criterion = nn.CrossEntropyLoss()
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
    best_val_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history: list[dict] = []
    start = time.time()

    iterator = range(1, epochs + 1)
    if progress:
        iterator = tqdm(iterator, desc=f"{config_name(config)} fold {fold}")

    for epoch in iterator:
        train_metrics = _loop_batches(
            model,
            loaders.train_loader,
            device=torch_device,
            criterion=criterion,
            optimizer=optimizer,
            max_batches=max_train_batches,
        )
        val_metrics = evaluate_model(
            model,
            loaders.val_loader,
            device=torch_device,
            criterion=criterion,
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
            iterator.set_postfix(
                val_acc=f"{val_metrics['accuracy']:.3f}",
                val_f1=f"{val_metrics['macro_f1']:.3f}",
                best=best_epoch,
            )

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    train_metrics = evaluate_model(
        model,
        loaders.train_loader,
        device=torch_device,
        criterion=criterion,
        max_batches=max_eval_batches,
    )
    test_metrics = evaluate_model(
        model,
        loaders.test_loader,
        device=torch_device,
        criterion=criterion,
        max_batches=max_eval_batches,
    )
    seconds = time.time() - start

    ckpt_path = checkpoint_path(checkpoints_dir, config, fold)
    ensure_parent(ckpt_path)
    torch.save(
        {
            "config": config,
            "fold": fold,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "metadata": loaders.tensors.metadata,
        },
        ckpt_path,
    )

    result = {
        "name": config_name(config),
        "fold": fold,
        "config": config,
        "data": loaders.tensors.metadata,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "seconds": seconds,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }

    out_path = result_path(results_dir, config, fold)
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
