from __future__ import annotations

import copy
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from tqdm.auto import tqdm

from .configs import config_name, load_config
from .constants import DEFAULT_CHECKPOINTS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_TRAINFOLD_DIR
from .hybrid_models import build_hybrid_model
from .metrics import classification_metrics
from .shallow_features import prepare_shallow_feature_loaders
from .utils import ensure_parent, resolve_device, set_seed


def _supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.10,
) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"SupCon expects [B,D] embeddings, got {tuple(embeddings.shape)}")
    if temperature <= 0.0:
        raise ValueError("SupCon temperature must be positive")
    batch_size = embeddings.shape[0]
    if batch_size < 2:
        return embeddings.sum() * 0.0
    z = F.normalize(embeddings, dim=1)
    logits = (z @ z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    valid_pair_mask = ~self_mask
    positive_mask = (labels[:, None] == labels[None, :]) & valid_pair_mask
    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0
    if not bool(valid_anchor.any()):
        return embeddings.sum() * 0.0
    exp_logits = torch.exp(logits) * valid_pair_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count.clamp_min(1)
    return -mean_log_prob_pos[valid_anchor].mean()


def _compute_hybrid_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    aux: dict[str, torch.Tensor],
    criterion: nn.Module,
    loss_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    ce_loss = criterion(logits, labels)
    supcon_weight = float(loss_config.get("lambda_supcon", loss_config.get("supcon_weight", 0.0)))
    proto_sep_weight = float(
        loss_config.get("lambda_proto_sep", loss_config.get("proto_sep_weight", 0.0))
    )
    supcon_loss = ce_loss.new_zeros(())
    proto_sep_loss = ce_loss.new_zeros(())
    if supcon_weight > 0.0:
        if "embedding" not in aux:
            raise ValueError("SupCon loss requires model aux['embedding']")
        supcon_loss = _supervised_contrastive_loss(
            aux["embedding"],
            labels,
            temperature=float(loss_config.get("supcon_temperature", 0.10)),
        )
    if proto_sep_weight > 0.0:
        if "prototype_sep_loss" not in aux:
            raise ValueError("Prototype separation loss requires model aux['prototype_sep_loss']")
        proto_sep_loss = aux["prototype_sep_loss"]
    total_loss = ce_loss + supcon_weight * supcon_loss + proto_sep_weight * proto_sep_loss
    return {
        "total_loss": total_loss,
        "ce_loss": ce_loss,
        "supcon_loss": supcon_loss,
        "proto_sep_loss": proto_sep_loss,
    }


def _loop_batches(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    criterion: nn.Module,
    loss_config: dict[str, Any] | None = None,
    optimizer: optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict:
    is_train = optimizer is not None
    loss_config = loss_config or {}
    model.train(is_train)
    loss_sums = {
        "total_loss": 0.0,
        "ce_loss": 0.0,
        "supcon_loss": 0.0,
        "proto_sep_loss": 0.0,
    }
    y_true = []
    y_pred = []
    aux_values: dict[str, list[float]] = {"spike_rate": [], "mean_gate": [], "gamma": []}
    for batch_idx, (features, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        features = features.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits, aux = model(features, return_aux=True)
            loss_terms = _compute_hybrid_loss(
                logits=logits,
                labels=labels,
                aux=aux,
                criterion=criterion,
                loss_config=loss_config,
            )
            loss = loss_terms["total_loss"]
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        for key, value in loss_terms.items():
            loss_sums[key] += float(value.detach().cpu().item()) * labels.size(0)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        for key in aux_values:
            if key in aux:
                aux_values[key].append(float(aux[key].detach().cpu().item()))
    metrics = classification_metrics(np.array(y_true), np.array(y_pred))
    denom = max(len(y_true), 1)
    metrics["loss"] = float(loss_sums["total_loss"] / denom)
    for key, value in loss_sums.items():
        metrics[key] = float(value / denom)
    for key, values in aux_values.items():
        if values:
            metrics[key] = float(np.mean(values))
    return metrics


def _evaluate(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    criterion: nn.Module,
    loss_config: dict[str, Any] | None = None,
) -> dict:
    with torch.no_grad():
        return _loop_batches(
            model,
            loader,
            device=device,
            criterion=criterion,
            loss_config=loss_config,
            optimizer=None,
        )


def hybrid_checkpoint_path(checkpoints_dir: str | Path, config: dict[str, Any], fold: int) -> Path:
    return Path(checkpoints_dir) / "v2_hybrid" / config_name(config) / f"fold_{fold}.pt"


def train_hybrid_fold(
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
    max_train_batches: int | None = None,
) -> dict:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else copy.deepcopy(config_or_path)
    training = config.get("training", {})
    model_config = config.get("model", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    torch_device = resolve_device(device)
    batch_size = int(batch_size_override or training.get("batch_size", 64))
    epochs = int(epochs_override or training.get("epochs", 120))
    patience = int(training.get("patience", 20))
    loss_config = config.get("loss", {})

    loaders = prepare_shallow_feature_loaders(
        fold=fold,
        feature_mode=model_config.get("feature_mode", "80d"),
        checkpoint_template=shallow_checkpoint_template,
        trainfold_dir=trainfold_dir,
        batch_size=batch_size,
        feature_batch_size=int(training.get("feature_batch_size", 256)),
        val_size=float(training.get("val_size", 0.20)),
        seed=seed,
        device=torch_device,
    )
    model = build_hybrid_model(config).to(torch_device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(loss_config.get("label_smoothing", training.get("label_smoothing", 0.0)))
    )
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
            criterion=criterion,
            loss_config=loss_config,
            optimizer=optimizer,
            max_batches=max_train_batches,
        )
        val_metrics = _evaluate(
            model,
            loaders["val_loader"],
            device=torch_device,
            criterion=criterion,
            loss_config=loss_config,
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
        criterion=criterion,
        loss_config=loss_config,
    )
    test_metrics = _evaluate(
        model,
        loaders["test_loader"],
        device=torch_device,
        criterion=criterion,
        loss_config=loss_config,
    )
    seconds = time.time() - start

    ckpt_path = hybrid_checkpoint_path(checkpoints_dir, config, fold)
    ensure_parent(ckpt_path)
    torch.save(
        {
            "config": config,
            "fold": fold,
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "metadata": loaders["metadata"],
        },
        ckpt_path,
    )
    result = {
        "name": config_name(config),
        "fold": fold,
        "config": config,
        "data": loaders["metadata"],
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "seconds": seconds,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }
    out_path = Path(results_dir) / "v2_hybrid" / config_name(config) / f"fold_{fold}.json"
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def _extract_embeddings(model, loader, *, device: torch.device, mode: str):
    model.eval()
    xs = []
    ys = []
    with torch.no_grad():
        for features, labels in loader:
            emb = model.extract_embedding(features.to(device), mode=mode)
            xs.append(emb.detach().cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _probe_estimator(name: str, x_train: np.ndarray, seed: int):
    dual = x_train.shape[1] > x_train.shape[0]
    if name == "linearsvc_c1":
        return make_pipeline(StandardScaler(), LinearSVC(C=1.0, dual=dual, max_iter=5000, tol=1e-3))
    if name == "rbf_svc_c10_gamma_scale":
        return make_pipeline(StandardScaler(), SVC(C=10.0, kernel="rbf", gamma="scale", cache_size=1000))
    if name == "pca128_rbf_svc_c3_gamma_scale":
        n_components = min(128, x_train.shape[1], x_train.shape[0] - 1)
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=seed),
            SVC(C=3.0, kernel="rbf", gamma="scale", cache_size=1000),
        )
    raise ValueError(f"Unsupported hybrid probe head: {name}")


def _rbf_probe_estimator() -> Any:
    return make_pipeline(
        StandardScaler(),
        SVC(
            C=10.0,
            kernel="rbf",
            gamma="scale",
            decision_function_shape="ovr",
            cache_size=1000,
        ),
    )


def _combined_decision(est_spike, est_mean, x_spike: np.ndarray, x_mean: np.ndarray, spike_weight: float):
    return (
        spike_weight * est_spike.decision_function(x_spike)
        + (1.0 - spike_weight) * est_mean.decision_function(x_mean)
    )


def _predict_from_decision(estimator, decision: np.ndarray) -> np.ndarray:
    return estimator[-1].classes_[np.argmax(decision, axis=1)]


def _run_spike_mean_late_fusion_probe(
    *,
    x_train_spike: np.ndarray,
    x_test_spike: np.ndarray,
    x_train_mean: np.ndarray,
    x_test_mean: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    start = time.time()
    indices = np.arange(len(y_train))
    inner_train_idx, inner_val_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=y_train,
    )
    inner_spike = _rbf_probe_estimator()
    inner_mean = _rbf_probe_estimator()
    inner_spike.fit(x_train_spike[inner_train_idx], y_train[inner_train_idx])
    inner_mean.fit(x_train_mean[inner_train_idx], y_train[inner_train_idx])

    best_weight = 1.0
    best_val_macro_f1 = -1.0
    validation_rows = []
    for weight in np.linspace(0.0, 1.0, 11):
        val_decision = _combined_decision(
            inner_spike,
            inner_mean,
            x_train_spike[inner_val_idx],
            x_train_mean[inner_val_idx],
            float(weight),
        )
        val_pred = _predict_from_decision(inner_spike, val_decision)
        metrics = classification_metrics(y_train[inner_val_idx], val_pred)
        validation_rows.append({"spike_weight": float(weight), "metrics": metrics})
        if metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = metrics["macro_f1"]
            best_weight = float(weight)

    final_spike = _rbf_probe_estimator()
    final_mean = _rbf_probe_estimator()
    final_spike.fit(x_train_spike, y_train)
    final_mean.fit(x_train_mean, y_train)
    train_decision = _combined_decision(
        final_spike,
        final_mean,
        x_train_spike,
        x_train_mean,
        best_weight,
    )
    test_decision = _combined_decision(
        final_spike,
        final_mean,
        x_test_spike,
        x_test_mean,
        best_weight,
    )
    return {
        "name": "late_fusion_rbf_c10_spike_mean",
        "test_metrics": classification_metrics(y_test, _predict_from_decision(final_spike, test_decision)),
        "train_metrics": classification_metrics(y_train, _predict_from_decision(final_spike, train_decision)),
        "seconds": time.time() - start,
        "fusion": {
            "selected_spike_weight": best_weight,
            "validation_macro_f1": best_val_macro_f1,
            "validation_rows": validation_rows,
        },
        "warnings": [],
    }


def run_hybrid_probe(
    config_or_path: dict[str, Any] | str | Path,
    *,
    fold: int,
    checkpoint: str | Path | None = None,
    trainfold_dir: str | Path = DEFAULT_TRAINFOLD_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    checkpoints_dir: str | Path = DEFAULT_CHECKPOINTS_DIR,
    shallow_checkpoint_template: str = "baselines/checkpoints/shallow_original_fold{fold}_seed42.pth",
    device: str = "auto",
    mode: str = "concat",
) -> dict:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else copy.deepcopy(config_or_path)
    training = config.get("training", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    torch_device = resolve_device(device)
    if checkpoint is None:
        checkpoint = hybrid_checkpoint_path(checkpoints_dir, config, fold)
    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_hybrid_model(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch_device)
    loaders = prepare_shallow_feature_loaders(
        fold=fold,
        feature_mode=config.get("model", {}).get("feature_mode", "80d"),
        checkpoint_template=shallow_checkpoint_template,
        trainfold_dir=trainfold_dir,
        batch_size=int(training.get("batch_size", 64)),
        feature_batch_size=int(training.get("feature_batch_size", 256)),
        val_size=float(training.get("val_size", 0.20)),
        seed=seed,
        device=torch_device,
    )
    if mode == "spike_mean_late_fusion":
        x_train_spike, y_train = _extract_embeddings(
            model,
            loaders["full_train_loader"],
            device=torch_device,
            mode="spike_count",
        )
        x_test_spike, y_test = _extract_embeddings(
            model,
            loaders["test_loader"],
            device=torch_device,
            mode="spike_count",
        )
        x_train_mean, y_train_mean = _extract_embeddings(
            model,
            loaders["full_train_loader"],
            device=torch_device,
            mode="mean_membrane",
        )
        x_test_mean, y_test_mean = _extract_embeddings(
            model,
            loaders["test_loader"],
            device=torch_device,
            mode="mean_membrane",
        )
        if not np.array_equal(y_train, y_train_mean) or not np.array_equal(y_test, y_test_mean):
            raise ValueError("Late fusion labels differ between spike and mean embeddings")
        head_results = [
            _run_spike_mean_late_fusion_probe(
                x_train_spike=x_train_spike,
                x_test_spike=x_test_spike,
                x_train_mean=x_train_mean,
                x_test_mean=x_test_mean,
                y_train=y_train,
                y_test=y_test,
                seed=seed,
            )
        ]
        embedding_shape = {
            "train": {
                "spike_count": list(x_train_spike.shape),
                "mean_membrane": list(x_train_mean.shape),
            },
            "test": {
                "spike_count": list(x_test_spike.shape),
                "mean_membrane": list(x_test_mean.shape),
            },
        }
    else:
        x_train, y_train = _extract_embeddings(model, loaders["full_train_loader"], device=torch_device, mode=mode)
        x_test, y_test = _extract_embeddings(model, loaders["test_loader"], device=torch_device, mode=mode)
        embedding_shape = {"train": list(x_train.shape), "test": list(x_test.shape)}
        heads = ["linearsvc_c1", "rbf_svc_c10_gamma_scale", "pca128_rbf_svc_c3_gamma_scale"]
        head_results = []
        for head in heads:
            estimator = _probe_estimator(head, x_train, seed)
            start = time.time()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(x_train, y_train)
            head_results.append(
                {
                    "name": head,
                    "test_metrics": classification_metrics(y_test, estimator.predict(x_test)),
                    "train_metrics": classification_metrics(y_train, estimator.predict(x_train)),
                    "seconds": time.time() - start,
                    "warnings": [
                        {"category": w.category.__name__, "message": str(w.message)}
                        for w in caught
                    ],
                }
            )
    base_name = f"{config_name(config)}_embedding_probe"
    result_name = base_name if mode == "concat" else f"{base_name}_{mode}"
    result = {
        "name": result_name,
        "source_config": config,
        "fold": fold,
        "checkpoint_path": str(checkpoint),
        "embedding_mode": mode,
        "embedding_shape": embedding_shape,
        "heads": head_results,
    }
    out_path = Path(results_dir) / "v2_hybrid" / result_name / f"fold_{fold}.json"
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
