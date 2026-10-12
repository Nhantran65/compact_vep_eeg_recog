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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .configs import config_name, load_config
from .constants import DEFAULT_CHECKPOINTS_DIR, DEFAULT_RESULTS_DIR, N_CLASSES
from .hybrid_models import DifferentiableRBFPrototypeHead
from .metrics import classification_metrics
from .models import make_surrogate
from .utils import ensure_parent, resolve_device, set_seed


def eegpt_token_snn_checkpoint_path(
    checkpoints_dir: str | Path,
    config: dict[str, Any],
    fold: int,
) -> Path:
    return Path(checkpoints_dir) / "fm_token_snn" / config_name(config) / f"fold_{fold}.pt"


def eegpt_token_snn_result_path(
    results_dir: str | Path,
    config: dict[str, Any],
    fold: int,
) -> Path:
    return Path(results_dir) / "fm_token_snn" / config_name(config) / f"fold_{fold}.json"


class EEGPTTokenSpikeFeatureExtractor(nn.Module):
    """LIF feature extractor over frozen EEGPT token sequences [B,T,D]."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int = 64,
        beta: float = 0.90,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        surrogate: str = "atan",
        gate_kernel: int = 3,
    ) -> None:
        super().__init__()
        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError(
                "snnTorch is required for EEGPT token SNN heads. Install with: "
                "uv pip install -r SNN/requirements.txt"
            ) from exc

        if gate_kernel < 1 or gate_kernel % 2 == 0:
            raise ValueError("gate_kernel must be a positive odd integer")
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=make_surrogate(surrogate),
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )
        self.gate = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=gate_kernel,
            padding=gate_kernel // 2,
            groups=hidden_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"Expected EEGPT tokens [B,T,D], got {tuple(tokens.shape)}")
        if tokens.shape[2] != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {tokens.shape[2]}")
        batch_size, n_steps, _ = tokens.shape
        mem = torch.zeros(batch_size, self.hidden_dim, device=tokens.device, dtype=tokens.dtype)
        spike_sum = torch.zeros_like(mem)
        mem_steps = []
        spike_steps = []

        for step in range(n_steps):
            current = self.input_proj(tokens[:, step, :])
            spk, mem = self.lif(current, mem)
            spike_sum = spike_sum + spk
            mem_steps.append(mem)
            spike_steps.append(spk)

        mem_seq = torch.stack(mem_steps, dim=2)
        spike_seq = torch.stack(spike_steps, dim=2)
        gate = torch.sigmoid(self.gate(mem_seq))
        gated_mean = (mem_seq * gate).sum(dim=2) / gate.sum(dim=2).clamp_min(1.0e-6)
        spike_count = spike_sum / float(n_steps)
        fused = self.fusion(torch.cat([gated_mean, spike_count], dim=1))
        return {
            "spike_count": spike_count,
            "gated_mean_membrane": gated_mean,
            "fused_embedding": fused,
            "spike_sequence": spike_seq,
            "mean_gate": gate.mean(),
            "spike_rate": spike_seq.mean(),
        }


class EEGPTTokenSpikeAwareGatedLIF(nn.Module):
    """Spike-aware gated LIF over frozen EEGPT tokens with a linear head."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int = 64,
        beta: float = 0.90,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        surrogate: str = "atan",
        dropout: float = 0.30,
        gate_kernel: int = 3,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        self.features = EEGPTTokenSpikeFeatureExtractor(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            threshold=threshold,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            surrogate=surrogate,
            gate_kernel=gate_kernel,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, tokens: torch.Tensor, return_aux: bool = False):
        features = self.features(tokens)
        logits = self.classifier(self.dropout(features["fused_embedding"]))
        if return_aux:
            return logits, {
                "embedding": features["fused_embedding"],
                "spike_rate": features["spike_rate"],
                "mean_gate": features["mean_gate"],
            }
        return logits


class EEGPTTokenSpikeAwareProtoLIF(nn.Module):
    """Spike-aware gated LIF over frozen EEGPT tokens with differentiable RBF prototypes."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int = 64,
        beta: float = 0.90,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        surrogate: str = "atan",
        dropout: float = 0.30,
        gate_kernel: int = 3,
        prototypes_per_class: int = 4,
        init_gamma: float = 8.0,
        learn_gamma: bool = True,
        learn_prototype_bias: bool = True,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        self.features = EEGPTTokenSpikeFeatureExtractor(
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            beta=beta,
            threshold=threshold,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            surrogate=surrogate,
            gate_kernel=gate_kernel,
        )
        self.dropout = nn.Dropout(dropout)
        self.prototype_head = DifferentiableRBFPrototypeHead(
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            prototypes_per_class=prototypes_per_class,
            init_gamma=init_gamma,
            learn_gamma=learn_gamma,
            learn_bias=learn_prototype_bias,
        )

    def forward(self, tokens: torch.Tensor, return_aux: bool = False):
        features = self.features(tokens)
        embedding = features["fused_embedding"]
        logits = self.prototype_head(self.dropout(embedding))
        if return_aux:
            return logits, {
                "embedding": embedding,
                "spike_rate": features["spike_rate"],
                "mean_gate": features["mean_gate"],
                "gamma": self.prototype_head.gamma.detach(),
            }
        return logits


def build_eegpt_token_snn_model(
    config: dict[str, Any],
    *,
    token_dim: int,
) -> nn.Module:
    model = config.get("model", {})
    common = {
        "token_dim": token_dim,
        "hidden_dim": int(model.get("hidden_dim", 64)),
        "beta": float(model.get("beta", 0.90)),
        "threshold": float(model.get("threshold", 1.0)),
        "learn_beta": bool(model.get("learn_beta", False)),
        "learn_threshold": bool(model.get("learn_threshold", False)),
        "surrogate": model.get("surrogate", "atan"),
        "dropout": float(model.get("dropout", 0.30)),
        "gate_kernel": int(model.get("gate_kernel", 3)),
        "n_classes": int(model.get("n_classes", N_CLASSES)),
    }
    architecture = model.get("architecture", "eegpt_token_spike_aware_gated_lif")
    if architecture == "eegpt_token_spike_aware_gated_lif":
        return EEGPTTokenSpikeAwareGatedLIF(**common)
    if architecture == "eegpt_token_spike_aware_proto_lif":
        return EEGPTTokenSpikeAwareProtoLIF(
            **common,
            prototypes_per_class=int(model.get("prototypes_per_class", 4)),
            init_gamma=float(model.get("init_gamma", 8.0)),
            learn_gamma=bool(model.get("learn_gamma", True)),
            learn_prototype_bias=bool(model.get("learn_prototype_bias", True)),
        )
    raise ValueError(f"Unsupported EEGPT token SNN architecture: {architecture}")


def _load_eegpt_token_cache(
    *,
    fold: int,
    embedding_template: str,
) -> dict[str, Any]:
    path = Path(embedding_template.format(fold=fold))
    if not path.exists():
        raise FileNotFoundError(f"Missing EEGPT embedding cache: {path}")
    payload = torch.load(path, map_location="cpu")
    train_tokens = payload["train_tokens"].float().numpy().astype(np.float32)
    test_tokens = payload["test_tokens"].float().numpy().astype(np.float32)
    train_labels = payload["train_labels"].long().numpy().astype(np.int64)
    test_labels = payload["test_labels"].long().numpy().astype(np.int64)
    if train_tokens.ndim != 3 or test_tokens.ndim != 3:
        raise ValueError(
            f"Expected token tensors [N,T,D], got train={train_tokens.shape}, test={test_tokens.shape}"
        )
    if train_tokens.shape[1:] != test_tokens.shape[1:]:
        raise ValueError(
            f"Train/test token shapes differ: train={train_tokens.shape}, test={test_tokens.shape}"
        )
    return {
        "path": path,
        "train_tokens": train_tokens,
        "test_tokens": test_tokens,
        "train_labels": train_labels,
        "test_labels": test_labels,
        "metadata": payload.get("metadata", {}),
    }


def _standardize_tokens(
    train_tokens: np.ndarray,
    test_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = train_tokens.mean(axis=(0, 1), keepdims=True)
    std = train_tokens.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1.0e-6)
    train_std = ((train_tokens - mean) / std).astype(np.float32)
    test_std = ((test_tokens - mean) / std).astype(np.float32)
    return train_std, test_std, {
        "enabled": True,
        "fit": "full_train_fold",
        "mean_shape": list(mean.shape),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
    }


def prepare_eegpt_token_loaders(
    *,
    fold: int,
    config: dict[str, Any],
    batch_size: int,
    val_size: float,
    seed: int,
) -> dict[str, Any]:
    data_config = config.get("data", {})
    cache = _load_eegpt_token_cache(
        fold=fold,
        embedding_template=data_config.get(
            "embedding_template",
            "FoundationModel_EEGPT/results/embeddings/fold_{fold}_eegpt_1s.pt",
        ),
    )
    train_tokens = cache["train_tokens"]
    test_tokens = cache["test_tokens"]
    standardize = bool(data_config.get("standardize_tokens", True))
    if standardize:
        train_tokens, test_tokens, standardizer = _standardize_tokens(train_tokens, test_tokens)
    else:
        standardizer = {"enabled": False}

    y_train = cache["train_labels"]
    y_test = cache["test_labels"]
    indices = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=y_train,
    )

    def make_loader(tokens: np.ndarray, labels: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(
            TensorDataset(torch.from_numpy(tokens), torch.from_numpy(labels).long()),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return {
        "train_loader": make_loader(train_tokens[train_idx], y_train[train_idx], True),
        "val_loader": make_loader(train_tokens[val_idx], y_train[val_idx], False),
        "full_train_loader": make_loader(train_tokens, y_train, False),
        "test_loader": make_loader(test_tokens, y_test, False),
        "metadata": {
            "fold": fold,
            "feature_source": "eegpt_adapter58_cached_tokens",
            "embedding_path": str(cache["path"]),
            "train_size": int(len(train_idx)),
            "val_size_count": int(len(val_idx)),
            "full_train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "token_shape": list(train_tokens.shape[1:]),
            "standardizer": standardizer,
            "cache_metadata": cache["metadata"],
        },
    }


def _loop_batches(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    y_true = []
    y_pred = []
    aux_values: dict[str, list[float]] = {"spike_rate": [], "mean_gate": [], "gamma": []}
    for batch_idx, (tokens, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        tokens = tokens.to(device)
        labels = labels.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits, aux = model(tokens, return_aux=True)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        loss_sum += float(loss.detach().cpu().item()) * labels.size(0)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        for key in aux_values:
            if key in aux:
                aux_values[key].append(float(aux[key].detach().cpu().item()))

    metrics = classification_metrics(np.array(y_true), np.array(y_pred))
    denom = max(len(y_true), 1)
    metrics["loss"] = float(loss_sum / denom)
    for key, values in aux_values.items():
        if values:
            metrics[key] = float(np.mean(values))
    return metrics


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module,
) -> dict[str, Any]:
    with torch.no_grad():
        return _loop_batches(model, loader, device=device, criterion=criterion)


def train_eegpt_token_snn_fold(
    config_or_path: dict[str, Any] | str | Path,
    *,
    fold: int,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    checkpoints_dir: str | Path = DEFAULT_CHECKPOINTS_DIR,
    device: str = "auto",
    progress: bool = True,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else copy.deepcopy(config_or_path)
    training = config.get("training", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    torch_device = resolve_device(device)
    batch_size = int(batch_size_override or training.get("batch_size", 64))
    epochs = int(epochs_override or training.get("epochs", 120))
    patience = int(training.get("patience", 20))

    loaders = prepare_eegpt_token_loaders(
        fold=fold,
        config=config,
        batch_size=batch_size,
        val_size=float(training.get("val_size", 0.20)),
        seed=seed,
    )
    token_shape = loaders["metadata"]["token_shape"]
    model = build_eegpt_token_snn_model(config, token_dim=int(token_shape[1])).to(torch_device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config.get("loss", {}).get("label_smoothing", 0.0))
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
            optimizer=optimizer,
            max_batches=max_train_batches,
        )
        val_loader = loaders["val_loader"]
        if max_eval_batches is None:
            val_metrics = _evaluate(model, val_loader, device=torch_device, criterion=criterion)
        else:
            val_metrics = _loop_batches(
                model,
                val_loader,
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
            iterator.set_postfix(val_f1=f"{val_metrics['macro_f1']:.3f}", best=best_epoch)
        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    eval_train_loader = loaders["full_train_loader"]
    eval_test_loader = loaders["test_loader"]
    if max_eval_batches is None:
        train_metrics = _evaluate(model, eval_train_loader, device=torch_device, criterion=criterion)
        test_metrics = _evaluate(model, eval_test_loader, device=torch_device, criterion=criterion)
    else:
        train_metrics = _loop_batches(
            model,
            eval_train_loader,
            device=torch_device,
            criterion=criterion,
            max_batches=max_eval_batches,
        )
        test_metrics = _loop_batches(
            model,
            eval_test_loader,
            device=torch_device,
            criterion=criterion,
            max_batches=max_eval_batches,
        )
    seconds = time.time() - start

    ckpt_path = eegpt_token_snn_checkpoint_path(checkpoints_dir, config, fold)
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
        "train_minutes": seconds / 60.0,
        "checkpoint_path": str(ckpt_path),
        "history": history,
    }
    out_path = eegpt_token_snn_result_path(results_dir, config, fold)
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
