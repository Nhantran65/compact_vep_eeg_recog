from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from .configs import config_name, load_config
from .constants import DEFAULT_CHECKPOINTS_DIR, DEFAULT_RESULTS_DIR, DEFAULT_TRAINFOLD_DIR
from .hybrid_models import ShallowFeatureSNNHead
from .hybrid_training import hybrid_checkpoint_path
from .metrics import classification_metrics
from .shallow_features import prepare_shallow_feature_loaders
from .utils import ensure_parent, resolve_device, set_seed


DEFAULT_PREFIX_STEPS = (8, 16, 24, 32, 48, 64, 80, 104)


def coverage_samples(prefix_steps: int, temporal_kernel: int = 25) -> int:
    return int(prefix_steps + temporal_kernel - 1)


def coverage_ms(prefix_steps: int, *, sfreq: float = 128.0, temporal_kernel: int = 25) -> float:
    return float(coverage_samples(prefix_steps, temporal_kernel=temporal_kernel) / sfreq * 1000.0)


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_prefix(
    model: ShallowFeatureSNNHead,
    loader,
    *,
    device: torch.device,
    criterion: nn.Module,
    prefix_steps: int,
    timing_repeats: int = 1,
) -> dict:
    model.eval()
    losses: list[float] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    spike_rates: list[float] = []
    sample_count = 0
    timed_seconds = 0.0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            _sync_if_cuda(device)
            start = time.perf_counter()
            logits = None
            aux = None
            for _ in range(timing_repeats):
                logits, aux = model(features, return_aux=True, prefix_steps=prefix_steps)
            _sync_if_cuda(device)
            elapsed = time.perf_counter() - start
            timed_seconds += elapsed / float(timing_repeats)

            assert logits is not None and aux is not None
            loss = criterion(logits, labels)
            losses.append(float(loss.item()) * labels.size(0))
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
            spike_rates.append(float(aux["spike_rate"].detach().cpu().item()))
            sample_count += int(labels.size(0))

    metrics = classification_metrics(np.array(y_true), np.array(y_pred))
    metrics["loss"] = float(np.sum(losses) / len(y_true))
    metrics["spike_rate"] = float(np.mean(spike_rates))
    metrics["head_inference_ms_per_sample"] = float(timed_seconds / sample_count * 1000.0)
    return metrics


def run_early_exit_fold(
    config_or_path: dict | str | Path,
    *,
    fold: int,
    prefix_steps: Iterable[int] = DEFAULT_PREFIX_STEPS,
    trainfold_dir: str | Path = DEFAULT_TRAINFOLD_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    checkpoints_dir: str | Path = DEFAULT_CHECKPOINTS_DIR,
    shallow_checkpoint_template: str = "baselines/checkpoints/shallow_original_fold{fold}_seed42.pth",
    device: str = "auto",
    batch_size: int | None = None,
    feature_batch_size: int | None = None,
    timing_repeats: int = 1,
) -> dict:
    config = load_config(config_or_path) if isinstance(config_or_path, (str, Path)) else dict(config_or_path)
    training = config.get("training", {})
    model_config = config.get("model", {})
    seed = int(training.get("seed", 42))
    set_seed(seed)
    torch_device = resolve_device(device)
    checkpoint = hybrid_checkpoint_path(checkpoints_dir, config, fold)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained hybrid checkpoint: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu")
    model = ShallowFeatureSNNHead.from_config(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch_device)
    model.eval()

    loaders = prepare_shallow_feature_loaders(
        fold=fold,
        feature_mode=model_config.get("feature_mode", "temporal_map"),
        checkpoint_template=shallow_checkpoint_template,
        trainfold_dir=trainfold_dir,
        batch_size=int(batch_size or training.get("batch_size", 64)),
        feature_batch_size=int(feature_batch_size or training.get("feature_batch_size", 256)),
        val_size=float(training.get("val_size", 0.20)),
        seed=seed,
        device=torch_device,
    )
    feature_shape = loaders["metadata"]["feature_shape"]
    if len(feature_shape) != 2:
        raise ValueError(f"Early-exit requires temporal feature map, got {feature_shape}")
    max_steps = int(feature_shape[1])
    requested_prefixes = [int(p) for p in prefix_steps]
    for prefix in requested_prefixes:
        if prefix < 1 or prefix > max_steps:
            raise ValueError(f"Invalid prefix {prefix}; expected 1..{max_steps}")

    criterion = nn.CrossEntropyLoss()
    prefix_results = []
    for prefix in requested_prefixes:
        metrics = _evaluate_prefix(
            model,
            loaders["test_loader"],
            device=torch_device,
            criterion=criterion,
            prefix_steps=prefix,
            timing_repeats=timing_repeats,
        )
        prefix_results.append(
            {
                "prefix_steps": prefix,
                "coverage_samples": coverage_samples(prefix),
                "coverage_ms": coverage_ms(prefix),
                "metrics": metrics,
            }
        )

    full_latency = next(
        row["metrics"]["head_inference_ms_per_sample"]
        for row in prefix_results
        if row["prefix_steps"] == max_steps
    )
    for row in prefix_results:
        row["relative_latency_vs_104"] = (
            float(row["metrics"]["head_inference_ms_per_sample"] / full_latency)
            if full_latency > 0
            else float("nan")
        )

    result = {
        "name": config_name(config),
        "fold": fold,
        "checkpoint_path": str(checkpoint),
        "data": loaders["metadata"],
        "timing_repeats": timing_repeats,
        "prefix_results": prefix_results,
    }
    out_path = Path(results_dir) / "v3_early_exit" / config_name(config) / f"fold_{fold}.json"
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
