from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .configs import config_name, load_config
from .constants import (
    DEFAULT_CHECKPOINTS_DIR,
    DEFAULT_EMBEDDINGS_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TRAINFOLD_DIR,
)
from .data import prepare_fold_loaders
from .metrics import classification_metrics
from .models import build_model
from .utils import ensure_parent, resolve_device, set_seed


def _extract_embeddings(model, loader, *, device: torch.device, mode: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            emb = model.extract_embedding(inputs.to(device), mode=mode)
            xs.append(emb.detach().cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _estimator(name: str, x_train: np.ndarray, seed: int):
    dual = x_train.shape[1] > x_train.shape[0]
    if name == "linearsvc_c1":
        return {
            "family": "linear_margin",
            "config": {"C": 1.0, "dual": dual, "max_iter": 5000, "tol": 1e-3},
            "estimator": make_pipeline(
                StandardScaler(),
                LinearSVC(C=1.0, dual=dual, max_iter=5000, tol=1e-3),
            ),
        }
    if name == "rbf_svc_c10_gamma_scale":
        return {
            "family": "rbf_svc",
            "config": {"C": 10.0, "kernel": "rbf", "gamma": "scale"},
            "estimator": make_pipeline(
                StandardScaler(),
                SVC(C=10.0, kernel="rbf", gamma="scale", cache_size=1000),
            ),
        }
    if name == "pca128_rbf_svc_c3_gamma_scale":
        n_components = min(128, x_train.shape[1], x_train.shape[0] - 1)
        return {
            "family": "pca_rbf_svc",
            "config": {
                "pca_components": int(n_components),
                "requested_pca_components": 128,
                "C": 3.0,
                "kernel": "rbf",
                "gamma": "scale",
            },
            "estimator": make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components, random_state=seed),
                SVC(C=3.0, kernel="rbf", gamma="scale", cache_size=1000),
            ),
        }
    raise ValueError(f"Unsupported probe head: {name}")


def run_embedding_probe(
    probe_config_or_path: dict[str, Any] | str | Path,
    *,
    fold: int,
    checkpoint: str | Path | None = None,
    trainfold_dir: str | Path = DEFAULT_TRAINFOLD_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    checkpoints_dir: str | Path = DEFAULT_CHECKPOINTS_DIR,
    embeddings_dir: str | Path = DEFAULT_EMBEDDINGS_DIR,
    device: str = "auto",
) -> dict:
    probe_config = (
        load_config(probe_config_or_path)
        if isinstance(probe_config_or_path, (str, Path))
        else dict(probe_config_or_path)
    )
    source_config = load_config(probe_config["source_config"])
    source_name = config_name(source_config)
    seed = int(source_config.get("training", {}).get("seed", 42))
    set_seed(seed)
    torch_device = resolve_device(device)

    if checkpoint is None:
        checkpoint = Path(checkpoints_dir) / source_name / f"fold_{fold}.pt"
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing SNN checkpoint: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu")
    model = build_model(source_config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(torch_device)

    training = source_config.get("training", {})
    loaders = prepare_fold_loaders(
        trainfold_dir,
        fold,
        batch_size=int(training.get("batch_size", 64)),
        val_size=float(training.get("val_size", 0.20)),
        seed=seed,
    )
    mode = probe_config.get("embedding", {}).get("mode", "concat")
    x_train, y_train = _extract_embeddings(
        model,
        loaders.full_train_loader,
        device=torch_device,
        mode=mode,
    )
    x_test, y_test = _extract_embeddings(
        model,
        loaders.test_loader,
        device=torch_device,
        mode=mode,
    )

    emb_path = Path(embeddings_dir) / source_name / f"fold_{fold}_{mode}.npz"
    ensure_parent(emb_path)
    np.savez_compressed(
        emb_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    heads = probe_config.get(
        "heads",
        ["linearsvc_c1", "rbf_svc_c10_gamma_scale", "pca128_rbf_svc_c3_gamma_scale"],
    )
    head_results = []
    for head in heads:
        spec = _estimator(head, x_train, seed)
        start = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            spec["estimator"].fit(x_train, y_train)
        seconds = time.time() - start
        head_results.append(
            {
                "name": head,
                "family": spec["family"],
                "config": spec["config"],
                "train_metrics": classification_metrics(y_train, spec["estimator"].predict(x_train)),
                "test_metrics": classification_metrics(y_test, spec["estimator"].predict(x_test)),
                "seconds": seconds,
                "warnings": [
                    {"category": w.category.__name__, "message": str(w.message)}
                    for w in caught
                ],
            }
        )

    result = {
        "name": config_name(probe_config),
        "fold": fold,
        "source_config": source_config,
        "checkpoint_path": str(checkpoint),
        "embedding_mode": mode,
        "embedding_path": str(emb_path),
        "embedding_shape": {
            "train": list(x_train.shape),
            "test": list(x_test.shape),
        },
        "heads": head_results,
    }

    out_path = Path(results_dir) / "by_config" / config_name(probe_config) / f"fold_{fold}.json"
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
