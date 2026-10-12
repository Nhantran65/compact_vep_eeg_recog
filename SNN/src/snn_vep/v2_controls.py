from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

from .constants import DEFAULT_RESULTS_DIR, DEFAULT_TRAINFOLD_DIR, N_CHANNELS, N_TIMES
from .data import encode_labels, load_fold_arrays
from .metrics import classification_metrics
from .utils import ensure_parent


TRANSFORMS = ("raw", "delta", "signed_delta")
HEADS = ("linearsvc_c1", "rbf_svc_c10_gamma_scale", "pca128_rbf_svc_c3_gamma_scale")


def _reshape(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1, N_CHANNELS, N_TIMES).astype(np.float32)


def transform_flat_features(x: np.ndarray, transform: str) -> np.ndarray:
    eeg = _reshape(x)
    if transform == "raw":
        return eeg.reshape(eeg.shape[0], -1)

    diff = np.zeros_like(eeg)
    diff[:, :, 1:] = eeg[:, :, 1:] - eeg[:, :, :-1]
    if transform == "delta":
        return diff.reshape(diff.shape[0], -1)
    if transform == "signed_delta":
        pos = np.maximum(diff, 0.0)
        neg = np.maximum(-diff, 0.0)
        signed = np.concatenate([pos, neg], axis=1)
        return signed.reshape(signed.shape[0], -1)
    raise ValueError(f"Unsupported V2 control transform: {transform}")


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
    raise ValueError(f"Unsupported V2 control head: {name}")


def run_v2_control_fold(
    *,
    transform: str,
    fold: int,
    heads: Iterable[str] = HEADS,
    trainfold_dir: str | Path = DEFAULT_TRAINFOLD_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    seed: int = 42,
) -> dict:
    arrays = load_fold_arrays(trainfold_dir, fold)
    x_train = transform_flat_features(arrays.X_train, transform)
    x_test = transform_flat_features(arrays.X_test, transform)
    y_train = encode_labels(arrays.y_train)
    y_test = encode_labels(arrays.y_test)

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
        "name": f"{transform}_flat_control",
        "transform": transform,
        "fold": fold,
        "feature_shape": {"train": list(x_train.shape), "test": list(x_test.shape)},
        "heads": head_results,
    }

    out_path = Path(results_dir) / "v2_controls" / result["name"] / f"fold_{fold}.json"
    ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result
