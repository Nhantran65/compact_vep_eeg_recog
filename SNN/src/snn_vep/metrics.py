from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from .constants import N_CLASSES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(N_CLASSES))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def mean_metric(rows: list[dict], key: str) -> tuple[float, float]:
    values = np.array([row[key] for row in rows], dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1))
