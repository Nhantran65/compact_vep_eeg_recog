#!/usr/bin/env python3
"""Run classical ML baselines on the fixed EEG train/test folds.

The script mirrors the classical baseline notebooks, but writes a compact,
paper-ready set of CSV files from the current `trainfold/fold_*` split files.
All preprocessing is fit on each training fold only.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - dependency is optional at import time.
    lgb = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - dependency is optional at import time.
    xgb = None


RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classical ML baselines on fixed EEG folds."
    )
    parser.add_argument(
        "--fold-dir",
        type=Path,
        default=Path("trainfold"),
        help="Directory containing fold_1 ... fold_5 train/test CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/classical_ml"),
        help="Directory for fold details, summary tables, and metadata.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of fixed folds to evaluate.",
    )
    return parser.parse_args()


def load_fold(fold_dir: Path, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_path = fold_dir / f"fold_{fold}" / "train.csv"
    test_path = fold_dir / f"fold_{fold}" / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if "Class" not in train_df.columns or "Class" not in test_df.columns:
        raise ValueError(f"Missing Class column in fold {fold}")

    x_train = train_df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
    y_train = train_df["Class"].to_numpy()
    x_test = test_df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
    y_test = test_df["Class"].to_numpy()
    return x_train, y_train, x_test, y_test


def build_models() -> dict[str, Callable[[], object]]:
    models: dict[str, Callable[[], object]] = {
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM (RBF)": lambda: SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            random_state=RANDOM_STATE,
        ),
        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": lambda: DecisionTreeClassifier(
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": lambda: LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
    }

    if xgb is not None:
        models["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    if lgb is not None:
        models["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def summarize(fold_rows: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
        "balanced_accuracy",
        "cohen_kappa",
    ]
    grouped = fold_rows.groupby("model", sort=False)[metric_cols]
    means = grouped.mean()
    stds = grouped.std(ddof=0)
    summary = pd.concat({"mean": means, "std": stds}, axis=1)
    summary = summary.swaplevel(axis=1).sort_index(axis=1, level=0)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
        category=UserWarning,
    )

    models = build_models()
    fold_rows: list[dict[str, float | int | str]] = []
    confusion_rows: list[dict[str, int | str]] = []

    for fold in range(1, args.folds + 1):
        x_train, y_train_raw, x_test, y_test_raw = load_fold(args.fold_dir, fold)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)
        labels = np.arange(len(label_encoder.classes_))

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        for model_name, model_factory in models.items():
            model = model_factory()
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            metrics = compute_metrics(y_test, y_pred)

            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    **metrics,
                }
            )

            cm = confusion_matrix(y_test, y_pred, labels=labels)
            for true_idx, true_label in enumerate(label_encoder.classes_):
                for pred_idx, pred_label in enumerate(label_encoder.classes_):
                    confusion_rows.append(
                        {
                            "model": model_name,
                            "fold": fold,
                            "true_label": int(true_label),
                            "pred_label": int(pred_label),
                            "count": int(cm[true_idx, pred_idx]),
                        }
                    )

            print(
                f"fold={fold} model={model_name} "
                f"acc={metrics['accuracy']:.4f} "
                f"macro_f1={metrics['macro_f1']:.4f} "
                f"bal_acc={metrics['balanced_accuracy']:.4f}",
                flush=True,
            )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize(fold_df)
    confusion_df = pd.DataFrame(confusion_rows)

    fold_df.to_csv(args.output_dir / "classical_ml_fold_details.csv", index=False)
    summary_df.to_csv(args.output_dir / "classical_ml_results_summary.csv", index=False)
    confusion_df.to_csv(args.output_dir / "classical_ml_confusion_by_fold.csv", index=False)

    metadata = {
        "fold_dir": str(args.fold_dir),
        "folds": args.folds,
        "random_state": RANDOM_STATE,
        "preprocessing": "StandardScaler fit on train fold only",
        "std_ddof": 0,
        "sklearn_version": sklearn.__version__,
        "xgboost_version": getattr(xgb, "__version__", None) if xgb is not None else None,
        "lightgbm_version": getattr(lgb, "__version__", None) if lgb is not None else None,
        "models": list(models.keys()),
    }
    (args.output_dir / "classical_ml_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    display_cols = [
        "model",
        "accuracy_mean",
        "accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "cohen_kappa_mean",
        "cohen_kappa_std",
    ]
    print("\nSummary")
    print(summary_df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
