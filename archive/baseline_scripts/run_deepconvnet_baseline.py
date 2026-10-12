#!/usr/bin/env python3
"""Run the DeepConvNet baseline on the fixed EEG folds.

This reproduces the DeepConvNet notebook protocol in script form and exports
fold-level metrics, confusion matrices, and a metric-complete summary including
Cohen's kappa.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
from torch.utils.data import DataLoader, TensorDataset


RANDOM_STATE = 42
CLASS_NAMES = ["Apple", "Car", "Flower", "Human Face"]


class DeepConvNet(nn.Module):
    def __init__(self, nb_classes: int = 4, chans: int = 14, samples: int = 128, dropout_rate: float = 0.5):
        super().__init__()
        self.chans = chans
        self.samples = samples

        self.conv1 = nn.Conv2d(1, 25, (1, 5), bias=True)
        self.conv2 = nn.Conv2d(25, 25, (chans, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(25, eps=1e-5, momentum=0.1)
        self.pool1 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv3 = nn.Conv2d(25, 50, (1, 5), bias=True)
        self.bn2 = nn.BatchNorm2d(50, eps=1e-5, momentum=0.1)
        self.pool2 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv4 = nn.Conv2d(50, 100, (1, 5), bias=True)
        self.bn3 = nn.BatchNorm2d(100, eps=1e-5, momentum=0.1)
        self.pool3 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.dropout3 = nn.Dropout(dropout_rate)

        self.conv5 = nn.Conv2d(100, 200, (1, 5), bias=True)
        self.bn4 = nn.BatchNorm2d(200, eps=1e-5, momentum=0.1)
        self.pool4 = nn.MaxPool2d((1, 2), stride=(1, 2))
        self.dropout4 = nn.Dropout(dropout_rate)

        self.feature_size = self._calculate_feature_size()
        self.fc = nn.Linear(self.feature_size, nb_classes)

    def _calculate_feature_size(self) -> int:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, 1, self.chans, self.samples)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pool1(F.elu(self.bn1(x)))
            x = self.pool2(F.elu(self.bn2(self.conv3(x))))
            x = self.pool3(F.elu(self.bn3(self.conv4(x))))
            x = self.pool4(F.elu(self.bn4(self.conv5(x))))
            size = x.numel()
        self.train(was_training)
        return size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(self.pool1(F.elu(self.bn1(x))))
        x = self.dropout2(self.pool2(F.elu(self.bn2(self.conv3(x)))))
        x = self.dropout3(self.pool3(F.elu(self.bn3(self.conv4(x)))))
        x = self.dropout4(self.pool4(F.elu(self.bn4(self.conv5(x)))))
        return self.fc(x.reshape(x.size(0), -1))


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_fold(fold_dir: Path, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = pd.read_csv(fold_dir / f"fold_{fold}" / "train.csv")
    test_df = pd.read_csv(fold_dir / f"fold_{fold}" / "test.csv")

    x_train = train_df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
    y_train = train_df["Class"].to_numpy()
    x_test = test_df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
    y_test = test_df["Class"].to_numpy()
    return x_train, y_train, x_test, y_test


def prepare_data(
    x_train: np.ndarray,
    y_train_raw: np.ndarray,
    x_test: np.ndarray,
    y_test_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    x_train_prepared = x_train_scaled.reshape(-1, 1, 14, 128).astype(np.float32)
    x_test_prepared = x_test_scaled.reshape(-1, 1, 14, 128).astype(np.float32)
    return x_train_prepared, y_train, x_test_prepared, y_test


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_pred: list[int] = []
    y_true: list[int] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
    return total_loss / len(y_true), np.array(y_pred), np.array(y_true)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }


def summarize(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "weighted_precision",
        "macro_recall",
        "weighted_recall",
        "balanced_accuracy",
        "cohen_kappa",
    ]
    grouped = fold_df.groupby("model", sort=False)[metric_cols]
    means = grouped.mean()
    stds = grouped.std(ddof=1)
    summary = pd.concat({"mean": means, "std": stds}, axis=1)
    summary = summary.swaplevel(axis=1).sort_index(axis=1, level=0)
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepConvNet on fixed EEG folds.")
    parser.add_argument("--fold-dir", type=Path, default=Path("trainfold"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/deepconvnet"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    fold_rows: list[dict[str, float | int | str]] = []
    confusion_rows: list[dict[str, int | str]] = []

    for fold in range(1, 6):
        x_train, y_train_raw, x_test, y_test_raw = load_fold(args.fold_dir, fold)
        x_train_prep, y_train, x_test_prep, y_test = prepare_data(
            x_train, y_train_raw, x_test, y_test_raw
        )

        train_dataset = TensorDataset(
            torch.from_numpy(x_train_prep), torch.from_numpy(y_train).long()
        )
        test_dataset = TensorDataset(
            torch.from_numpy(x_test_prep), torch.from_numpy(y_test).long()
        )
        generator = torch.Generator()
        generator.manual_seed(args.seed + fold)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            generator=generator,
            pin_memory=device.type == "cuda",
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        model = DeepConvNet(dropout_rate=args.dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        early_stopping = EarlyStopping(patience=args.patience)

        best_epoch = 0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, y_pred_epoch, y_true_epoch = evaluate(model, test_loader, criterion, device)
            scheduler.step(test_loss)
            if early_stopping.best_loss is None or test_loss < early_stopping.best_loss:
                best_epoch = epoch
            should_stop = early_stopping.step(test_loss, model)

            if epoch == 1 or epoch % 10 == 0 or should_stop:
                acc = accuracy_score(y_true_epoch, y_pred_epoch)
                print(
                    f"fold={fold} epoch={epoch} train_loss={train_loss:.4f} "
                    f"train_acc={train_acc:.4f} test_loss={test_loss:.4f} "
                    f"test_acc={acc:.4f}",
                    flush=True,
                )
            if should_stop:
                break

        if early_stopping.best_model_state is not None:
            model.load_state_dict(early_stopping.best_model_state)

        test_loss, y_pred, y_true = evaluate(model, test_loader, criterion, device)
        metrics = metric_row(y_true, y_pred)
        fold_rows.append(
            {
                "model": "DeepConvNet",
                "fold": fold,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "best_epoch": best_epoch,
                "test_loss": test_loss,
                **metrics,
            }
        )

        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASS_NAMES)))
        for i, true_label in enumerate(CLASS_NAMES):
            for j, pred_label in enumerate(CLASS_NAMES):
                confusion_rows.append(
                    {
                        "model": "DeepConvNet",
                        "fold": fold,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(cm[i, j]),
                    }
                )

        print(
            f"fold={fold} final acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"kappa={metrics['cohen_kappa']:.4f}",
            flush=True,
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize(fold_df)
    confusion_df = pd.DataFrame(confusion_rows)

    fold_df.to_csv(args.output_dir / "deepconvnet_fold_details.csv", index=False)
    summary_df.to_csv(args.output_dir / "deepconvnet_results_summary.csv", index=False)
    confusion_df.to_csv(args.output_dir / "deepconvnet_confusion_by_fold.csv", index=False)

    metadata = {
        "fold_dir": str(args.fold_dir),
        "folds": 5,
        "seed": args.seed,
        "std_ddof": 1,
        "preprocessing": "StandardScaler fit on train fold only",
        "architecture": "DeepConvNet notebook architecture",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (args.output_dir / "deepconvnet_run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    cols = [
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
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
