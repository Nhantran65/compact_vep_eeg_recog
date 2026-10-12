from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .constants import N_CHANNELS, N_TIMES


@dataclass
class FoldArrays:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class FoldTensors:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_full_train: torch.Tensor
    y_full_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    scaler: StandardScaler
    metadata: dict


@dataclass
class FoldLoaders:
    train_loader: DataLoader
    val_loader: DataLoader
    full_train_loader: DataLoader
    test_loader: DataLoader
    tensors: FoldTensors


def load_fold_arrays(trainfold_dir: str | Path, fold: int) -> FoldArrays:
    fold_dir = Path(trainfold_dir) / f"fold_{fold}"
    train_path = fold_dir / "train.csv"
    test_path = fold_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing train/test CSV files in {fold_dir}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train = train_df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_train = train_df.iloc[:, -1].to_numpy()
    X_test = test_df.iloc[:, :-1].to_numpy(dtype=np.float32)
    y_test = test_df.iloc[:, -1].to_numpy()
    return FoldArrays(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)


def encode_labels(y: np.ndarray) -> np.ndarray:
    y_int = y.astype(np.int64)
    unique = set(np.unique(y_int).tolist())
    if unique.issubset({1, 2, 3, 4}):
        return y_int - 1
    if unique.issubset({0, 1, 2, 3}):
        return y_int
    raise ValueError(f"Unsupported labels: {sorted(unique)}")


def _reshape_eeg(x: np.ndarray) -> np.ndarray:
    expected = N_CHANNELS * N_TIMES
    if x.shape[1] != expected:
        raise ValueError(f"Expected {expected} EEG features, got {x.shape[1]}")
    return x.reshape(-1, N_CHANNELS, N_TIMES).astype(np.float32)


def prepare_fold_tensors(
    trainfold_dir: str | Path,
    fold: int,
    val_size: float = 0.20,
    seed: int = 42,
) -> FoldTensors:
    arrays = load_fold_arrays(trainfold_dir, fold)
    y_train_all = encode_labels(arrays.y_train)
    y_test = encode_labels(arrays.y_test)

    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        arrays.X_train,
        y_train_all,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_all,
    )

    scaler = StandardScaler()
    X_subtrain = scaler.fit_transform(X_subtrain).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_full_train = scaler.transform(arrays.X_train).astype(np.float32)
    X_test = scaler.transform(arrays.X_test).astype(np.float32)

    metadata = {
        "fold": fold,
        "val_size": val_size,
        "seed": seed,
        "scaler_fit": "stratified_train_split",
        "train_size": int(len(X_subtrain)),
        "val_size_count": int(len(X_val)),
        "full_train_size": int(len(X_full_train)),
        "test_size": int(len(X_test)),
    }

    return FoldTensors(
        X_train=torch.from_numpy(_reshape_eeg(X_subtrain)),
        y_train=torch.from_numpy(y_subtrain).long(),
        X_val=torch.from_numpy(_reshape_eeg(X_val)),
        y_val=torch.from_numpy(y_val).long(),
        X_full_train=torch.from_numpy(_reshape_eeg(X_full_train)),
        y_full_train=torch.from_numpy(y_train_all).long(),
        X_test=torch.from_numpy(_reshape_eeg(X_test)),
        y_test=torch.from_numpy(y_test).long(),
        scaler=scaler,
        metadata=metadata,
    )


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def prepare_fold_loaders(
    trainfold_dir: str | Path,
    fold: int,
    batch_size: int = 64,
    val_size: float = 0.20,
    seed: int = 42,
    num_workers: int = 0,
) -> FoldLoaders:
    tensors = prepare_fold_tensors(trainfold_dir, fold, val_size=val_size, seed=seed)
    return FoldLoaders(
        train_loader=make_loader(tensors.X_train, tensors.y_train, batch_size, True, num_workers),
        val_loader=make_loader(tensors.X_val, tensors.y_val, batch_size, False, num_workers),
        full_train_loader=make_loader(
            tensors.X_full_train,
            tensors.y_full_train,
            batch_size,
            False,
            num_workers,
        ),
        test_loader=make_loader(tensors.X_test, tensors.y_test, batch_size, False, num_workers),
        tensors=tensors,
    )
