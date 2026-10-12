from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .constants import DEFAULT_TRAINFOLD_DIR, N_CHANNELS, N_TIMES
from .data import encode_labels, load_fold_arrays


from .backbone import ShallowConvNet


class ShallowFeatureExtractor(nn.Module):
    """Frozen fold-specific ShallowConvNet feature extractor."""

    def __init__(self, checkpoint: str | Path, map_location: str | torch.device = "cpu") -> None:
        super().__init__()
        shallow_cls = ShallowConvNet
        self.backbone = shallow_cls(nb_classes=4, Chans=N_CHANNELS, Samples=N_TIMES)
        state = torch.load(checkpoint, map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.extract_80d(x)

    def _as_shallow_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x.unsqueeze(1)
        if x.ndim == 4:
            return x
        raise ValueError(f"Expected [B,14,128] or [B,1,14,128], got {tuple(x.shape)}")

    @torch.no_grad()
    def extract_80d(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        return self.backbone.extract_features(self._as_shallow_input(x), apply_dropout=False)

    @torch.no_grad()
    def extract_temporal_map(self, x: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        x4 = self._as_shallow_input(x)
        z = self.backbone.conv1(x4)
        z = self.backbone.conv2(z)
        z = self.backbone.bn(z)
        z = torch.square(z)
        return z.squeeze(2)


def shallow_checkpoint_path(
    fold: int,
    *,
    template: str = "baselines/checkpoints/shallow_original_fold{fold}_seed42.pth",
) -> Path:
    path = Path(template.format(fold=fold))
    if not path.exists():
        raise FileNotFoundError(f"Missing fold-specific ShallowConvNet checkpoint: {path}")
    return path


def _standardized_fold_arrays(trainfold_dir: str | Path, fold: int):
    arrays = load_fold_arrays(trainfold_dir, fold)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(arrays.X_train).astype(np.float32)
    x_test = scaler.transform(arrays.X_test).astype(np.float32)
    x_train = x_train.reshape(-1, N_CHANNELS, N_TIMES)
    x_test = x_test.reshape(-1, N_CHANNELS, N_TIMES)
    y_train = encode_labels(arrays.y_train)
    y_test = encode_labels(arrays.y_test)
    return x_train, y_train, x_test, y_test, scaler


def _extract_feature_array(
    extractor: ShallowFeatureExtractor,
    x: np.ndarray,
    *,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    extractor.to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    chunks = []
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            if mode == "80d":
                features = extractor.extract_80d(inputs)
            elif mode == "temporal_map":
                features = extractor.extract_temporal_map(inputs)
            else:
                raise ValueError(f"Unsupported Shallow feature mode: {mode}")
            chunks.append(features.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def prepare_shallow_feature_loaders(
    *,
    fold: int,
    feature_mode: str,
    checkpoint_template: str = "baselines/checkpoints/shallow_original_fold{fold}_seed42.pth",
    trainfold_dir: str | Path = DEFAULT_TRAINFOLD_DIR,
    batch_size: int = 64,
    feature_batch_size: int = 256,
    val_size: float = 0.20,
    seed: int = 42,
    device: torch.device | None = None,
):
    device = device or torch.device("cpu")
    x_train_raw, y_train, x_test_raw, y_test, _ = _standardized_fold_arrays(trainfold_dir, fold)
    extractor = ShallowFeatureExtractor(shallow_checkpoint_path(fold, template=checkpoint_template))
    x_train_features = _extract_feature_array(
        extractor,
        x_train_raw,
        mode=feature_mode,
        batch_size=feature_batch_size,
        device=device,
    )
    x_test_features = _extract_feature_array(
        extractor,
        x_test_raw,
        mode=feature_mode,
        batch_size=feature_batch_size,
        device=device,
    )

    indices = np.arange(len(y_train))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=y_train,
    )

    def make_loader(x_arr, y_arr, shuffle: bool):
        return DataLoader(
            TensorDataset(torch.from_numpy(x_arr), torch.from_numpy(y_arr).long()),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    return {
        "train_loader": make_loader(x_train_features[train_idx], y_train[train_idx], True),
        "val_loader": make_loader(x_train_features[val_idx], y_train[val_idx], False),
        "full_train_loader": make_loader(x_train_features, y_train, False),
        "test_loader": make_loader(x_test_features, y_test, False),
        "metadata": {
            "fold": fold,
            "feature_mode": feature_mode,
            "checkpoint_template": checkpoint_template,
            "checkpoint_path": str(shallow_checkpoint_path(fold, template=checkpoint_template)),
            "scaler_fit": "full_train_fold",
            "train_size": int(len(train_idx)),
            "val_size_count": int(len(val_idx)),
            "full_train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "feature_shape": list(x_train_features.shape[1:]),
        },
    }
