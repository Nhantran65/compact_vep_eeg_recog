from __future__ import annotations

import torch


def _windowed_mean(sequence: torch.Tensor, n_bins: int) -> torch.Tensor:
    if sequence.ndim != 3:
        raise ValueError(f"Expected [B,H,T] sequence, got {tuple(sequence.shape)}")
    if n_bins < 1 or n_bins > sequence.shape[2]:
        raise ValueError(f"n_bins must be in [1,{sequence.shape[2]}], got {n_bins}")
    chunks = torch.chunk(sequence, n_bins, dim=2)
    return torch.cat([chunk.mean(dim=2) for chunk in chunks], dim=1)


def select_readout(features: dict[str, torch.Tensor], readout: str) -> torch.Tensor:
    if readout == "spike_count":
        return features["spike_count"]
    if readout == "mean_membrane":
        return features["mean_membrane"]
    if readout == "final_membrane":
        return features["final_membrane"]
    if readout == "max_membrane":
        return features["max_membrane"]
    raise ValueError(f"Unsupported readout: {readout}")


def select_embedding(features: dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "spike_count":
        return features["spike_count"]
    if mode == "mean_membrane":
        return features["mean_membrane"]
    if mode == "final_membrane":
        return features["final_membrane"]
    if mode == "max_membrane":
        return features["max_membrane"]
    if mode == "spike_mean_concat":
        return torch.cat([features["spike_count"], features["mean_membrane"]], dim=1)
    if mode.startswith("windowed_spike_count_"):
        n_bins = int(mode.rsplit("_", 1)[-1])
        return _windowed_mean(features["spike_sequence"], n_bins)
    if mode == "concat":
        return torch.cat(
            [
                features["spike_count"],
                features["mean_membrane"],
                features["final_membrane"],
                features["max_membrane"],
            ],
            dim=1,
        )
    raise ValueError(f"Unsupported embedding mode: {mode}")
