from __future__ import annotations

import torch


ENCODERS = {"direct", "rate", "delta", "signed_delta", "adaptive_signed_delta"}


def encoded_channel_count(input_channels: int, encoder_name: str) -> int:
    if encoder_name in {"signed_delta", "adaptive_signed_delta"}:
        return input_channels * 2
    if encoder_name in ENCODERS:
        return input_channels
    raise ValueError(f"Unsupported encoder: {encoder_name}")


def encode_batch(
    x: torch.Tensor,
    encoder_name: str,
    *,
    rate_gain: float = 1.0,
    delta_threshold: float = 0.20,
    training: bool = False,
) -> torch.Tensor:
    """Encode [B,C,T] EEG into [B,C',T] current or spike-like input."""
    if x.ndim != 3:
        raise ValueError(f"Expected [B,C,T] EEG tensor, got shape {tuple(x.shape)}")

    if encoder_name == "direct":
        return x

    if encoder_name == "rate":
        probs = torch.sigmoid(rate_gain * x)
        if training:
            return torch.bernoulli(probs)
        return probs

    diff = torch.zeros_like(x)
    diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]

    if encoder_name == "delta":
        return (diff.abs() >= delta_threshold).to(x.dtype)

    if encoder_name == "signed_delta":
        pos = (diff >= delta_threshold).to(x.dtype)
        neg = (diff <= -delta_threshold).to(x.dtype)
        return torch.cat([pos, neg], dim=1)

    if encoder_name == "adaptive_signed_delta":
        raise ValueError(
            "adaptive_signed_delta is a trainable model-level encoder. "
            "Use SpikingShallowConvNet instead of encode_batch directly."
        )

    raise ValueError(f"Unsupported encoder: {encoder_name}")


def spike_density(encoded: torch.Tensor, encoder_name: str) -> float:
    if encoder_name == "direct":
        return float("nan")
    return float((encoded > 0).to(torch.float32).mean().item())
