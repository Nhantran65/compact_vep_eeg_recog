from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .constants import N_CHANNELS, N_CLASSES
from .encoders import encode_batch, encoded_channel_count
from .readouts import select_embedding, select_readout


def make_surrogate(name: str):
    try:
        import snntorch.surrogate as surrogate
    except ImportError as exc:
        raise ImportError(
            "snnTorch is required for SNN models. Install with: "
            "uv pip install -r SNN/requirements.txt"
        ) from exc

    if name == "atan":
        return surrogate.atan()
    if name == "fast_sigmoid":
        return surrogate.fast_sigmoid(slope=25)
    if name == "straight_through":
        return surrogate.straight_through_estimator()
    raise ValueError(f"Unsupported surrogate gradient: {name}")


class SpikingShallowConvNet(nn.Module):
    """ShallowConvNet-like convolutional SNN for [B,14,128] VEP epochs."""

    def __init__(
        self,
        *,
        encoder_name: str = "direct",
        input_channels: int = N_CHANNELS,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        beta: float = 0.90,
        threshold: float = 1.0,
        surrogate: str = "atan",
        dropout: float = 0.50,
        readout: str = "mean_membrane",
        rate_gain: float = 1.0,
        delta_threshold: float = 0.20,
        adaptive_slope: float = 12.0,
        adaptive_threshold_init: float = 0.20,
        adaptive_gain_init: float = 1.0,
        n_classes: int = N_CLASSES,
    ) -> None:
        super().__init__()
        if temporal_kernel % 2 == 0:
            raise ValueError("temporal_kernel must be odd to preserve time length")

        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError(
                "snnTorch is required for SNN models. Install with: "
                "uv pip install -r SNN/requirements.txt"
            ) from exc

        self.encoder_name = encoder_name
        self.raw_input_channels = input_channels
        self.input_channels = encoded_channel_count(input_channels, encoder_name)
        self.n_filters = n_filters
        self.readout = readout
        self.rate_gain = rate_gain
        self.delta_threshold = delta_threshold
        self.adaptive_slope = adaptive_slope

        if encoder_name == "adaptive_signed_delta":
            if adaptive_threshold_init <= 0:
                raise ValueError("adaptive_threshold_init must be positive")
            if adaptive_gain_init <= 0:
                raise ValueError("adaptive_gain_init must be positive")
            self.adaptive_log_gain = nn.Parameter(
                torch.full((input_channels,), float(torch.log(torch.tensor(adaptive_gain_init))))
            )
            self.adaptive_log_pos_threshold = nn.Parameter(
                torch.full((input_channels,), float(torch.log(torch.tensor(adaptive_threshold_init))))
            )
            self.adaptive_log_neg_threshold = nn.Parameter(
                torch.full((input_channels,), float(torch.log(torch.tensor(adaptive_threshold_init))))
            )
        else:
            self.register_parameter("adaptive_log_gain", None)
            self.register_parameter("adaptive_log_pos_threshold", None)
            self.register_parameter("adaptive_log_neg_threshold", None)

        padding = temporal_kernel // 2
        self.temporal_conv = nn.Conv2d(
            1,
            n_filters,
            kernel_size=(1, temporal_kernel),
            padding=(0, padding),
            bias=False,
        )
        self.spatial_conv = nn.Conv2d(
            n_filters,
            n_filters,
            kernel_size=(self.input_channels, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm1d(n_filters)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=make_surrogate(surrogate),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(n_filters, n_classes)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SpikingShallowConvNet":
        encoder = config.get("encoder", {})
        model = config.get("model", {})
        architecture = model.get("architecture", "spiking_shallow_convnet")
        if architecture != "spiking_shallow_convnet":
            raise ValueError(f"Unsupported SNN architecture: {architecture}")
        return cls(
            encoder_name=encoder.get("name", "direct"),
            n_filters=int(model.get("n_filters", 40)),
            temporal_kernel=int(model.get("temporal_kernel", 25)),
            beta=float(model.get("beta", 0.90)),
            threshold=float(model.get("threshold", 1.0)),
            surrogate=model.get("surrogate", "atan"),
            dropout=float(model.get("dropout", 0.50)),
            readout=model.get("readout", "mean_membrane"),
            rate_gain=float(encoder.get("rate_gain", 1.0)),
            delta_threshold=float(encoder.get("delta_threshold", 0.20)),
            adaptive_slope=float(encoder.get("adaptive_slope", 12.0)),
            adaptive_threshold_init=float(encoder.get("adaptive_threshold_init", 0.20)),
            adaptive_gain_init=float(encoder.get("adaptive_gain_init", 1.0)),
        )

    def _adaptive_signed_delta(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.adaptive_log_gain is None
            or self.adaptive_log_pos_threshold is None
            or self.adaptive_log_neg_threshold is None
        ):
            raise RuntimeError("Adaptive signed-delta parameters are not initialized")

        diff = torch.zeros_like(x)
        diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        gain = torch.exp(self.adaptive_log_gain).view(1, -1, 1)
        pos_threshold = torch.exp(self.adaptive_log_pos_threshold).view(1, -1, 1)
        neg_threshold = torch.exp(self.adaptive_log_neg_threshold).view(1, -1, 1)
        scaled = diff * gain
        pos = torch.sigmoid(self.adaptive_slope * (scaled - pos_threshold))
        neg = torch.sigmoid(self.adaptive_slope * (-scaled - neg_threshold))
        return torch.cat([pos, neg], dim=1)

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder_name == "adaptive_signed_delta":
            return self._adaptive_signed_delta(x)
        return encode_batch(
            x,
            self.encoder_name,
            rate_gain=self.rate_gain,
            delta_threshold=self.delta_threshold,
            training=self.training,
        )

    def _conv_current(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_input(x)
        current = self.temporal_conv(encoded.unsqueeze(1))
        current = self.spatial_conv(current).squeeze(2)
        return self.bn(current)

    def _spiking_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        current = self._conv_current(x)
        batch_size, _, n_steps = current.shape
        mem = torch.zeros(batch_size, self.n_filters, device=current.device, dtype=current.dtype)
        spike_sum = torch.zeros_like(mem)
        mem_sum = torch.zeros_like(mem)
        mem_max = torch.full_like(mem, -1.0e9)

        for step in range(n_steps):
            spk, mem = self.lif(current[:, :, step], mem)
            spike_sum = spike_sum + spk
            mem_sum = mem_sum + mem
            mem_max = torch.maximum(mem_max, mem)

        spike_count = spike_sum / float(n_steps)
        mean_membrane = mem_sum / float(n_steps)
        spike_rate = spike_sum.mean() / float(n_steps)
        return {
            "spike_count": spike_count,
            "mean_membrane": mean_membrane,
            "final_membrane": mem,
            "max_membrane": mem_max,
            "spike_rate": spike_rate,
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self._spiking_features(x)
        readout_features = self.dropout(select_readout(features, self.readout))
        logits = self.classifier(readout_features)
        if return_aux:
            return logits, {"spike_rate": features["spike_rate"]}
        return logits

    def extract_embedding(self, x: torch.Tensor, mode: str = "concat") -> torch.Tensor:
        features = self._spiking_features(x)
        return select_embedding(features, mode)


def build_model(config: dict[str, Any]) -> SpikingShallowConvNet:
    return SpikingShallowConvNet.from_config(config)
