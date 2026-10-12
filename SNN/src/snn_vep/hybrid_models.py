from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import make_surrogate
from .readouts import select_embedding, select_readout


class ShallowFeatureSNNHead(nn.Module):
    """SNN readout over frozen ShallowConvNet 80D or temporal-map features."""

    def __init__(
        self,
        *,
        feature_mode: str = "80d",
        hidden_dim: int = 64,
        beta: float = 0.90,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        surrogate: str = "atan",
        dropout: float = 0.30,
        readout: str = "mean_membrane",
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError(
                "snnTorch is required for SNN hybrid heads. Install with: "
                "uv pip install -r SNN/requirements.txt"
            ) from exc

        if feature_mode not in {"80d", "temporal_map"}:
            raise ValueError(f"Unsupported hybrid feature mode: {feature_mode}")
        self.feature_mode = feature_mode
        self.hidden_dim = hidden_dim
        self.readout = readout
        in_dim = 40 if feature_mode == "80d" else 40
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=make_surrogate(surrogate),
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ShallowFeatureSNNHead":
        model = config.get("model", {})
        return cls(
            feature_mode=model.get("feature_mode", "80d"),
            hidden_dim=int(model.get("hidden_dim", 64)),
            beta=float(model.get("beta", 0.90)),
            threshold=float(model.get("threshold", 1.0)),
            learn_beta=bool(model.get("learn_beta", False)),
            learn_threshold=bool(model.get("learn_threshold", False)),
            surrogate=model.get("surrogate", "atan"),
            dropout=float(model.get("dropout", 0.30)),
            readout=model.get("readout", "mean_membrane"),
        )

    def _sequence(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_mode == "80d":
            if x.ndim != 2 or x.shape[1] != 80:
                raise ValueError(f"Expected [B,80] Shallow features, got {tuple(x.shape)}")
            return x.view(x.size(0), 40, 2)
        if x.ndim != 3 or x.shape[1] != 40:
            raise ValueError(f"Expected [B,40,T] Shallow temporal map, got {tuple(x.shape)}")
        return x

    def _features(self, x: torch.Tensor, prefix_steps: int | None = None) -> dict[str, torch.Tensor]:
        seq = self._sequence(x)
        if prefix_steps is not None:
            if prefix_steps < 1 or prefix_steps > seq.shape[2]:
                raise ValueError(
                    f"prefix_steps must be in [1,{seq.shape[2]}], got {prefix_steps}"
                )
            seq = seq[:, :, :prefix_steps]
        batch_size, _, n_steps = seq.shape
        mem = torch.zeros(batch_size, self.hidden_dim, device=seq.device, dtype=seq.dtype)
        spike_sum = torch.zeros_like(mem)
        mem_sum = torch.zeros_like(mem)
        mem_max = torch.full_like(mem, -1.0e9)
        spike_steps = []

        for step in range(n_steps):
            current = self.input_proj(seq[:, :, step])
            spk, mem = self.lif(current, mem)
            spike_sum = spike_sum + spk
            mem_sum = mem_sum + mem
            mem_max = torch.maximum(mem_max, mem)
            spike_steps.append(spk)

        return {
            "spike_count": spike_sum / float(n_steps),
            "mean_membrane": mem_sum / float(n_steps),
            "final_membrane": mem,
            "max_membrane": mem_max,
            "spike_sequence": torch.stack(spike_steps, dim=2),
            "spike_rate": spike_sum.mean() / float(n_steps),
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False, prefix_steps: int | None = None):
        features = self._features(x, prefix_steps=prefix_steps)
        readout_features = self.dropout(select_readout(features, self.readout))
        logits = self.classifier(readout_features)
        if return_aux:
            return logits, {"spike_rate": features["spike_rate"]}
        return logits

    def extract_embedding(
        self,
        x: torch.Tensor,
        mode: str = "concat",
        prefix_steps: int | None = None,
    ) -> torch.Tensor:
        return select_embedding(self._features(x, prefix_steps=prefix_steps), mode)


class ShallowTemporalSpikeAwareGatedLIFHead(nn.Module):
    """Gated membrane readout fused with spike-count evidence."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        beta: float = 0.90,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        surrogate: str = "atan",
        dropout: float = 0.30,
        gate_kernel: int = 9,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError(
                "snnTorch is required for SNN hybrid heads. Install with: "
                "uv pip install -r SNN/requirements.txt"
            ) from exc

        if gate_kernel < 1 or gate_kernel % 2 == 0:
            raise ValueError("gate_kernel must be a positive odd integer")
        self.feature_mode = "temporal_map"
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(40, hidden_dim)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=make_surrogate(surrogate),
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )
        self.gate = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=gate_kernel,
            padding=gate_kernel // 2,
            groups=hidden_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ShallowTemporalSpikeAwareGatedLIFHead":
        model = config.get("model", {})
        return cls(
            hidden_dim=int(model.get("hidden_dim", 64)),
            beta=float(model.get("beta", 0.90)),
            threshold=float(model.get("threshold", 1.0)),
            learn_beta=bool(model.get("learn_beta", False)),
            learn_threshold=bool(model.get("learn_threshold", False)),
            surrogate=model.get("surrogate", "atan"),
            dropout=float(model.get("dropout", 0.30)),
            gate_kernel=int(model.get("gate_kernel", 9)),
        )

    def _sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 40:
            raise ValueError(f"Expected [B,40,T] Shallow temporal map, got {tuple(x.shape)}")
        return x

    def _features(self, x: torch.Tensor, prefix_steps: int | None = None) -> dict[str, torch.Tensor]:
        seq = self._sequence(x)
        if prefix_steps is not None:
            if prefix_steps < 1 or prefix_steps > seq.shape[2]:
                raise ValueError(
                    f"prefix_steps must be in [1,{seq.shape[2]}], got {prefix_steps}"
                )
            seq = seq[:, :, :prefix_steps]
        batch_size, _, n_steps = seq.shape
        mem = torch.zeros(batch_size, self.hidden_dim, device=seq.device, dtype=seq.dtype)
        spike_sum = torch.zeros_like(mem)
        mem_steps = []
        spike_steps = []

        for step in range(n_steps):
            current = self.input_proj(seq[:, :, step])
            spk, mem = self.lif(current, mem)
            spike_sum = spike_sum + spk
            mem_steps.append(mem)
            spike_steps.append(spk)

        mem_seq = torch.stack(mem_steps, dim=2)
        spike_seq = torch.stack(spike_steps, dim=2)
        gate = torch.sigmoid(self.gate(mem_seq))
        gated_mean = (mem_seq * gate).sum(dim=2) / gate.sum(dim=2).clamp_min(1.0e-6)
        return {
            "spike_count": spike_sum / float(n_steps),
            "mean_membrane": mem_seq.mean(dim=2),
            "final_membrane": mem,
            "max_membrane": mem_seq.max(dim=2).values,
            "gated_mean_membrane": gated_mean,
            "spike_sequence": spike_seq,
            "mean_gate": gate.mean(),
            "spike_rate": spike_seq.mean(),
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False, prefix_steps: int | None = None):
        features = self._features(x, prefix_steps=prefix_steps)
        fused = self.fusion(
            torch.cat([features["gated_mean_membrane"], features["spike_count"]], dim=1)
        )
        logits = self.classifier(self.dropout(fused))
        if return_aux:
            return logits, {
                "spike_rate": features["spike_rate"],
                "mean_gate": features["mean_gate"],
            }
        return logits

    def extract_embedding(
        self,
        x: torch.Tensor,
        mode: str = "concat",
        prefix_steps: int | None = None,
    ) -> torch.Tensor:
        features = self._features(x, prefix_steps=prefix_steps)
        if mode == "gated_mean_membrane":
            return features["gated_mean_membrane"]
        return select_embedding(features, mode)


class ShallowTemporalWindowedSpikeLIFHead(nn.Module):
    """Supervised readout over temporal bins of LIF spike counts."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        beta: float = 0.90,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        surrogate: str = "atan",
        dropout: float = 0.30,
        n_bins: int = 4,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError(
                "snnTorch is required for SNN hybrid heads. Install with: "
                "uv pip install -r SNN/requirements.txt"
            ) from exc

        if n_bins < 1:
            raise ValueError("n_bins must be positive")
        self.feature_mode = "temporal_map"
        self.hidden_dim = hidden_dim
        self.n_bins = n_bins
        self.input_proj = nn.Linear(40, hidden_dim)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=make_surrogate(surrogate),
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )
        in_dim = hidden_dim * n_bins
        self.readout = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ShallowTemporalWindowedSpikeLIFHead":
        model = config.get("model", {})
        return cls(
            hidden_dim=int(model.get("hidden_dim", 64)),
            beta=float(model.get("beta", 0.90)),
            threshold=float(model.get("threshold", 1.0)),
            learn_beta=bool(model.get("learn_beta", False)),
            learn_threshold=bool(model.get("learn_threshold", False)),
            surrogate=model.get("surrogate", "atan"),
            dropout=float(model.get("dropout", 0.30)),
            n_bins=int(model.get("n_bins", 4)),
        )

    def _sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 40:
            raise ValueError(f"Expected [B,40,T] Shallow temporal map, got {tuple(x.shape)}")
        return x

    def _features(self, x: torch.Tensor, prefix_steps: int | None = None) -> dict[str, torch.Tensor]:
        seq = self._sequence(x)
        if prefix_steps is not None:
            if prefix_steps < 1 or prefix_steps > seq.shape[2]:
                raise ValueError(
                    f"prefix_steps must be in [1,{seq.shape[2]}], got {prefix_steps}"
                )
            seq = seq[:, :, :prefix_steps]
        batch_size, _, n_steps = seq.shape
        if self.n_bins > n_steps:
            raise ValueError(f"n_bins must be <= n_steps ({n_steps}), got {self.n_bins}")
        mem = torch.zeros(batch_size, self.hidden_dim, device=seq.device, dtype=seq.dtype)
        spike_sum = torch.zeros_like(mem)
        mem_steps = []
        spike_steps = []

        for step in range(n_steps):
            current = self.input_proj(seq[:, :, step])
            spk, mem = self.lif(current, mem)
            spike_sum = spike_sum + spk
            mem_steps.append(mem)
            spike_steps.append(spk)

        mem_seq = torch.stack(mem_steps, dim=2)
        spike_seq = torch.stack(spike_steps, dim=2)
        chunks = torch.chunk(spike_seq, self.n_bins, dim=2)
        windowed_spike_count = torch.cat([chunk.mean(dim=2) for chunk in chunks], dim=1)
        return {
            "spike_count": spike_sum / float(n_steps),
            "mean_membrane": mem_seq.mean(dim=2),
            "final_membrane": mem,
            "max_membrane": mem_seq.max(dim=2).values,
            "windowed_spike_count": windowed_spike_count,
            "spike_sequence": spike_seq,
            "spike_rate": spike_seq.mean(),
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False, prefix_steps: int | None = None):
        features = self._features(x, prefix_steps=prefix_steps)
        logits = self.readout(features["windowed_spike_count"])
        if return_aux:
            return logits, {"spike_rate": features["spike_rate"]}
        return logits

    def extract_embedding(
        self,
        x: torch.Tensor,
        mode: str = "windowed_spike_count",
        prefix_steps: int | None = None,
    ) -> torch.Tensor:
        features = self._features(x, prefix_steps=prefix_steps)
        if mode == "windowed_spike_count":
            return features["windowed_spike_count"]
        return select_embedding(features, mode)


class DifferentiableRBFPrototypeHead(nn.Module):
    """Class-prototype RBF-style classifier trained with backprop."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        n_classes: int = 4,
        prototypes_per_class: int = 4,
        init_gamma: float = 8.0,
        learn_gamma: bool = True,
        learn_bias: bool = True,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if prototypes_per_class < 1:
            raise ValueError("prototypes_per_class must be positive")
        if init_gamma <= 0.0:
            raise ValueError("init_gamma must be positive")
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.prototypes_per_class = prototypes_per_class
        self.eps = eps
        self.prototypes = nn.Parameter(torch.empty(n_classes, prototypes_per_class, hidden_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.05)

        raw_gamma = torch.log(torch.expm1(torch.tensor(float(init_gamma))))
        if learn_gamma:
            self.raw_gamma = nn.Parameter(raw_gamma)
        else:
            self.register_buffer("raw_gamma", raw_gamma)
        bias = torch.zeros(n_classes, prototypes_per_class)
        if learn_bias:
            self.prototype_bias = nn.Parameter(bias)
        else:
            self.register_buffer("prototype_bias", bias)

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma) + self.eps

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        z = F.normalize(embedding, dim=1)
        prototypes = F.normalize(self.prototypes, dim=2)
        d2 = (z[:, None, None, :] - prototypes[None, :, :, :]).pow(2).sum(dim=-1)
        scores = -self.gamma * d2 + self.prototype_bias
        return torch.logsumexp(scores, dim=2)

    def separation_loss(self, margin: float = 0.0) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=2).reshape(
            self.n_classes * self.prototypes_per_class,
            self.hidden_dim,
        )
        labels = torch.arange(self.n_classes, device=prototypes.device).repeat_interleave(
            self.prototypes_per_class
        )
        same_class = labels[:, None] == labels[None, :]
        cross_class = ~same_class
        similarity = prototypes @ prototypes.T
        return F.relu(similarity[cross_class] - margin).pow(2).mean()


class ShallowTemporalSpikeAwareProtoLIFHead(nn.Module):
    """Spike-aware gated LIF with an end-to-end differentiable RBF prototype head."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        beta: float = 0.90,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        surrogate: str = "atan",
        dropout: float = 0.30,
        gate_kernel: int = 9,
        prototypes_per_class: int = 4,
        init_gamma: float = 8.0,
        learn_gamma: bool = True,
        learn_prototype_bias: bool = True,
        prototype_sep_margin: float = 0.0,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError(
                "snnTorch is required for SNN hybrid heads. Install with: "
                "uv pip install -r SNN/requirements.txt"
            ) from exc

        if gate_kernel < 1 or gate_kernel % 2 == 0:
            raise ValueError("gate_kernel must be a positive odd integer")
        self.feature_mode = "temporal_map"
        self.hidden_dim = hidden_dim
        self.prototype_sep_margin = prototype_sep_margin
        self.input_proj = nn.Linear(40, hidden_dim)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=make_surrogate(surrogate),
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )
        self.gate = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=gate_kernel,
            padding=gate_kernel // 2,
            groups=hidden_dim,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.prototype_head = DifferentiableRBFPrototypeHead(
            hidden_dim=hidden_dim,
            n_classes=n_classes,
            prototypes_per_class=prototypes_per_class,
            init_gamma=init_gamma,
            learn_gamma=learn_gamma,
            learn_bias=learn_prototype_bias,
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ShallowTemporalSpikeAwareProtoLIFHead":
        model = config.get("model", {})
        return cls(
            hidden_dim=int(model.get("hidden_dim", 64)),
            beta=float(model.get("beta", 0.90)),
            threshold=float(model.get("threshold", 1.0)),
            learn_beta=bool(model.get("learn_beta", False)),
            learn_threshold=bool(model.get("learn_threshold", False)),
            surrogate=model.get("surrogate", "atan"),
            dropout=float(model.get("dropout", 0.30)),
            gate_kernel=int(model.get("gate_kernel", 9)),
            prototypes_per_class=int(model.get("prototypes_per_class", 4)),
            init_gamma=float(model.get("init_gamma", 8.0)),
            learn_gamma=bool(model.get("learn_gamma", True)),
            learn_prototype_bias=bool(model.get("learn_prototype_bias", True)),
            prototype_sep_margin=float(model.get("prototype_sep_margin", 0.0)),
        )

    def _sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 40:
            raise ValueError(f"Expected [B,40,T] Shallow temporal map, got {tuple(x.shape)}")
        return x

    def _features(self, x: torch.Tensor, prefix_steps: int | None = None) -> dict[str, torch.Tensor]:
        seq = self._sequence(x)
        if prefix_steps is not None:
            if prefix_steps < 1 or prefix_steps > seq.shape[2]:
                raise ValueError(
                    f"prefix_steps must be in [1,{seq.shape[2]}], got {prefix_steps}"
                )
            seq = seq[:, :, :prefix_steps]
        batch_size, _, n_steps = seq.shape
        mem = torch.zeros(batch_size, self.hidden_dim, device=seq.device, dtype=seq.dtype)
        spike_sum = torch.zeros_like(mem)
        mem_steps = []
        spike_steps = []

        for step in range(n_steps):
            current = self.input_proj(seq[:, :, step])
            spk, mem = self.lif(current, mem)
            spike_sum = spike_sum + spk
            mem_steps.append(mem)
            spike_steps.append(spk)

        mem_seq = torch.stack(mem_steps, dim=2)
        spike_seq = torch.stack(spike_steps, dim=2)
        gate = torch.sigmoid(self.gate(mem_seq))
        gated_mean = (mem_seq * gate).sum(dim=2) / gate.sum(dim=2).clamp_min(1.0e-6)
        fused = self.fusion(torch.cat([gated_mean, spike_sum / float(n_steps)], dim=1))
        return {
            "spike_count": spike_sum / float(n_steps),
            "mean_membrane": mem_seq.mean(dim=2),
            "final_membrane": mem,
            "max_membrane": mem_seq.max(dim=2).values,
            "gated_mean_membrane": gated_mean,
            "fused_embedding": fused,
            "spike_sequence": spike_seq,
            "mean_gate": gate.mean(),
            "spike_rate": spike_seq.mean(),
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False, prefix_steps: int | None = None):
        features = self._features(x, prefix_steps=prefix_steps)
        embedding = features["fused_embedding"]
        logits = self.prototype_head(self.dropout(embedding))
        if return_aux:
            return logits, {
                "embedding": embedding,
                "spike_rate": features["spike_rate"],
                "mean_gate": features["mean_gate"],
                "gamma": self.prototype_head.gamma.detach(),
                "prototype_sep_loss": self.prototype_head.separation_loss(
                    margin=self.prototype_sep_margin
                ),
            }
        return logits

    def extract_embedding(
        self,
        x: torch.Tensor,
        mode: str = "fused_embedding",
        prefix_steps: int | None = None,
    ) -> torch.Tensor:
        features = self._features(x, prefix_steps=prefix_steps)
        if mode == "fused_embedding":
            return features["fused_embedding"]
        if mode == "gated_mean_membrane":
            return features["gated_mean_membrane"]
        return select_embedding(features, mode)


class ShallowTemporalGatedLIFHead(nn.Module):
    """VEP-aware gated LIF readout over frozen Shallow temporal-map features."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        beta: float = 0.90,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        surrogate: str = "atan",
        dropout: float = 0.30,
        gate_kernel: int = 9,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError(
                "snnTorch is required for SNN hybrid heads. Install with: "
                "uv pip install -r SNN/requirements.txt"
            ) from exc

        if gate_kernel < 1 or gate_kernel % 2 == 0:
            raise ValueError("gate_kernel must be a positive odd integer")
        self.feature_mode = "temporal_map"
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(40, hidden_dim)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=make_surrogate(surrogate),
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )
        self.gate = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=gate_kernel,
            padding=gate_kernel // 2,
            groups=hidden_dim,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ShallowTemporalGatedLIFHead":
        model = config.get("model", {})
        return cls(
            hidden_dim=int(model.get("hidden_dim", 64)),
            beta=float(model.get("beta", 0.90)),
            threshold=float(model.get("threshold", 1.0)),
            learn_beta=bool(model.get("learn_beta", False)),
            learn_threshold=bool(model.get("learn_threshold", False)),
            surrogate=model.get("surrogate", "atan"),
            dropout=float(model.get("dropout", 0.30)),
            gate_kernel=int(model.get("gate_kernel", 9)),
        )

    def _sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 40:
            raise ValueError(f"Expected [B,40,T] Shallow temporal map, got {tuple(x.shape)}")
        return x

    def _features(self, x: torch.Tensor, prefix_steps: int | None = None) -> dict[str, torch.Tensor]:
        seq = self._sequence(x)
        if prefix_steps is not None:
            if prefix_steps < 1 or prefix_steps > seq.shape[2]:
                raise ValueError(
                    f"prefix_steps must be in [1,{seq.shape[2]}], got {prefix_steps}"
                )
            seq = seq[:, :, :prefix_steps]
        batch_size, _, n_steps = seq.shape
        mem = torch.zeros(batch_size, self.hidden_dim, device=seq.device, dtype=seq.dtype)
        spike_sum = torch.zeros_like(mem)
        mem_steps = []
        spike_steps = []

        for step in range(n_steps):
            current = self.input_proj(seq[:, :, step])
            spk, mem = self.lif(current, mem)
            spike_sum = spike_sum + spk
            mem_steps.append(mem)
            spike_steps.append(spk)

        mem_seq = torch.stack(mem_steps, dim=2)
        spike_seq = torch.stack(spike_steps, dim=2)
        gate = torch.sigmoid(self.gate(mem_seq))
        gated_mean = (mem_seq * gate).sum(dim=2) / gate.sum(dim=2).clamp_min(1.0e-6)
        return {
            "spike_count": spike_sum / float(n_steps),
            "mean_membrane": mem_seq.mean(dim=2),
            "final_membrane": mem,
            "max_membrane": mem_seq.max(dim=2).values,
            "gated_mean_membrane": gated_mean,
            "spike_sequence": spike_seq,
            "mean_gate": gate.mean(),
            "spike_rate": spike_seq.mean(),
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False, prefix_steps: int | None = None):
        features = self._features(x, prefix_steps=prefix_steps)
        readout_features = self.dropout(features["gated_mean_membrane"])
        logits = self.classifier(readout_features)
        if return_aux:
            return logits, {
                "spike_rate": features["spike_rate"],
                "mean_gate": features["mean_gate"],
            }
        return logits

    def extract_embedding(
        self,
        x: torch.Tensor,
        mode: str = "concat",
        prefix_steps: int | None = None,
    ) -> torch.Tensor:
        features = self._features(x, prefix_steps=prefix_steps)
        if mode == "gated_mean_membrane":
            return features["gated_mean_membrane"]
        return select_embedding(features, mode)


class ShallowTemporalMultiscaleGatedLIFHead(nn.Module):
    """Gated LIF readout with multiple temporal gate scales."""

    def __init__(
        self,
        *,
        hidden_dim: int = 64,
        beta: float = 0.90,
        threshold: float = 1.0,
        learn_beta: bool = False,
        learn_threshold: bool = False,
        surrogate: str = "atan",
        dropout: float = 0.30,
        gate_kernels: list[int] | None = None,
        n_classes: int = 4,
    ) -> None:
        super().__init__()
        try:
            import snntorch as snn
        except ImportError as exc:
            raise ImportError(
                "snnTorch is required for SNN hybrid heads. Install with: "
                "uv pip install -r SNN/requirements.txt"
            ) from exc

        gate_kernels = gate_kernels or [3, 9, 17]
        for kernel in gate_kernels:
            if kernel < 1 or kernel % 2 == 0:
                raise ValueError("All gate_kernels must be positive odd integers")
        self.feature_mode = "temporal_map"
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(40, hidden_dim)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=make_surrogate(surrogate),
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        )
        self.gates = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel,
                    padding=kernel // 2,
                    groups=hidden_dim,
                )
                for kernel in gate_kernels
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(gate_kernels)))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ShallowTemporalMultiscaleGatedLIFHead":
        model = config.get("model", {})
        return cls(
            hidden_dim=int(model.get("hidden_dim", 64)),
            beta=float(model.get("beta", 0.90)),
            threshold=float(model.get("threshold", 1.0)),
            learn_beta=bool(model.get("learn_beta", False)),
            learn_threshold=bool(model.get("learn_threshold", False)),
            surrogate=model.get("surrogate", "atan"),
            dropout=float(model.get("dropout", 0.30)),
            gate_kernels=[int(k) for k in model.get("gate_kernels", [3, 9, 17])],
        )

    def _sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 40:
            raise ValueError(f"Expected [B,40,T] Shallow temporal map, got {tuple(x.shape)}")
        return x

    def _features(self, x: torch.Tensor, prefix_steps: int | None = None) -> dict[str, torch.Tensor]:
        seq = self._sequence(x)
        if prefix_steps is not None:
            if prefix_steps < 1 or prefix_steps > seq.shape[2]:
                raise ValueError(
                    f"prefix_steps must be in [1,{seq.shape[2]}], got {prefix_steps}"
                )
            seq = seq[:, :, :prefix_steps]
        batch_size, _, n_steps = seq.shape
        mem = torch.zeros(batch_size, self.hidden_dim, device=seq.device, dtype=seq.dtype)
        spike_sum = torch.zeros_like(mem)
        mem_steps = []
        spike_steps = []

        for step in range(n_steps):
            current = self.input_proj(seq[:, :, step])
            spk, mem = self.lif(current, mem)
            spike_sum = spike_sum + spk
            mem_steps.append(mem)
            spike_steps.append(spk)

        mem_seq = torch.stack(mem_steps, dim=2)
        spike_seq = torch.stack(spike_steps, dim=2)
        gate_stack = torch.stack([torch.sigmoid(gate(mem_seq)) for gate in self.gates], dim=0)
        weights = torch.softmax(self.scale_logits, dim=0).view(-1, 1, 1, 1)
        gate = (gate_stack * weights).sum(dim=0)
        gated_mean = (mem_seq * gate).sum(dim=2) / gate.sum(dim=2).clamp_min(1.0e-6)
        return {
            "spike_count": spike_sum / float(n_steps),
            "mean_membrane": mem_seq.mean(dim=2),
            "final_membrane": mem,
            "max_membrane": mem_seq.max(dim=2).values,
            "gated_mean_membrane": gated_mean,
            "spike_sequence": spike_seq,
            "mean_gate": gate.mean(),
            "spike_rate": spike_seq.mean(),
        }

    def forward(self, x: torch.Tensor, return_aux: bool = False, prefix_steps: int | None = None):
        features = self._features(x, prefix_steps=prefix_steps)
        logits = self.classifier(self.dropout(features["gated_mean_membrane"]))
        if return_aux:
            return logits, {
                "spike_rate": features["spike_rate"],
                "mean_gate": features["mean_gate"],
            }
        return logits

    def extract_embedding(
        self,
        x: torch.Tensor,
        mode: str = "concat",
        prefix_steps: int | None = None,
    ) -> torch.Tensor:
        features = self._features(x, prefix_steps=prefix_steps)
        if mode == "gated_mean_membrane":
            return features["gated_mean_membrane"]
        return select_embedding(features, mode)


def build_hybrid_model(config: dict[str, Any]) -> nn.Module:
    architecture = config.get("model", {}).get("architecture", "shallow_feature_snn_head")
    if architecture == "shallow_feature_snn_head":
        return ShallowFeatureSNNHead.from_config(config)
    if architecture == "shallow_temporal_gated_lif":
        return ShallowTemporalGatedLIFHead.from_config(config)
    if architecture == "shallow_temporal_spike_aware_gated_lif":
        return ShallowTemporalSpikeAwareGatedLIFHead.from_config(config)
    if architecture == "shallow_temporal_spike_aware_proto_lif":
        return ShallowTemporalSpikeAwareProtoLIFHead.from_config(config)
    if architecture == "shallow_temporal_windowed_spike_lif":
        return ShallowTemporalWindowedSpikeLIFHead.from_config(config)
    if architecture == "shallow_temporal_multiscale_gated_lif":
        return ShallowTemporalMultiscaleGatedLIFHead.from_config(config)
    raise ValueError(f"Unsupported hybrid architecture: {architecture}")
