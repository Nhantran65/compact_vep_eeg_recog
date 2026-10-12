from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.configs import load_config
from snn_vep.data import prepare_fold_loaders
from snn_vep.encoders import encode_batch
from snn_vep.models import build_model
from snn_vep.probes import run_embedding_probe
from snn_vep.training import train_one_fold
from snn_vep.utils import resolve_device, set_seed


CONFIGS = [
    ROOT / "configs" / "spiking_shallow_direct.yaml",
    ROOT / "configs" / "spiking_shallow_rate.yaml",
    ROOT / "configs" / "spiking_shallow_delta.yaml",
    ROOT / "configs" / "spiking_shallow_signed_delta.yaml",
]


def smoke_forward_backward(fold: int, device: str) -> list[dict]:
    torch_device = resolve_device(device)
    loaders = prepare_fold_loaders("trainfold", fold, batch_size=8, val_size=0.20, seed=42)
    inputs, labels = next(iter(loaders.train_loader))
    inputs = inputs.to(torch_device)
    labels = labels.to(torch_device)
    rows = []

    for path in CONFIGS:
        config = load_config(path)
        set_seed(42)
        model = build_model(config).to(torch_device)
        model.train()
        encoded = encode_batch(
            inputs,
            config["encoder"]["name"],
            rate_gain=float(config["encoder"].get("rate_gain", 1.0)),
            delta_threshold=float(config["encoder"].get("delta_threshold", 0.20)),
            training=True,
        )
        if torch.isnan(encoded).any():
            raise RuntimeError(f"NaN in encoded input for {config['name']}")
        logits, aux = model(inputs, return_aux=True)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        rows.append(
            {
                "config": config["name"],
                "encoded_shape": list(encoded.shape),
                "logits_shape": list(logits.shape),
                "loss": float(loss.detach().cpu().item()),
                "spike_rate": float(aux["spike_rate"].detach().cpu().item()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the SNN VEP pipeline.")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    rows = smoke_forward_backward(args.fold, args.device)
    result = train_one_fold(
        ROOT / "configs" / "spiking_shallow_direct.yaml",
        fold=args.fold,
        results_dir=ROOT / "results" / "smoke",
        checkpoints_dir=ROOT / "artifacts" / "smoke_checkpoints",
        device=args.device,
        epochs_override=1,
        batch_size_override=16,
        max_train_batches=2,
        max_eval_batches=2,
        progress=not args.no_progress,
    )

    probe_summary = None
    if not args.skip_probe:
        probe_result = run_embedding_probe(
            ROOT / "configs" / "snn_embedding_probe.yaml",
            fold=args.fold,
            checkpoint=result["checkpoint_path"],
            results_dir=ROOT / "results" / "smoke",
            checkpoints_dir=ROOT / "artifacts" / "smoke_checkpoints",
            embeddings_dir=ROOT / "artifacts" / "smoke_embeddings",
            device=args.device,
        )
        probe_summary = [
            {
                "head": head["name"],
                "test_accuracy": head["test_metrics"]["accuracy"],
                "test_macro_f1": head["test_metrics"]["macro_f1"],
            }
            for head in probe_result["heads"]
        ]

    summary = {
        "forward_backward": rows,
        "mini_train": {
            "best_epoch": result["best_epoch"],
            "test_accuracy": result["test_metrics"]["accuracy"],
            "test_macro_f1": result["test_metrics"]["macro_f1"],
            "checkpoint_path": result["checkpoint_path"],
        },
        "probe": probe_summary,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
