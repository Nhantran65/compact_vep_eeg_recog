from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.configs import load_config
from snn_vep.hybrid_models import ShallowFeatureSNNHead
from snn_vep.hybrid_training import run_hybrid_probe, train_hybrid_fold
from snn_vep.shallow_features import ShallowFeatureExtractor, prepare_shallow_feature_loaders, shallow_checkpoint_path
from snn_vep.utils import resolve_device, set_seed
from snn_vep.v2_controls import run_v2_control_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test SNN V2 controls and Shallow hybrid.")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = resolve_device(args.device)
    checkpoint = shallow_checkpoint_path(args.fold)
    extractor = ShallowFeatureExtractor(checkpoint)
    sample_loaders = prepare_shallow_feature_loaders(
        fold=args.fold,
        feature_mode="80d",
        batch_size=8,
        feature_batch_size=16,
        device=device,
    )
    features, labels = next(iter(sample_loaders["train_loader"]))
    temporal_loaders = prepare_shallow_feature_loaders(
        fold=args.fold,
        feature_mode="temporal_map",
        batch_size=8,
        feature_batch_size=16,
        device=device,
    )
    temporal_features, _ = next(iter(temporal_loaders["train_loader"]))

    lif = ShallowFeatureSNNHead.from_config(load_config(ROOT / "configs" / "shallow80_lif_head.yaml")).to(device)
    plif = ShallowFeatureSNNHead.from_config(load_config(ROOT / "configs" / "shallow80_plif_head.yaml")).to(device)
    temporal_lif = ShallowFeatureSNNHead.from_config(
        load_config(ROOT / "configs" / "shallow_temporal_lif_head.yaml")
    ).to(device)
    labels = labels.to(device)
    losses = {}
    for name, model, batch in [
        ("lif", lif, features),
        ("plif", plif, features),
        ("temporal_lif", temporal_lif, temporal_features),
    ]:
        model.train()
        logits, aux = model(batch.to(device), return_aux=True)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        losses[name] = {
            "logits_shape": list(logits.shape),
            "loss": float(loss.detach().cpu().item()),
            "spike_rate": float(aux["spike_rate"].detach().cpu().item()),
        }

    control = run_v2_control_fold(
        transform="delta",
        fold=args.fold,
        heads=["rbf_svc_c10_gamma_scale"],
        results_dir=ROOT / "results" / "v2_smoke",
    )
    hybrid = train_hybrid_fold(
        ROOT / "configs" / "shallow80_lif_head.yaml",
        fold=args.fold,
        results_dir=ROOT / "results" / "v2_smoke",
        checkpoints_dir=ROOT / "artifacts" / "v2_smoke_checkpoints",
        device=args.device,
        epochs_override=1,
        batch_size_override=16,
        max_train_batches=2,
        progress=not args.no_progress,
    )
    probe = run_hybrid_probe(
        ROOT / "configs" / "shallow80_lif_head.yaml",
        fold=args.fold,
        checkpoint=hybrid["checkpoint_path"],
        results_dir=ROOT / "results" / "v2_smoke",
        checkpoints_dir=ROOT / "artifacts" / "v2_smoke_checkpoints",
        device=args.device,
    )

    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "extract_80d_shape": list(features.shape),
                "extract_temporal_shape": list(temporal_features.shape),
                "forward_backward": losses,
                "control_delta_rbf": control["heads"][0]["test_metrics"],
                "mini_hybrid": {
                    "best_epoch": hybrid["best_epoch"],
                    "test_metrics": hybrid["test_metrics"],
                    "checkpoint_path": hybrid["checkpoint_path"],
                },
                "probe_heads": [
                    {"name": h["name"], "test_metrics": h["test_metrics"]}
                    for h in probe["heads"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
