from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.training import train_one_fold


DEFAULT_CONFIGS = [
    ROOT / "configs" / "spiking_shallow_direct.yaml",
    ROOT / "configs" / "spiking_shallow_rate.yaml",
    ROOT / "configs" / "spiking_shallow_delta.yaml",
    ROOT / "configs" / "spiking_shallow_signed_delta.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SNN encoding ablation.")
    parser.add_argument("--configs", nargs="*", default=[str(p) for p in DEFAULT_CONFIGS])
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trainfold-dir", default="trainfold")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "artifacts" / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    rows = []
    for config in args.configs:
        for fold in args.folds:
            result = train_one_fold(
                config,
                fold=fold,
                trainfold_dir=args.trainfold_dir,
                results_dir=args.results_dir,
                checkpoints_dir=args.checkpoints_dir,
                device=args.device,
                epochs_override=args.epochs,
                batch_size_override=args.batch_size,
                progress=not args.no_progress,
            )
            rows.append(
                {
                    "name": result["name"],
                    "fold": result["fold"],
                    "accuracy": result["test_metrics"]["accuracy"],
                    "macro_f1": result["test_metrics"]["macro_f1"],
                    "balanced_accuracy": result["test_metrics"]["balanced_accuracy"],
                    "checkpoint_path": result["checkpoint_path"],
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
