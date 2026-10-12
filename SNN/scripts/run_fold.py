from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.training import train_one_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one SNN config on one fold.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trainfold-dir", default="trainfold")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "artifacts" / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    result = train_one_fold(
        args.config,
        fold=args.fold,
        trainfold_dir=args.trainfold_dir,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        device=args.device,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        progress=not args.no_progress,
    )
    print(
        json.dumps(
            {
                "name": result["name"],
                "fold": result["fold"],
                "best_epoch": result["best_epoch"],
                "test_metrics": result["test_metrics"],
                "checkpoint_path": result["checkpoint_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
