from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.distillation import train_prefix_distill_fold


CONFIG_MAP = {
    "shallow_temporal_lif_distill_k48": ROOT / "configs" / "shallow_temporal_lif_distill_k48.yaml",
    "shallow_temporal_lif_distill_k64": ROOT / "configs" / "shallow_temporal_lif_distill_k64.yaml",
    "shallow_temporal_lif_distill_k80": ROOT / "configs" / "shallow_temporal_lif_distill_k80.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train prefix-specific distilled SNN temporal heads.")
    parser.add_argument("--configs", nargs="+", default=["shallow_temporal_lif_distill_k64"])
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trainfold-dir", default="trainfold")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "artifacts" / "checkpoints"))
    parser.add_argument("--shallow-checkpoint-template", default="baselines/checkpoints/shallow_original_fold{fold}_seed42.pth")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prefix-steps", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    rows = []
    for config_name in args.configs:
        config_path = CONFIG_MAP.get(config_name, Path(config_name))
        for fold in args.folds:
            result = train_prefix_distill_fold(
                config_path,
                fold=fold,
                trainfold_dir=args.trainfold_dir,
                results_dir=args.results_dir,
                checkpoints_dir=args.checkpoints_dir,
                shallow_checkpoint_template=args.shallow_checkpoint_template,
                device=args.device,
                epochs_override=args.epochs,
                batch_size_override=args.batch_size,
                prefix_steps_override=args.prefix_steps,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                progress=not args.no_progress,
            )
            rows.append(
                {
                    "name": result["name"],
                    "fold": fold,
                    "prefix_steps": result["prefix_steps"],
                    "best_epoch": result["best_epoch"],
                    "accuracy": result["test_metrics"]["accuracy"],
                    "macro_f1": result["test_metrics"]["macro_f1"],
                    "balanced_accuracy": result["test_metrics"]["balanced_accuracy"],
                    "checkpoint_path": result["checkpoint_path"],
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
