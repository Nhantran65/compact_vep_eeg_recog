from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.early_exit import DEFAULT_PREFIX_STEPS, run_early_exit_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc early-exit curve for SNN temporal hybrid.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "shallow_temporal_lif_head.yaml"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--prefix-steps", nargs="+", type=int, default=list(DEFAULT_PREFIX_STEPS))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trainfold-dir", default="trainfold")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "artifacts" / "checkpoints"))
    parser.add_argument("--shallow-checkpoint-template", default="baselines/checkpoints/shallow_original_fold{fold}_seed42.pth")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=None)
    parser.add_argument("--timing-repeats", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for fold in args.folds:
        result = run_early_exit_fold(
            args.config,
            fold=fold,
            prefix_steps=args.prefix_steps,
            trainfold_dir=args.trainfold_dir,
            results_dir=args.results_dir,
            checkpoints_dir=args.checkpoints_dir,
            shallow_checkpoint_template=args.shallow_checkpoint_template,
            device=args.device,
            batch_size=args.batch_size,
            feature_batch_size=args.feature_batch_size,
            timing_repeats=args.timing_repeats,
        )
        for prefix_row in result["prefix_results"]:
            metrics = prefix_row["metrics"]
            rows.append(
                {
                    "fold": fold,
                    "prefix_steps": prefix_row["prefix_steps"],
                    "coverage_ms": prefix_row["coverage_ms"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "spike_rate": metrics["spike_rate"],
                    "head_inference_ms_per_sample": metrics["head_inference_ms_per_sample"],
                    "relative_latency_vs_104": prefix_row["relative_latency_vs_104"],
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
