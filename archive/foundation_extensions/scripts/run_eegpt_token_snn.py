from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.eegpt_token_snn import train_eegpt_token_snn_fold


CONFIG_MAP = {
    "eegpt_token_spike_aware_gated_lif": ROOT / "configs" / "eegpt_token_spike_aware_gated_lif.yaml",
    "eegpt_token_spike_aware_proto_lif_ce": ROOT / "configs" / "eegpt_token_spike_aware_proto_lif_ce.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SNN heads over cached EEGPT token embeddings.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["eegpt_token_spike_aware_gated_lif", "eegpt_token_spike_aware_proto_lif_ce"],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[1])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "artifacts" / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    rows = []
    for config_name in args.configs:
        config_path = CONFIG_MAP.get(config_name, Path(config_name))
        for fold in args.folds:
            result = train_eegpt_token_snn_fold(
                config_path,
                fold=fold,
                results_dir=args.results_dir,
                checkpoints_dir=args.checkpoints_dir,
                device=args.device,
                progress=not args.no_progress,
                epochs_override=args.epochs,
                batch_size_override=args.batch_size,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
            )
            rows.append(
                {
                    "name": result["name"],
                    "fold": fold,
                    "best_epoch": result["best_epoch"],
                    "accuracy": result["test_metrics"]["accuracy"],
                    "macro_f1": result["test_metrics"]["macro_f1"],
                    "balanced_accuracy": result["test_metrics"]["balanced_accuracy"],
                    "train_minutes": result["train_minutes"],
                    "spike_rate": result["test_metrics"].get("spike_rate"),
                    "mean_gate": result["test_metrics"].get("mean_gate"),
                    "gamma": result["test_metrics"].get("gamma"),
                    "checkpoint_path": result["checkpoint_path"],
                }
            )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
