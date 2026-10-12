from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_shallow_snn_hybrid import CONFIG_MAP
from snn_vep.hybrid_training import run_hybrid_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run embedding probes for trained Shallow+SNN hybrid heads.")
    parser.add_argument("--config", default="shallow80_lif_head")
    parser.add_argument("--fold", type=int, default=None, help="Single fold for backward compatibility.")
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trainfold-dir", default="trainfold")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--checkpoints-dir", default=str(ROOT / "artifacts" / "checkpoints"))
    parser.add_argument("--shallow-checkpoint-template", default="baselines/checkpoints/shallow_original_fold{fold}_seed42.pth")
    parser.add_argument("--mode", default="concat")
    parser.add_argument("--modes", nargs="+", default=None)
    args = parser.parse_args()

    folds = args.folds if args.folds is not None else ([args.fold] if args.fold is not None else None)
    if not folds:
        parser.error("Provide --fold or --folds")
    modes = args.modes if args.modes is not None else [args.mode]
    if args.checkpoint is not None and (len(folds) > 1 or len(modes) > 1):
        parser.error("--checkpoint can only be used with one fold and one mode")

    config_path = CONFIG_MAP.get(args.config, Path(args.config))
    rows = []
    for fold in folds:
        for mode in modes:
            result = run_hybrid_probe(
                config_path,
                fold=fold,
                checkpoint=args.checkpoint,
                trainfold_dir=args.trainfold_dir,
                results_dir=args.results_dir,
                checkpoints_dir=args.checkpoints_dir,
                shallow_checkpoint_template=args.shallow_checkpoint_template,
                device=args.device,
                mode=mode,
            )
            rows.append(
                {
                    "name": result["name"],
                    "fold": result["fold"],
                    "embedding_mode": result["embedding_mode"],
                    "embedding_shape": result["embedding_shape"],
                    "heads": [
                        {"name": h["name"], "test_metrics": h["test_metrics"]}
                        for h in result["heads"]
                    ],
                }
            )
    print(
        json.dumps(
            rows[0] if len(rows) == 1 and args.folds is None and args.modes is None else rows,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
