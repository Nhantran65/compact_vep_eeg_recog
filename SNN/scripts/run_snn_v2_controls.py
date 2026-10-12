from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.v2_controls import HEADS, TRANSFORMS, run_v2_control_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SNN V2 raw/delta classical controls.")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--transforms", nargs="+", default=list(TRANSFORMS))
    parser.add_argument("--heads", nargs="+", default=list(HEADS))
    parser.add_argument("--trainfold-dir", default="trainfold")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for transform in args.transforms:
        for fold in args.folds:
            result = run_v2_control_fold(
                transform=transform,
                fold=fold,
                heads=args.heads,
                trainfold_dir=args.trainfold_dir,
                results_dir=args.results_dir,
                seed=args.seed,
            )
            for head in result["heads"]:
                rows.append(
                    {
                        "config": result["name"],
                        "head": head["name"],
                        "fold": fold,
                        "accuracy": head["test_metrics"]["accuracy"],
                        "macro_f1": head["test_metrics"]["macro_f1"],
                    }
                )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
