from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.early_exit import run_early_exit_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test post-hoc SNN early-exit evaluation.")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timing-repeats", type=int, default=1)
    args = parser.parse_args()

    result = run_early_exit_fold(
        ROOT / "configs" / "shallow_temporal_lif_head.yaml",
        fold=args.fold,
        prefix_steps=[8, 104],
        results_dir=ROOT / "results" / "v3_smoke",
        device=args.device,
        timing_repeats=args.timing_repeats,
    )
    for row in result["prefix_results"]:
        metrics = row["metrics"]
        for key in ("accuracy", "macro_f1", "balanced_accuracy", "loss", "spike_rate", "head_inference_ms_per_sample"):
            if not math.isfinite(float(metrics[key])):
                raise RuntimeError(f"Non-finite {key} for prefix {row['prefix_steps']}: {metrics[key]}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
