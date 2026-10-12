from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def collect_rows(results_dir: Path, config: str) -> list[dict]:
    rows = []
    for path in sorted((results_dir / "v3_early_exit" / config).glob("fold_*.json")):
        payload = json.load(path.open("r", encoding="utf-8"))
        for row in payload["prefix_results"]:
            metrics = row["metrics"]
            rows.append(
                {
                    "config": payload["name"],
                    "fold": payload["fold"],
                    "prefix_steps": row["prefix_steps"],
                    "coverage_samples": row["coverage_samples"],
                    "coverage_ms": row["coverage_ms"],
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "loss": metrics["loss"],
                    "spike_rate": metrics["spike_rate"],
                    "head_inference_ms_per_sample": metrics["head_inference_ms_per_sample"],
                    "relative_latency_vs_104": row["relative_latency_vs_104"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SNN early-exit prefix results.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--config", default="shallow_temporal_lif_head")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = collect_rows(results_dir, args.config)
    if not rows:
        print("No early-exit results found.")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["prefix_steps"]].append(row)
    summaries = []
    for prefix, group in sorted(grouped.items()):
        summary = {
            "config": group[0]["config"],
            "prefix_steps": prefix,
            "coverage_samples": group[0]["coverage_samples"],
            "coverage_ms": group[0]["coverage_ms"],
            "folds": len(group),
        }
        for key in (
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "loss",
            "spike_rate",
            "head_inference_ms_per_sample",
            "relative_latency_vs_104",
        ):
            mean, std = _mean_std([float(row[key]) for row in group])
            summary[f"{key}_mean"] = mean
            summary[f"{key}_std"] = std
        summaries.append(summary)

    out_dir = results_dir / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "snn_early_exit_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"Wrote {out_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
