from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from snn_vep.metrics import mean_metric


def format_percent(mean: float, std: float) -> str:
    return f"{mean * 100:.2f} +/- {std * 100:.2f}%"


def collect_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((results_dir / "fm_token_snn").glob("*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload["test_metrics"]
        rows.append(
            {
                "config": payload["name"],
                "fold": payload["fold"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "train_minutes": payload.get("train_minutes", payload.get("seconds", 0.0) / 60.0),
                "best_epoch": payload["best_epoch"],
                "spike_rate": metrics.get("spike_rate"),
                "mean_gate": metrics.get("mean_gate"),
                "gamma": metrics.get("gamma"),
                "path": str(path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SNN heads trained over EEGPT token caches.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    rows = collect_rows(results_dir)
    if not rows:
        print("No SNN FM-token result JSON files found.")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    summaries = []
    for config, group in sorted(grouped.items()):
        acc_mean, acc_std = mean_metric(group, "accuracy")
        f1_mean, f1_std = mean_metric(group, "macro_f1")
        bal_mean, bal_std = mean_metric(group, "balanced_accuracy")
        minutes_mean, minutes_std = mean_metric(group, "train_minutes")
        summaries.append(
            {
                "config": config,
                "folds": len(group),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "balanced_accuracy_mean": bal_mean,
                "balanced_accuracy_std": bal_std,
                "train_minutes_mean": minutes_mean,
                "train_minutes_std": minutes_std,
            }
        )

    out_dir = results_dir / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "snn_fm_token_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    md_path = out_dir / "snn_fm_token_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# SNN Foundation-Model Token Summary\n\n")
        handle.write("| Config | Folds | Accuracy | Macro-F1 | Balanced acc | Train time |\n")
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in summaries:
            handle.write(
                "| {config} | {folds} | {acc} | {f1} | {bal} | {minutes:.2f} +/- {minutes_std:.2f} min |\n".format(
                    config=row["config"],
                    folds=row["folds"],
                    acc=format_percent(row["accuracy_mean"], row["accuracy_std"]),
                    f1=format_percent(row["macro_f1_mean"], row["macro_f1_std"]),
                    bal=format_percent(row["balanced_accuracy_mean"], row["balanced_accuracy_std"]),
                    minutes=row["train_minutes_mean"],
                    minutes_std=row["train_minutes_std"],
                )
            )

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
