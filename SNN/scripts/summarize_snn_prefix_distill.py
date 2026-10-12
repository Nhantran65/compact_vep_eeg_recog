from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _collect_rows(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((results_dir / "v4_prefix_distill").glob("*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        metrics = result["test_metrics"]
        rows.append(
            {
                "name": result["name"],
                "fold": result["fold"],
                "prefix_steps": result["prefix_steps"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "spike_rate": metrics["spike_rate"],
                "best_epoch": result["best_epoch"],
                "seconds": result["seconds"],
                "path": str(path),
            }
        )
    return rows


def _write_markdown(rows: list[dict], path: Path) -> None:
    lines = [
        "# SNN Prefix Distillation Summary",
        "",
        "| Config | Fold | Prefix | Accuracy | Macro-F1 | Balanced acc | Spike rate | Best epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {name} | {fold} | {prefix_steps} | {accuracy:.4f} | {macro_f1:.4f} | "
            "{balanced_accuracy:.4f} | {spike_rate:.6f} | {best_epoch} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SNN prefix distillation results.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = _collect_rows(results_dir)
    out_dir = results_dir / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "snn_prefix_distill_summary.csv"
    md_path = out_dir / "snn_prefix_distill_summary.md"

    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    _write_markdown(rows, md_path)
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
