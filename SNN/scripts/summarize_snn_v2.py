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
    for section in ("v2_controls", "v2_hybrid"):
        for path in sorted((results_dir / section).glob("*/*.json")):
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if "heads" in payload:
                for head in payload["heads"]:
                    rows.append(
                        {
                            "section": section,
                            "config": payload["name"],
                            "head": head["name"],
                            "fold": payload["fold"],
                            "accuracy": head["test_metrics"]["accuracy"],
                            "macro_f1": head["test_metrics"]["macro_f1"],
                            "balanced_accuracy": head["test_metrics"]["balanced_accuracy"],
                            "path": str(path),
                        }
                    )
            elif "test_metrics" in payload:
                rows.append(
                    {
                        "section": section,
                        "config": payload["name"],
                        "head": "supervised",
                        "fold": payload["fold"],
                        "accuracy": payload["test_metrics"]["accuracy"],
                        "macro_f1": payload["test_metrics"]["macro_f1"],
                        "balanced_accuracy": payload["test_metrics"]["balanced_accuracy"],
                        "path": str(path),
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SNN V2 controls and hybrid results.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    rows = collect_rows(results_dir)
    if not rows:
        print("No SNN V2 result JSON files found.")
        return

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["section"], row["config"], row["head"])].append(row)
    summaries = []
    for (section, config, head), group in sorted(grouped.items()):
        acc_mean, acc_std = mean_metric(group, "accuracy")
        f1_mean, f1_std = mean_metric(group, "macro_f1")
        bal_mean, bal_std = mean_metric(group, "balanced_accuracy")
        summaries.append(
            {
                "section": section,
                "config": config,
                "head": head,
                "folds": len(group),
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "macro_f1_mean": f1_mean,
                "macro_f1_std": f1_std,
                "balanced_accuracy_mean": bal_mean,
                "balanced_accuracy_std": bal_std,
            }
        )

    out_dir = results_dir / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "snn_v2_summary.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    md_path = out_dir / "snn_v2_summary.md"
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# SNN V2/V5a/V6a Summary\n\n")
        handle.write("| Section | Config | Head | Folds | Accuracy | Macro-F1 | Balanced acc |\n")
        handle.write("|---|---|---|---:|---:|---:|---:|\n")
        for row in summaries:
            handle.write(
                "| {section} | {config} | {head} | {folds} | {acc} | {f1} | {bal} |\n".format(
                    section=row["section"],
                    config=row["config"],
                    head=row["head"],
                    folds=row["folds"],
                    acc=format_percent(row["accuracy_mean"], row["accuracy_std"]),
                    f1=format_percent(row["macro_f1_mean"], row["macro_f1_std"]),
                    bal=format_percent(
                        row["balanced_accuracy_mean"],
                        row["balanced_accuracy_std"],
                    ),
                )
            )

    print(f"Wrote {out_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
