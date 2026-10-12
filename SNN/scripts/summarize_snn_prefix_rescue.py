from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASELINE_NAMES = {
    64: "shallow_temporal_lif_distill_k64",
    80: "shallow_temporal_lif_distill_k80",
}
PREFIXES = (64, 80)
K80_GATE = 0.80


def _metric(payload: dict[str, Any], key: str) -> float:
    if "test_metrics" in payload:
        return float(payload["test_metrics"][key])
    return float(payload[key])


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else float("nan")


def _std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def _read_full_window_reference(results_dir: Path) -> dict[str, float]:
    refs = {
        "fold1_accuracy": 0.8215767634854771,
        "fivefold_accuracy": 0.8255862959429585,
    }
    fold1_path = results_dir / "v2_hybrid" / "shallow_temporal_lif_head" / "fold_1.json"
    if fold1_path.exists():
        with fold1_path.open("r", encoding="utf-8") as handle:
            refs["fold1_accuracy"] = _metric(json.load(handle), "accuracy")

    summary_path = results_dir / "summaries" / "snn_v2_summary.csv"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get("section") == "v2_hybrid"
                    and row.get("config") == "shallow_temporal_lif_head"
                    and row.get("head") == "supervised"
                ):
                    refs["fivefold_accuracy"] = float(row["accuracy_mean"])
                    break
    return refs


def _collect_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((results_dir / "v4_prefix_distill").glob("*/*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        prefix = int(payload["prefix_steps"])
        if prefix not in PREFIXES:
            continue
        config = payload.get("config", {})
        distill = config.get("distillation", {})
        model = config.get("model", {})
        metrics = payload["test_metrics"]
        rows.append(
            {
                "name": str(payload["name"]),
                "fold": int(payload["fold"]),
                "prefix_steps": prefix,
                "alpha": float(payload.get("alpha", distill.get("alpha", 0.5))),
                "temperature": float(payload.get("temperature", distill.get("temperature", 2.0))),
                "dropout": float(model.get("dropout", 0.30)),
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "spike_rate": float(metrics["spike_rate"]),
                "best_epoch": int(payload["best_epoch"]),
                "seconds": float(payload["seconds"]),
                "path": str(path),
            }
        )
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["name"], []).append(row)

    out = []
    for name, group in sorted(grouped.items()):
        group = sorted(group, key=lambda item: item["fold"])
        out.append(
            {
                "name": name,
                "prefix_steps": group[0]["prefix_steps"],
                "alpha": group[0]["alpha"],
                "temperature": group[0]["temperature"],
                "dropout": group[0]["dropout"],
                "folds": len(group),
                "fold_list": " ".join(str(row["fold"]) for row in group),
                "accuracy_mean": _mean([row["accuracy"] for row in group]),
                "accuracy_std": _std([row["accuracy"] for row in group]),
                "macro_f1_mean": _mean([row["macro_f1"] for row in group]),
                "macro_f1_std": _std([row["macro_f1"] for row in group]),
                "balanced_accuracy_mean": _mean([row["balanced_accuracy"] for row in group]),
                "balanced_accuracy_std": _std([row["balanced_accuracy"] for row in group]),
                "spike_rate_mean": _mean([row["spike_rate"] for row in group]),
                "best_epoch_mean": _mean([float(row["best_epoch"]) for row in group]),
                "seconds_mean": _mean([row["seconds"] for row in group]),
            }
        )
    return sorted(out, key=lambda item: (item["prefix_steps"], -item["accuracy_mean"], -item["macro_f1_mean"]))


def _best_by_prefix(rows: list[dict[str, Any]], *, fold: int = 1) -> dict[int, dict[str, Any]]:
    best = {}
    for prefix in PREFIXES:
        candidates = [row for row in rows if row["prefix_steps"] == prefix and row["fold"] == fold]
        if not candidates:
            continue
        best[prefix] = max(candidates, key=lambda row: (row["accuracy"], row["macro_f1"]))
    return best


def _baseline_by_prefix(rows: list[dict[str, Any]], *, fold: int = 1) -> dict[int, dict[str, Any]]:
    out = {}
    for prefix, name in BASELINE_NAMES.items():
        matches = [row for row in rows if row["name"] == name and row["fold"] == fold]
        if matches:
            out[prefix] = matches[0]
    return out


def _gate_status(prefix: int, accuracy: float, full_window_accuracy: float) -> tuple[str, float, float]:
    if prefix == 64:
        threshold = full_window_accuracy - 0.02
        return ("pass" if accuracy >= threshold else "fail", threshold, accuracy - threshold)
    threshold = K80_GATE
    return ("pass" if accuracy >= threshold else "fail", threshold, accuracy - threshold)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(
    *,
    rows: list[dict[str, Any]],
    grouped: list[dict[str, Any]],
    refs: dict[str, float],
    path: Path,
) -> None:
    fold1_best = _best_by_prefix(rows)
    fold1_baseline = _baseline_by_prefix(rows)
    complete = [
        row for row in grouped
        if row["folds"] >= 5 and set(row["fold_list"].split()) >= {"1", "2", "3", "4", "5"}
    ]

    lines = [
        "# SNN Prefix Rescue Summary",
        "",
        "Promotion gates:",
        "",
        f"- `K64`: five-fold mean accuracy >= full-window minus 2 pp = {refs['fivefold_accuracy'] - 0.02:.4f}.",
        f"- `K80`: five-fold mean accuracy >= {K80_GATE:.4f}.",
        "",
        "## Fold-1 Winners",
        "",
        "| Prefix | Best config | Acc | Macro-F1 | Delta vs baseline | Fold-1 gate | Gate margin |",
        "|---:|---|---:|---:|---:|---|---:|",
    ]
    for prefix in PREFIXES:
        best = fold1_best.get(prefix)
        if not best:
            continue
        baseline = fold1_baseline.get(prefix)
        delta = best["accuracy"] - baseline["accuracy"] if baseline else float("nan")
        status, threshold, margin = _gate_status(prefix, best["accuracy"], refs["fold1_accuracy"])
        lines.append(
            f"| {prefix} | `{best['name']}` | {best['accuracy']:.4f} | {best['macro_f1']:.4f} | "
            f"{delta:+.4f} | {status} vs {threshold:.4f} | {margin:+.4f} |"
        )

    lines.extend([
        "",
        "## Five-Fold Gate Status",
        "",
        "| Prefix | Config | Folds | Acc mean | Acc std | Macro-F1 mean | Gate | Margin |",
        "|---:|---|---:|---:|---:|---:|---|---:|",
    ])
    if complete:
        for row in sorted(complete, key=lambda item: (item["prefix_steps"], -item["accuracy_mean"])):
            status, threshold, margin = _gate_status(
                int(row["prefix_steps"]),
                float(row["accuracy_mean"]),
                refs["fivefold_accuracy"],
            )
            lines.append(
                f"| {row['prefix_steps']} | `{row['name']}` | {row['folds']} | "
                f"{row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} | "
                f"{row['macro_f1_mean']:.4f} | {status} vs {threshold:.4f} | {margin:+.4f} |"
            )
    else:
        lines.append("| - | - | 0 | - | - | - | pending 5-fold | - |")

    lines.extend([
        "",
        "## All Rescue Rows",
        "",
        "| Prefix | Config | Folds | Alpha | Dropout | Acc mean | Macro-F1 mean | Spike rate | Folds present |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in grouped:
        lines.append(
            f"| {row['prefix_steps']} | `{row['name']}` | {row['folds']} | "
            f"{row['alpha']:.2f} | {row['dropout']:.2f} | {row['accuracy_mean']:.4f} | "
            f"{row['macro_f1_mean']:.4f} | {row['spike_rate_mean']:.4f} | {row['fold_list']} |"
        )

    lines.extend([
        "",
        "Recommended five-fold configs from fold-1 selection:",
    ])
    for prefix in PREFIXES:
        best = fold1_best.get(prefix)
        if best:
            lines.append(f"- `K{prefix}`: `{best['name']}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SNN prefix rescue results and gates.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = _collect_rows(results_dir)
    grouped = _group_rows(rows)
    refs = _read_full_window_reference(results_dir)
    out_dir = results_dir / "summaries"
    _write_csv(grouped, out_dir / "snn_prefix_rescue_summary.csv")
    _write_markdown(rows=rows, grouped=grouped, refs=refs, path=out_dir / "snn_prefix_rescue_summary.md")
    print(f"Saved {out_dir / 'snn_prefix_rescue_summary.csv'}")
    print(f"Saved {out_dir / 'snn_prefix_rescue_summary.md'}")


if __name__ == "__main__":
    main()
