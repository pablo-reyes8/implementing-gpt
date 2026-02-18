#!/usr/bin/env python3
"""Plot research results from compare/ablation runs."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.plotting import (
    infer_metric,
    load_results,
    plot_ablation,
    plot_compare,
    save_summary_csv,
    summarize_by,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera plots a partir de results.jsonl/results.csv")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Rutas a archivos de resultados (.jsonl o .csv)",
    )
    parser.add_argument(
        "--kind",
        choices=["auto", "compare", "ablation"],
        default="auto",
        help="Tipo de plot a generar",
    )
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="research_runs/plots")
    parser.add_argument("--plot-name", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)
    return parser.parse_args()


def print_summary(rows: list[dict], group_key: str, metric: str) -> None:
    summary = summarize_by(rows, group_key=group_key, metric=metric)
    print(f"\n=== Summary ({group_key}, {metric}) ===")
    for item in summary:
        print(
            f"{item[group_key]:<24} n={item['n']:>3} "
            f"mean={item['mean']:.4f} std={item['std']:.4f} "
            f"min={item['min']:.4f} max={item['max']:.4f}"
        )


def main() -> None:
    args = parse_args()

    rows: list[dict] = []
    for path in args.results:
        rows.extend(load_results(path))

    if not rows:
        raise ValueError("No se encontraron filas de resultados")

    metric = args.metric or infer_metric(rows)

    inferred_kind = args.kind
    if inferred_kind == "auto":
        if any(row.get("ablation_axis") for row in rows):
            inferred_kind = "ablation"
        else:
            inferred_kind = "compare"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.plot_name = f"{inferred_kind}_{metric}_{ts}.png"

    plot_path = output_dir / args.plot_name

    if inferred_kind == "ablation":
        plot_ablation(rows, metric=metric, output_path=plot_path, title=args.title)
        print_summary(rows, group_key="ablation_value", metric=metric)
        group_key = "ablation_value"
    else:
        plot_compare(rows, metric=metric, output_path=plot_path, title=args.title)
        print_summary(rows, group_key="model_version", metric=metric)
        group_key = "model_version"

    print(f"\nPlot guardado en: {plot_path}")

    summary_rows = summarize_by(rows, group_key=group_key, metric=metric)
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_dir / f"summary_{inferred_kind}_{metric}.csv"
    save_summary_csv(summary_csv, summary_rows)
    print(f"Resumen CSV: {summary_csv}")


if __name__ == "__main__":
    main()
