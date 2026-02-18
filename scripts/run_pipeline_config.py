#!/usr/bin/env python3
"""Run a reproducible multi-step research pipeline from YAML configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.pipeline_config import PipelineConfigError, run_pipeline_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta pipeline research desde YAML")
    parser.add_argument("--config", required=True, type=str, help="Ruta al YAML de pipeline")
    parser.add_argument(
        "--step",
        action="append",
        default=None,
        help="Ejecuta solo este step-id. Puede repetirse varias veces.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Solo imprime comandos, no ejecuta")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continúa con siguientes pasos aunque falle uno",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        report = run_pipeline_config(
            args.config,
            selected_steps=args.step,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error,
        )
    except PipelineConfigError as exc:
        print(f"ERROR: {exc}")
        return 2

    print(f"\nSummary JSON: {report['summary_path']}")
    failed = [r for r in report["results"] if r["status"] == "failed"]
    if failed:
        print(f"Pipeline finalizó con {len(failed)} step(s) fallidos")
        return 1

    print("Pipeline finalizado correctamente")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
