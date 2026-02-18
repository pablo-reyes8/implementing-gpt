"""Airflow DAG to orchestrate the reproducible GPT research pipeline."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs/pipeline/research_small_repro.yaml"
PIPELINE_CONFIG = Path(os.getenv("GPT_PIPELINE_CONFIG", str(DEFAULT_CONFIG)))
PYTHON_BIN = os.getenv("GPT_PYTHON_BIN", "python")


def _pipeline_cmd(step_id: str | None = None, dry_run: bool = False) -> str:
    parts = [
        f"cd {REPO_ROOT}",
        f"{PYTHON_BIN} scripts/run_pipeline_config.py --config {PIPELINE_CONFIG}",
    ]
    if step_id:
        parts[-1] += f" --step {step_id}"
    if dry_run:
        parts[-1] += " --dry-run"
    return " && ".join(parts)


with DAG(
    dag_id="gpt_research_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["gpt", "research", "reproducible"],
) as dag:
    if not PIPELINE_CONFIG.exists():
        missing_config = EmptyOperator(task_id="missing_pipeline_config")
    else:
        validate = BashOperator(
            task_id="validate_pipeline_config",
            bash_command=_pipeline_cmd(dry_run=True),
        )

        compare = BashOperator(
            task_id="run_compare",
            bash_command=_pipeline_cmd(step_id="compare"),
        )

        ablation = BashOperator(
            task_id="run_ablation",
            bash_command=_pipeline_cmd(step_id="ablation"),
        )

        plot_compare = BashOperator(
            task_id="plot_compare",
            bash_command=_pipeline_cmd(step_id="plot_compare"),
        )

        plot_ablation = BashOperator(
            task_id="plot_ablation",
            bash_command=_pipeline_cmd(step_id="plot_ablation"),
        )

        validate >> compare >> ablation >> [plot_compare, plot_ablation]
