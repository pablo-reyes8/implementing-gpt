# Airflow Orchestration (Simple)

This folder provides a lightweight Airflow DAG to run the reproducible GPT research pipeline defined in YAML configs.

## Files

- `orchestration/airflow/dags/gpt_research_pipeline.py`: DAG with compare -> ablation -> plotting flow.
- `configs/pipeline/research_small_repro.yaml`: pipeline definition used by default.
- `scripts/run_pipeline_config.py`: config runner called by the DAG.

## Local Usage (without Airflow)

```bash
python scripts/run_pipeline_config.py --config configs/pipeline/research_small_repro.yaml --dry-run
python scripts/run_pipeline_config.py --config configs/pipeline/research_small_repro.yaml
```

## Airflow Usage

Set variables before launching scheduler/webserver:

```bash
export AIRFLOW_HOME=$PWD/.airflow
export PYTHONPATH=$PWD
export GPT_PIPELINE_CONFIG=$PWD/configs/pipeline/research_small_repro.yaml
```

Then run your normal Airflow bootstrap and trigger DAG `gpt_research_pipeline`.

## Notes

- The DAG intentionally calls existing project CLIs, so behavior stays aligned with local runs.
- For GPU runs on workers, set `GPT_PYTHON_BIN` and your CUDA env vars in Airflow.
