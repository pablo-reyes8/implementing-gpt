from pathlib import Path

from src.research.pipeline_config import build_cli_args, run_pipeline_config


def test_build_cli_args_supports_bool_list_and_none():
    args = {
        "dataset": "small",
        "amp": False,
        "val-checking": True,
        "models": ["gpt2", "gpt3"],
        "unused": None,
    }
    cli = build_cli_args(args)
    assert "--dataset" in cli
    assert "small" in cli
    assert "--val-checking" in cli
    assert "--models" in cli
    assert "gpt2" in cli and "gpt3" in cli
    assert "--amp" not in cli
    assert "--unused" not in cli


def test_run_pipeline_config_dry_run_with_step_config(tmp_path: Path):
    project_root = tmp_path

    dummy_script = project_root / "scripts" / "dummy.py"
    dummy_script.parent.mkdir(parents=True, exist_ok=True)
    dummy_script.write_text("print('ok')\n", encoding="utf-8")

    stage_cfg = project_root / "configs" / "stage.yaml"
    stage_cfg.parent.mkdir(parents=True, exist_ok=True)
    stage_cfg.write_text(
        """
name: dummy
script: scripts/dummy.py
args:
  foo: bar
  flag: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    pipeline_cfg = project_root / "configs" / "pipeline.yaml"
    pipeline_cfg.write_text(
        """
name: test_pipeline
project_root: .
run_dir: runs/test_pipeline
steps:
  - id: s1
    enabled: true
    config: configs/stage.yaml
""".strip()
        + "\n",
        encoding="utf-8",
    )

    report = run_pipeline_config(pipeline_cfg, dry_run=True)
    assert report["dry_run"] is True
    assert report["results"][0]["step_id"] == "s1"
    assert report["results"][0]["status"] == "dry_run"
    assert Path(report["summary_path"]).exists()
