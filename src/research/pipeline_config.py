from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml


@dataclass
class StepResult:
    step_id: str
    command: list[str]
    returncode: int | None
    status: str
    started_at: str | None = None
    finished_at: str | None = None


class PipelineConfigError(RuntimeError):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise PipelineConfigError(f"Config no encontrada: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise PipelineConfigError(f"Config inválida (debe ser mapping): {path}")
    return data


def resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_cli_args(args: dict | None) -> list[str]:
    if not args:
        return []

    cli: list[str] = []
    for key, value in args.items():
        flag = f"--{key}"

        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if isinstance(value, list):
            cli.append(flag)
            cli.extend(str(item) for item in value)
            continue

        cli.extend([flag, str(value)])

    return cli


def _script_command(script_path: Path, args: dict | None) -> list[str]:
    return [sys.executable, str(script_path), *build_cli_args(args)]


def _load_step_payload(step: dict, *, project_root: Path) -> tuple[str, Path, dict | None]:
    step_id = str(step.get("id") or "unnamed")

    if "config" in step:
        stage_config_path = resolve_path(step["config"], base_dir=project_root)
        stage_cfg = load_yaml(stage_config_path)
        script_rel = stage_cfg.get("script")
        if not script_rel:
            raise PipelineConfigError(f"Step '{step_id}' sin 'script' en config {stage_config_path}")
        args = stage_cfg.get("args", {})
        script_path = resolve_path(script_rel, base_dir=project_root)
        if not script_path.exists():
            raise PipelineConfigError(f"Script no encontrado para step '{step_id}': {script_path}")
        return step_id, script_path, args

    script_rel = step.get("script")
    if not script_rel:
        raise PipelineConfigError(f"Step '{step_id}' requiere 'script' o 'config'")
    args = step.get("args", {})
    script_path = resolve_path(script_rel, base_dir=project_root)
    if not script_path.exists():
        raise PipelineConfigError(f"Script no encontrado para step '{step_id}': {script_path}")
    return step_id, script_path, args


def run_pipeline_config(
    config_path: str | Path,
    *,
    selected_steps: list[str] | None = None,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> dict:
    config_path = Path(config_path).resolve()
    cfg = load_yaml(config_path)
    raw_project_root = cfg.get("project_root")
    if raw_project_root is None:
        project_root = Path.cwd().resolve()
    else:
        candidate_root = Path(raw_project_root)
        if candidate_root.is_absolute():
            project_root = candidate_root
        else:
            project_root = (Path.cwd() / candidate_root).resolve()

    run_name = str(cfg.get("name", "pipeline_run"))
    default_run_dir = f"research_runs/pipeline/{run_name}"
    run_dir = resolve_path(cfg.get("run_dir", default_run_dir), base_dir=project_root)
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    for key, value in (cfg.get("env", {}) or {}).items():
        env[str(key)] = str(value)

    steps = cfg.get("steps", [])
    if not isinstance(steps, list):
        raise PipelineConfigError("'steps' debe ser una lista")

    if selected_steps:
        selected_set = set(selected_steps)
        available = {str(step.get("id")) for step in steps}
        missing = sorted(selected_set - available)
        if missing:
            raise PipelineConfigError(f"Steps no encontrados: {missing}")
        steps = [step for step in steps if str(step.get("id")) in selected_set]

    results: list[StepResult] = []
    for step in steps:
        step_id = str(step.get("id") or "unnamed")
        if not bool(step.get("enabled", True)):
            results.append(
                StepResult(
                    step_id=step_id,
                    command=[],
                    returncode=None,
                    status="skipped",
                )
            )
            continue

        _, script_path, step_args = _load_step_payload(step, project_root=project_root)
        cmd = _script_command(script_path, step_args)

        started_at = utc_now_iso()
        print(f"\n>>> [{step_id}] {' '.join(cmd)}")

        if dry_run:
            results.append(
                StepResult(
                    step_id=step_id,
                    command=cmd,
                    returncode=0,
                    status="dry_run",
                    started_at=started_at,
                    finished_at=utc_now_iso(),
                )
            )
            continue

        proc = subprocess.run(cmd, cwd=project_root, env=env, check=False)
        finished_at = utc_now_iso()

        status = "ok" if proc.returncode == 0 else "failed"
        results.append(
            StepResult(
                step_id=step_id,
                command=cmd,
                returncode=proc.returncode,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
            )
        )

        if proc.returncode != 0 and not continue_on_error:
            break

    payload = {
        "pipeline_config": str(config_path),
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "run_name": run_name,
        "dry_run": dry_run,
        "results": [result.__dict__ for result in results],
        "finished_at": utc_now_iso(),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = run_dir / f"execution_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    payload["summary_path"] = str(summary_path)
    return payload
