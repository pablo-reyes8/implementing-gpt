from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            return value
        return None

    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return None
    try:
        f = float(text)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def _to_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _normalize_row(row: dict) -> dict:
    row = dict(row)
    float_fields = [
        "train_loss_last",
        "train_ppl_last",
        "val_loss_best",
        "val_ppl_best",
        "elapsed_sec",
    ]
    int_fields = [
        "seed",
        "params_total",
        "params_trainable",
        "global_steps",
    ]

    for field in float_fields:
        if field in row:
            row[field] = _to_float(row[field])
    for field in int_fields:
        if field in row:
            row[field] = _to_int(row[field])
    return row


def load_results(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    rows: list[dict] = []
    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(_normalize_row(json.loads(line)))
        return rows

    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(_normalize_row(row))
        return rows

    raise ValueError("Formato no soportado. Usa .jsonl o .csv")


def infer_metric(rows: list[dict], preferred: tuple[str, ...] = ("val_loss_best", "train_loss_last")) -> str:
    for metric in preferred:
        if any(_to_float(row.get(metric)) is not None for row in rows):
            return metric
    raise ValueError("No hay métricas numéricas válidas en los resultados")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def summarize_by(rows: list[dict], *, group_key: str, metric: str) -> list[dict]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        key = str(row.get(group_key, "unknown"))
        value = _to_float(row.get(metric))
        if value is None:
            continue
        grouped.setdefault(key, []).append(value)

    summary = []
    for key, values in grouped.items():
        summary.append(
            {
                group_key: key,
                "n": len(values),
                "mean": _mean(values),
                "std": _std(values),
                "min": min(values),
                "max": max(values),
            }
        )

    summary.sort(key=lambda x: x["mean"])
    return summary


def save_summary_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib no está instalado. Instálalo con 'pip install matplotlib'.") from exc
    return plt


def plot_compare(
    rows: list[dict],
    *,
    metric: str,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    plt = _ensure_matplotlib()

    summary = summarize_by(rows, group_key="model_version", metric=metric)
    labels = [item["model_version"] for item in summary]
    means = [item["mean"] for item in summary]
    stds = [item["std"] for item in summary]

    if not labels:
        raise ValueError("No hay datos válidos para plot_compare")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=stds, capsize=5)

    for idx, label in enumerate(labels):
        vals = [
            _to_float(r.get(metric))
            for r in rows
            if str(r.get("model_version")) == label and _to_float(r.get(metric)) is not None
        ]
        jitter = [idx + ((i - (len(vals) - 1) / 2) * 0.04) for i in range(len(vals))]
        ax.scatter(jitter, vals, s=18, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Model Comparison ({metric})")
    ax.grid(alpha=0.2, axis="y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_ablation(
    rows: list[dict],
    *,
    metric: str,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    plt = _ensure_matplotlib()

    axis_name = str(rows[0].get("ablation_axis", "ablation"))
    summary = summarize_by(rows, group_key="ablation_value", metric=metric)

    if not summary:
        raise ValueError("No hay datos válidos para plot_ablation")

    def sort_key(item):
        as_float = _to_float(item["ablation_value"])
        if as_float is None:
            return (1, str(item["ablation_value"]))
        return (0, as_float)

    summary.sort(key=sort_key)

    labels = [str(item["ablation_value"]) for item in summary]
    means = [item["mean"] for item in summary]
    stds = [item["std"] for item in summary]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = list(range(len(labels)))
    ax.bar(x, means, yerr=stds, capsize=5)

    for idx, label in enumerate(labels):
        vals = [
            _to_float(r.get(metric))
            for r in rows
            if str(r.get("ablation_value")) == label and _to_float(r.get(metric)) is not None
        ]
        jitter = [idx + ((i - (len(vals) - 1) / 2) * 0.04) for i in range(len(vals))]
        ax.scatter(jitter, vals, s=18, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel(axis_name)
    ax.set_ylabel(metric)
    ax.set_title(title or f"Ablation ({axis_name}) - {metric}")
    ax.grid(alpha=0.2, axis="y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
