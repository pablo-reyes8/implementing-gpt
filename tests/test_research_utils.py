import json

from src.data.dataset_registry import DATASET_LOADERS
from src.research.plotting import infer_metric, load_results, summarize_by


def test_dataset_registry_contains_expected_keys():
    expected = {"small", "large", "wikitext2", "tinystories", "openwebtext", "c4_en", "pile", "redpajama", "local_jsonl"}
    assert expected.issubset(set(DATASET_LOADERS.keys()))


def test_plotting_helpers_jsonl_roundtrip(tmp_path):
    rows = [
        {"model_version": "gpt2", "val_loss_best": 2.1},
        {"model_version": "gpt3", "val_loss_best": 1.9},
        {"model_version": "gpt3", "val_loss_best": 2.0},
    ]
    path = tmp_path / "results.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    loaded = load_results(path)
    metric = infer_metric(loaded)
    summary = summarize_by(loaded, group_key="model_version", metric=metric)

    assert metric == "val_loss_best"
    assert len(summary) == 2
    assert summary[0]["mean"] <= summary[1]["mean"]
