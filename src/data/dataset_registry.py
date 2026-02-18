"""Dataset registry for training/benchmark CLIs."""

from __future__ import annotations


DATASET_CATALOG: dict[str, dict[str, str]] = {
    "small": {
        "loader": "src.data.load_small_data",
        "title": "OpenWebText-10K (quick baseline)",
        "source": "Hugging Face: Ankursingh/openwebtext_10K",
    },
    "large": {
        "loader": "src.data.load_large_data",
        "title": "WikiText-103",
        "source": "Hugging Face: wikitext/wikitext-103-raw-v1",
    },
    "wikitext2": {
        "loader": "src.data.load_wikitext2_data",
        "title": "WikiText-2",
        "source": "Hugging Face: wikitext/wikitext-2-raw-v1",
    },
    "tinystories": {
        "loader": "src.data.load_tinystories_data",
        "title": "TinyStories",
        "source": "Hugging Face: roneneldan/TinyStories",
    },
    "openwebtext": {
        "loader": "src.data.load_openwebtext_data",
        "title": "OpenWebText (full)",
        "source": "Hugging Face: Skylion007/openwebtext",
    },
    "c4_en": {
        "loader": "src.data.load_c4_data",
        "title": "C4 English",
        "source": "Hugging Face: allenai/c4 (en)",
    },
    "pile": {
        "loader": "src.data.load_pile_data",
        "title": "The Pile (uncopyrighted)",
        "source": "Hugging Face: monology/pile-uncopyrighted",
    },
    "redpajama": {
        "loader": "src.data.load_redpajama_data",
        "title": "RedPajama 1T Sample",
        "source": "Hugging Face: togethercomputer/RedPajama-Data-1T-Sample",
    },
    "local_jsonl": {
        "loader": "src.data.load_local_jsonl_data",
        "title": "Local JSONL/TXT corpus",
        "source": "Local files (JSONL/TXT) from any source",
    },
}


DATASET_LOADERS = {key: value["loader"] for key, value in DATASET_CATALOG.items()}


def dataset_choices() -> list[str]:
    return sorted(DATASET_LOADERS.keys())


def dataset_help_lines() -> str:
    lines = []
    for key in dataset_choices():
        item = DATASET_CATALOG[key]
        lines.append(f"- {key}: {item['title']} | {item['source']}")
    return "\n".join(lines)
