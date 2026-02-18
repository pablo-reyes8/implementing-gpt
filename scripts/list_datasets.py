#!/usr/bin/env python3
"""Print available dataset loader options for GPT experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import DATASET_CATALOG, dataset_choices


def main() -> None:
    print("Datasets disponibles en el CLI:\n")
    for key in dataset_choices():
        item = DATASET_CATALOG[key]
        print(f"- {key}")
        print(f"  title : {item['title']}")
        print(f"  source: {item['source']}")
        print(f"  loader: {item['loader']}")
        print()

    print("Para 'local_jsonl' configura estas variables de entorno:")
    print("  LOCAL_TEXT_DATA_PATH=/ruta/a/corpus.jsonl")
    print("  LOCAL_TEXT_DATA_FORMAT=json|text")
    print("  LOCAL_TEXT_FIELD=text   # solo para JSON")


if __name__ == "__main__":
    main()
