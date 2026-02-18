from __future__ import annotations

import os
from pathlib import Path

from src.data.hf_text_corpus import create_hf_causal_dataloaders

# Ejemplos:
#   export LOCAL_TEXT_DATA_PATH=/ruta/a/corpus.jsonl
#   export LOCAL_TEXT_DATA_FORMAT=json
#   export LOCAL_TEXT_FIELD=text
# o para txt:
#   export LOCAL_TEXT_DATA_FORMAT=text

LOCAL_TEXT_DATA_PATH = os.getenv("LOCAL_TEXT_DATA_PATH", "data/local_corpus.jsonl")
LOCAL_TEXT_DATA_FORMAT = os.getenv("LOCAL_TEXT_DATA_FORMAT", "json")
LOCAL_TEXT_FIELD = os.getenv("LOCAL_TEXT_FIELD", "text")

VOCAB_SIZE = 32000
MIN_FREQ = 2
BLOCK_SIZE = 256
TOKENIZER_PATH = Path("local_corpus_tokenizer.json")

CPU_COUNT = os.cpu_count() or 2
BATCH_SIZE = 64
NUM_WORKERS = 2 if CPU_COUNT <= 2 else min(4, CPU_COUNT - 1)


def _dataset_spec():
    fmt = LOCAL_TEXT_DATA_FORMAT.lower()
    if fmt not in {"json", "text"}:
        raise ValueError("LOCAL_TEXT_DATA_FORMAT debe ser 'json' o 'text'")
    return fmt, LOCAL_TEXT_DATA_PATH


def create_dataloaders(
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    dataset_format, data_path = _dataset_spec()
    data_files = {"train": data_path}

    # Usamos el builder de HF para archivos locales:
    # load_dataset("json", data_files=...) o load_dataset("text", data_files=...)
    return create_hf_causal_dataloaders(
        dataset_name=dataset_format,
        dataset_config=None,
        text_field=LOCAL_TEXT_FIELD if dataset_format == "json" else "text",
        tokenizer_path=TOKENIZER_PATH,
        vocab_size=VOCAB_SIZE,
        min_freq=MIN_FREQ,
        block_size=block_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split="train",
        val_split=None,
        tokenizer_train_docs=500_000,
        max_train_docs=2_000_000,
        max_val_docs=200_000,
        load_dataset_kwargs={"data_files": data_files},
    )
