from __future__ import annotations

import os
from pathlib import Path

from src.data.hf_text_corpus import create_hf_causal_dataloaders

DATASET_NAME = "monology/pile-uncopyrighted"
DATASET_CONFIG = None
TEXT_FIELD = "text"

VOCAB_SIZE = 50000
MIN_FREQ = 2
BLOCK_SIZE = 256
TOKENIZER_PATH = Path("pile_unc_tokenizer.json")

CPU_COUNT = os.cpu_count() or 2
BATCH_SIZE = 64
NUM_WORKERS = 2 if CPU_COUNT <= 2 else min(4, CPU_COUNT - 1)


def create_dataloaders(
    block_size: int = BLOCK_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    return create_hf_causal_dataloaders(
        dataset_name=DATASET_NAME,
        dataset_config=DATASET_CONFIG,
        text_field=TEXT_FIELD,
        tokenizer_path=TOKENIZER_PATH,
        vocab_size=VOCAB_SIZE,
        min_freq=MIN_FREQ,
        block_size=block_size,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split="train",
        val_split=None,
        tokenizer_train_docs=1_000_000,
        max_train_docs=2_000_000,
        max_val_docs=200_000,
    )
