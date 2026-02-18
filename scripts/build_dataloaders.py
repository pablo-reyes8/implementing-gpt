#!/usr/bin/env python3
"""Build and inspect dataloaders for any registered dataset."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import DATASET_LOADERS, dataset_choices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construye dataloaders para un dataset registrado")
    parser.add_argument("--dataset", choices=dataset_choices(), required=True)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--preview-batches", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    module = importlib.import_module(DATASET_LOADERS[args.dataset])
    train_loader, val_loader, tokenizer = module.create_dataloaders(
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size():,}")
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches:   {len(val_loader):,}")

    for idx, (x, y) in enumerate(train_loader):
        print(f"Batch {idx}: x={tuple(x.shape)} y={tuple(y.shape)}")
        if idx + 1 >= args.preview_batches:
            break


if __name__ == "__main__":
    main()
