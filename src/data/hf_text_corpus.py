"""Reusable Hugging Face text-corpus loader for GPT-style LM training."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset as TorchDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _safe_select(ds: Dataset, max_docs: int | None) -> Dataset:
    if max_docs is None:
        return ds
    if max_docs <= 0:
        raise ValueError("max_docs debe ser mayor a 0")
    max_docs = min(max_docs, len(ds))
    return ds.select(range(max_docs))


def _load_dataset_dict(
    dataset_name: str,
    dataset_config: str | None = None,
    *,
    load_dataset_kwargs: dict | None = None,
) -> DatasetDict:
    kwargs = dict(load_dataset_kwargs or {})
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, **kwargs)
    return load_dataset(dataset_name, **kwargs)


def _resolve_train_val_splits(
    ds_dict: DatasetDict,
    *,
    train_split: str,
    val_split: str | None,
    val_fraction: float,
    split_seed: int,
) -> tuple[Dataset, Dataset]:
    if train_split not in ds_dict:
        raise ValueError(
            f"Split de train '{train_split}' no encontrado. Disponibles: {list(ds_dict.keys())}"
        )

    train_ds = ds_dict[train_split]

    candidates = []
    if val_split:
        candidates.append(val_split)
    candidates.extend(["validation", "val", "test"])

    for split_name in candidates:
        if split_name in ds_dict and split_name != train_split:
            return train_ds, ds_dict[split_name]

    split = train_ds.train_test_split(test_size=val_fraction, seed=split_seed)
    return split["train"], split["test"]


def train_tokenizer(
    train_ds: Dataset,
    *,
    text_field: str,
    vocab_size: int,
    min_freq: int,
    save_path: Path,
    lowercase: bool,
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    norm_ops = [normalizers.NFKC()]
    if lowercase:
        norm_ops.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(norm_ops)

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,
    )

    def batch_iterator():
        for ex in train_ds:
            txt = ex.get(text_field, "")
            if txt is not None and txt.strip():
                yield txt

    print(f"Entrenando tokenizer BPE en '{save_path}'...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    print("Tamaño vocabulario:", tokenizer.get_vocab_size())

    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    print(f"Tokenizer guardado en {save_path.resolve()}")

    return tokenizer


def load_or_train_tokenizer(
    train_ds: Dataset,
    *,
    text_field: str,
    tokenizer_path: Path,
    vocab_size: int,
    min_freq: int,
    lowercase: bool,
) -> Tokenizer:
    if tokenizer_path.exists():
        print(f"Cargando tokenizer desde {tokenizer_path}...")
        return Tokenizer.from_file(str(tokenizer_path))

    return train_tokenizer(
        train_ds,
        text_field=text_field,
        vocab_size=vocab_size,
        min_freq=min_freq,
        save_path=tokenizer_path,
        lowercase=lowercase,
    )


class GPTTextDataset(TorchDataset):
    """
    Dataset autoregresivo:
      - concatena documentos en un stream largo de IDs
      - agrega <eos>
      - crea chunks de longitud block_size+1
      - input = ids[:-1], target = ids[1:]
    """

    def __init__(
        self,
        hf_split: Dataset,
        tokenizer: Tokenizer,
        *,
        text_field: str,
        block_size: int,
        max_docs: int | None,
    ):
        super().__init__()

        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is None:
            raise ValueError("El tokenizer no tiene token <eos>.")

        hf_split = _safe_select(hf_split, max_docs)

        all_ids: list[int] = []
        print("Tokenizando y concatenando textos...")
        for ex in hf_split:
            txt = ex.get(text_field, "")
            if txt is None or not txt.strip():
                continue
            enc = tokenizer.encode(txt)
            all_ids.extend(enc.ids + [eos_id])

        self.data = torch.tensor(all_ids, dtype=torch.long)
        print(f"Total de tokens en este split: {len(self.data):,}")

        chunk_len = block_size + 1
        n_chunks = len(self.data) // chunk_len
        if n_chunks == 0:
            raise ValueError("Muy pocos tokens para formar un chunk. Baja block_size o usa más datos.")

        self.data = self.data[: n_chunks * chunk_len].view(n_chunks, chunk_len)
        self.inputs = self.data[:, :-1]
        self.targets = self.data[:, 1:]

        print(f"Número de secuencias: {len(self.inputs):,}")
        print(f"Forma inputs:  {self.inputs.shape}")
        print(f"Forma targets: {self.targets.shape}")

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_hf_causal_dataloaders(
    *,
    dataset_name: str,
    dataset_config: str | None,
    text_field: str,
    tokenizer_path: Path,
    vocab_size: int,
    min_freq: int,
    block_size: int,
    batch_size: int,
    num_workers: int,
    train_split: str = "train",
    val_split: str | None = "validation",
    val_fraction: float = 0.1,
    split_seed: int = 42,
    lowercase: bool = True,
    max_train_docs: int | None = None,
    max_val_docs: int | None = None,
    tokenizer_train_docs: int | None = None,
    load_dataset_kwargs: dict | None = None,
):
    """Factory genérica para dataloaders autoregresivos desde datasets de texto de HF."""

    ds_dict = _load_dataset_dict(
        dataset_name,
        dataset_config,
        load_dataset_kwargs=load_dataset_kwargs,
    )
    train_hf, val_hf = _resolve_train_val_splits(
        ds_dict,
        train_split=train_split,
        val_split=val_split,
        val_fraction=val_fraction,
        split_seed=split_seed,
    )

    tokenizer_ds = _safe_select(train_hf, tokenizer_train_docs)
    tokenizer = load_or_train_tokenizer(
        tokenizer_ds,
        text_field=text_field,
        tokenizer_path=tokenizer_path,
        vocab_size=vocab_size,
        min_freq=min_freq,
        lowercase=lowercase,
    )

    train_ds = GPTTextDataset(
        train_hf,
        tokenizer,
        text_field=text_field,
        block_size=block_size,
        max_docs=max_train_docs,
    )
    val_ds = GPTTextDataset(
        val_hf,
        tokenizer,
        text_field=text_field,
        block_size=block_size,
        max_docs=max_val_docs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer
