from __future__ import annotations

import csv
import importlib
import json
import random
import time
from pathlib import Path

import torch

from src.model.gpt_model import GPT2, GPT3
from src.training.main_loop import train_gpt_lm

DATASET_LOADERS = {
    "small": "src.data.load_small_data",
    "large": "src.data.load_large_data",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataloaders(dataset_name: str, block_size: int, batch_size: int, num_workers: int):
    module = importlib.import_module(DATASET_LOADERS[dataset_name])
    return module.create_dataloaders(
        block_size=block_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def build_model(model_version: str, vocab_size: int, cfg: dict) -> torch.nn.Module:
    common = dict(
        vocab_size=vocab_size,
        block_size=cfg["block_size"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        d_model=cfg["d_model"],
        dropout=cfg.get("dropout", 0.1),
        layernorm_eps=cfg.get("layernorm_eps", 1e-5),
        norm_type=cfg.get("norm_type", "layernorm"),
        mlp_type=cfg.get("mlp_type", "gelu"),
        pos_encoding=cfg.get("pos_encoding", "learned"),
        attention_impl=cfg.get("attention_impl", "manual"),
        rope_base=cfg.get("rope_base", 10000),
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
    )

    if model_version == "gpt2":
        return GPT2(**common)

    resid_dropout = cfg.get("resid_dropout", cfg.get("dropout", 0.1))
    return GPT3(
        **common,
        mlp_expansion=cfg.get("mlp_expansion", 4),
        resid_dropout=resid_dropout,
    )


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def run_single_experiment(
    *,
    run_name: str,
    output_dir: Path,
    model_version: str,
    model_cfg: dict,
    train_cfg: dict,
    runtime_cfg: dict,
    data_bundle,
) -> dict:
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, tokenizer = data_bundle

    set_seed(int(runtime_cfg.get("seed", 42)))

    model = build_model(model_version, tokenizer.get_vocab_size(), model_cfg)
    total_params, trainable_params = count_parameters(model)

    device = runtime_cfg.get("device", "cpu")
    model.to(device)

    compile_enabled = bool(runtime_cfg.get("compile", False))
    if compile_enabled and hasattr(torch, "compile"):
        model = torch.compile(model, mode=runtime_cfg.get("compile_mode", "default"))

    ckpt_best = run_dir / "best.pt"
    ckpt_last = run_dir / "last.pt"
    history_path = run_dir / "history.pt"
    config_path = run_dir / "config.json"

    run_config = {
        "run_name": run_name,
        "model_version": model_version,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "runtime_cfg": runtime_cfg,
        "params_total": total_params,
        "params_trainable": trainable_params,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    start_time = time.time()
    history = train_gpt_lm(
        model,
        train_loader,
        val_loader=val_loader if train_cfg.get("val_checking", False) else None,
        epochs=train_cfg.get("epochs", 1),
        max_steps=train_cfg.get("max_steps"),
        base_lr=train_cfg.get("base_lr", 3e-4),
        weight_decay=train_cfg.get("weight_decay"),
        warmup_steps=train_cfg.get("warmup_steps"),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
        grad_clip=train_cfg.get("grad_clip"),
        betas=train_cfg.get("betas"),
        gpt_version=model_version,
        scheduler_type=train_cfg.get("scheduler_type"),
        device=device,
        ckpt_path=str(ckpt_best),
        log_every=train_cfg.get("log_every", 100),
        preview_every=None,
        id2tok_fn=None,
        amp_enabled=runtime_cfg.get("amp", False),
        amp_dtype=runtime_cfg.get("amp_dtype", "bf16"),
        val_checking=train_cfg.get("val_checking", False),
        save_ckpt_every=train_cfg.get("save_ckpt_every"),
    )
    elapsed = time.time() - start_time

    torch.save(history, history_path)

    raw_model = _unwrap_model(model)
    torch.save(
        {
            "model_state": raw_model.state_dict(),
            "run_config": run_config,
            "history": history,
        },
        ckpt_last,
    )

    train_loss_last = history["train_loss"][-1] if history["train_loss"] else None
    train_ppl_last = history["train_ppl"][-1] if history["train_ppl"] else None
    val_loss_best = min(history["val_loss"]) if history["val_loss"] else None
    val_ppl_best = min(history["val_ppl"]) if history["val_ppl"] else None

    result = {
        "run_name": run_name,
        "model_version": model_version,
        "seed": runtime_cfg.get("seed", 42),
        "params_total": total_params,
        "params_trainable": trainable_params,
        "train_loss_last": train_loss_last,
        "train_ppl_last": train_ppl_last,
        "val_loss_best": val_loss_best,
        "val_ppl_best": val_ppl_best,
        "global_steps": history.get("global_steps"),
        "elapsed_sec": elapsed,
        "best_ckpt": str(ckpt_best),
        "last_ckpt": str(ckpt_last),
        "history_path": str(history_path),
        "config_path": str(config_path),
    }
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
