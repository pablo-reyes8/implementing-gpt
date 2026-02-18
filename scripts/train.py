#!/usr/bin/env python3
"""Research-friendly training CLI for GPT-2/GPT-3 style models."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.gpt_model import GPT2, GPT3
from src.training.main_loop import train_gpt_lm

DATASET_LOADERS = {
    "small": "src.data.load_small_data",
    "large": "src.data.load_large_data",
}

MODEL_PRESETS = {
    "nano": {"n_layer": 4, "n_head": 4, "d_model": 256},
    "small": {"n_layer": 8, "n_head": 8, "d_model": 512},
    "base": {"n_layer": 12, "n_head": 12, "d_model": 768},
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


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena un modelo GPT-2/GPT-3 con opciones de investigación "
            "(ablation-friendly y reproducible)."
        )
    )

    data = parser.add_argument_group("Data")
    data.add_argument(
        "--dataset",
        choices=DATASET_LOADERS.keys(),
        default="small",
        help="Qué dataloader usar (small=openwebtext10k, large=wikitext-103).",
    )
    data.add_argument("--block-size", type=int, default=256)
    data.add_argument("--batch-size", type=int, default=32)
    data.add_argument("--num-workers", type=int, default=2)

    model = parser.add_argument_group("Model")
    model.add_argument("--model-version", choices=["gpt2", "gpt3"], default="gpt2")
    model.add_argument(
        "--model-preset",
        choices=MODEL_PRESETS.keys(),
        default=None,
        help="Sobrescribe n_layer/n_head/d_model con un preset.",
    )
    model.add_argument("--n-layer", type=int, default=4)
    model.add_argument("--n-head", type=int, default=4)
    model.add_argument("--d-model", type=int, default=256)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--layernorm-eps", type=float, default=1e-5)
    model.add_argument("--mlp-expansion", type=int, default=4)
    model.add_argument("--resid-dropout", type=float, default=None)

    model.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="layernorm")
    model.add_argument("--mlp-type", choices=["gelu", "swiglu"], default="gelu")
    model.add_argument("--pos-encoding", choices=["learned", "rope", "none"], default="learned")
    model.add_argument("--attention-impl", choices=["manual", "sdpa"], default="manual")
    model.add_argument("--rope-base", type=int, default=10000)
    model.add_argument("--gradient-checkpointing", action="store_true")

    train = parser.add_argument_group("Training")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--base-lr", type=float, default=3e-4)
    train.add_argument("--label-smoothing", type=float, default=0.0)
    train.add_argument("--warmup-steps", type=int, default=None)
    train.add_argument("--weight-decay", type=float, default=None)
    train.add_argument("--grad-clip", type=float, default=None)
    train.add_argument("--beta1", type=float, default=None)
    train.add_argument("--beta2", type=float, default=None)
    train.add_argument("--scheduler-type", type=str, default=None)
    train.add_argument("--log-every", type=int, default=100)
    train.add_argument("--preview-every", type=int, default=None)
    train.add_argument("--val-checking", action="store_true")
    train.add_argument("--save-ckpt-every", type=int, default=None)

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--seed", type=int, default=42)
    runtime.add_argument("--amp", action="store_true", default=False)
    runtime.add_argument("--amp-dtype", type=str, default="bf16")
    runtime.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    runtime.add_argument("--compile", action="store_true", help="Usa torch.compile si está disponible.")
    runtime.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )

    io = parser.add_argument_group("Output")
    io.add_argument("--output-dir", type=str, default="checkpoints")
    io.add_argument("--run-name", type=str, default=None)
    io.add_argument("--ckpt-name", type=str, default=None)
    io.add_argument("--history-path", type=str, default=None)
    io.add_argument("--config-path", type=str, default=None)

    return parser.parse_args()


def apply_model_preset(args: argparse.Namespace) -> None:
    if not args.model_preset:
        return

    preset = MODEL_PRESETS[args.model_preset]
    args.n_layer = preset["n_layer"]
    args.n_head = preset["n_head"]
    args.d_model = preset["d_model"]


def load_dataloaders(dataset_name: str, block_size: int, batch_size: int, num_workers: int):
    module = importlib.import_module(DATASET_LOADERS[dataset_name])
    return module.create_dataloaders(
        block_size=block_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def build_model(args: argparse.Namespace, vocab_size: int) -> torch.nn.Module:
    common = dict(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        layernorm_eps=args.layernorm_eps,
        norm_type=args.norm_type,
        mlp_type=args.mlp_type,
        pos_encoding=args.pos_encoding,
        attention_impl=args.attention_impl,
        rope_base=args.rope_base,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if args.model_version == "gpt2":
        return GPT2(**common)

    resid_dropout = args.resid_dropout if args.resid_dropout is not None else args.dropout
    return GPT3(
        **common,
        mlp_expansion=args.mlp_expansion,
        resid_dropout=resid_dropout,
    )


def to_jsonable_dict(args: argparse.Namespace) -> dict:
    return {k: getattr(args, k) for k in vars(args)}


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def main():
    args = parse_args()
    apply_model_preset(args)

    if args.d_model % args.n_head != 0:
        raise ValueError("d_model debe ser múltiplo de n_head.")

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.run_name is None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_name = f"{args.model_version}_{args.dataset}_{now}"

    if args.ckpt_name is None:
        args.ckpt_name = f"{args.run_name}.pt"

    ckpt_path = output_dir / args.ckpt_name

    if args.history_path is None:
        args.history_path = str(output_dir / f"{args.run_name}_history.pt")

    if args.config_path is None:
        args.config_path = str(output_dir / f"{args.run_name}_config.json")

    run_config = to_jsonable_dict(args)
    with open(args.config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)
    print(f"Config guardada en {args.config_path}")

    print(f"Usando dataset '{args.dataset}' con block_size={args.block_size} ...")
    train_loader, val_loader, tokenizer = load_dataloaders(
        args.dataset,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args, tokenizer.get_vocab_size())
    total_params, trainable_params = count_parameters(model)
    print(f"Parámetros totales: {total_params:,} | entrenables: {trainable_params:,}")

    device = torch.device(args.device)
    model.to(device)

    if args.compile:
        if hasattr(torch, "compile"):
            print(f"Compilando modelo con torch.compile(mode='{args.compile_mode}')...")
            model = torch.compile(model, mode=args.compile_mode)
        else:
            print("torch.compile no está disponible en esta versión de PyTorch. Se omite.")

    betas = None
    if args.beta1 is not None and args.beta2 is not None:
        betas = (args.beta1, args.beta2)

    print("Comenzando entrenamiento...")
    history = train_gpt_lm(
        model,
        train_loader,
        val_loader=val_loader if args.val_checking else None,
        epochs=args.epochs,
        max_steps=args.max_steps,
        base_lr=args.base_lr,
        label_smoothing=args.label_smoothing,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        betas=betas,
        gpt_version=args.model_version,
        scheduler_type=args.scheduler_type,
        device=args.device,
        ckpt_path=str(ckpt_path),
        log_every=args.log_every,
        preview_every=args.preview_every,
        id2tok_fn=(lambda ids: tokenizer.decode(ids)) if args.preview_every else None,
        amp_enabled=args.amp,
        amp_dtype=args.amp_dtype,
        val_checking=args.val_checking,
        save_ckpt_every=args.save_ckpt_every,
    )

    torch.save(history, args.history_path)
    print(f"Historial guardado en {args.history_path}")

    # Siempre guardamos un "last" checkpoint para reproducibilidad,
    # independientemente de si el loop guardó un "best".
    raw_model = unwrap_model(model)
    final_path = output_dir / f"{Path(args.ckpt_name).stem}.last.pt"
    torch.save(
        {
            "model_state": raw_model.state_dict(),
            "history": history,
            "run_config": run_config,
            "tokenizer_vocab_size": tokenizer.get_vocab_size(),
        },
        final_path,
    )

    print("Entrenamiento finalizado.")
    print(f"Checkpoint training-loop: {ckpt_path}")
    print(f"Checkpoint final (last): {final_path}")


if __name__ == "__main__":
    main()
