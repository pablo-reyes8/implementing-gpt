#!/usr/bin/env python3
"""Command-line training pipeline for GPT-2/GPT-3 style models."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.gpt_model import GPT2, GPT3
from src.training.main_loop import train_gpt_lm

DATASET_LOADERS = {
    "small": "src.data.load_small_data",
    "large": "src.data.load_large_data",}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un modelo GPT-2/GPT-3 usando los dataloaders del repo.")
    parser.add_argument("--dataset", choices=DATASET_LOADERS.keys(), default="small",
                        help="Qué dataloader usar (small=openwebtext10k, large=wikitext-103).")
    parser.add_argument("--model-version", choices=["gpt2", "gpt3"], default="gpt2",
                        help="Arquitectura base a utilizar.")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Longitud de contexto para el modelo y los dataloaders.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size de entrenamiento.")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Workers del DataLoader.")
    parser.add_argument("--n-layer", type=int, default=4,
                        help="Número de bloques del modelo.")
    parser.add_argument("--n-head", type=int, default=4,
                        help="Número de cabezas de atención.")
    parser.add_argument("--d-model", type=int, default=256,
                        help="Dimensión del embedding del modelo.")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout estándar del modelo.")
    parser.add_argument("--layernorm-eps", type=float, default=1e-5,
                        help="Epsilon usado en LayerNorm.")
    parser.add_argument("--mlp-expansion", type=int, default=4,
                        help="Factor de expansión del MLP para GPT-3.")
    parser.add_argument("--resid-dropout", type=float, default=None,
                        help="Dropout de las conexiones residuales en GPT-3 (None=usa dropout base).")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Número de epochs a entrenar.")
    parser.add_argument("--base-lr", type=float, default=3e-4,
                        help="Learning rate base.")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Sobrescribe los warmup steps (por defecto usa preset).")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="Sobrescribe weight decay (por defecto usa preset).")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="Sobrescribe el grad clip (por defecto usa preset).")
    parser.add_argument("--beta1", type=float, default=None,
                        help="Beta1 para AdamW (por defecto usa preset).")
    parser.add_argument("--beta2", type=float, default=None,
                        help="Beta2 para AdamW (por defecto usa preset).")
    parser.add_argument("--scheduler-type", type=str, default=None,
                        help="Tipo de scheduler (cosine|none). Por defecto usa preset del modelo.")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Frecuencia de logging en steps.")
    parser.add_argument("--preview-every", type=int, default=None,
                        help="Cada cuántos steps hacer preview de generación (requiere tokenizer).")
    parser.add_argument("--val-checking", action="store_true",
                        help="Habilita evaluación en val_loader + checkpoints.")
    parser.add_argument("--save-ckpt-every", type=int, default=None,
                        help="Si no hay validación, guarda checkpoint cada N epochs.")
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Habilita AMP (torch.amp.autocast).")
    parser.add_argument("--amp-dtype", type=str, default="bf16",
                        help="Dtype de autocast (bf16|fp16).")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Dispositivo donde entrenar.")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directorio donde guardar checkpoints e historial.")
    parser.add_argument("--ckpt-name", type=str, default="gpt_model.pt",
                        help="Nombre del checkpoint resultante.")
    parser.add_argument("--history-path", type=str, default=None,
                        help="Ruta opcional para guardar el historial en .pt.")
    return parser.parse_args()


def load_dataloaders(dataset_name: str, block_size: int, batch_size: int, num_workers: int):
    module = importlib.import_module(DATASET_LOADERS[dataset_name])
    return module.create_dataloaders(
        block_size=block_size,
        batch_size=batch_size,
        num_workers=num_workers)


def build_model(args: argparse.Namespace, vocab_size: int) -> torch.nn.Module:
    common = dict(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        layernorm_eps=args.layernorm_eps)
    
    if args.model_version == "gpt2":
        return GPT2(**common)
    resid_dropout = args.resid_dropout if args.resid_dropout is not None else args.dropout
    return GPT3(**common,
                mlp_expansion=args.mlp_expansion,
                resid_dropout=resid_dropout,
                )


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / args.ckpt_name

    print(f"Usando dataset '{args.dataset}' con block_size={args.block_size} ...")
    train_loader, val_loader, tokenizer = load_dataloaders(
        args.dataset,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = build_model(args, tokenizer.get_vocab_size())
    device = torch.device(args.device)
    model.to(device)

    betas = None
    if args.beta1 is not None and args.beta2 is not None:
        betas = (args.beta1, args.beta2)

    print("Comenzando entrenamiento...")
    history = train_gpt_lm(
        model,
        train_loader,
        val_loader=val_loader if args.val_checking else None,
        epochs=args.epochs,
        base_lr=args.base_lr,
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
        save_ckpt_every=args.save_ckpt_every)

    if args.history_path:
        torch.save(history, args.history_path)
        print(f"Historial guardado en {args.history_path}")

    print("Entrenamiento finalizado.")
    print(f"Checkpoint final: {ckpt_path}")


if __name__ == "__main__":
    main()
