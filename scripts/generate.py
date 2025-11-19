#!/usr/bin/env python3
"""Simple text generation CLI that loads a trained checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.gpt_model import GPT2, GPT3
from src.inference.generate_text import generate, generate_greedy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera texto usando un checkpoint entrenado.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Ruta al checkpoint (.pt) con model_state.")
    parser.add_argument("--tokenizer-path", type=str, required=True,
                        help="Ruta al tokenizer (tokenizers.Tokenizer).")
    parser.add_argument("--model-version", choices=["gpt2", "gpt3"], default="gpt2",
                        help="Arquitectura usada al entrenar.")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Contexto máximo del modelo.")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layernorm-eps", type=float, default=1e-5)
    parser.add_argument("--mlp-expansion", type=int, default=4,
                        help="Factor de expansión del MLP (GPT-3).")
    parser.add_argument("--resid-dropout", type=float, default=None,
                        help="Dropout residual en GPT-3 (None=usa dropout base).")
    parser.add_argument("--prompt", type=str, default="Hola",
                        help="Texto inicial para la generación.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--strategy", choices=["topk", "greedy"], default="topk",
                        help="Estrategia de generación.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(args: argparse.Namespace, vocab_size: int) -> torch.nn.Module:
    common = dict(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
        layernorm_eps=args.layernorm_eps,
    )
    if args.model_version == "gpt2":
        return GPT2(**common)
    resid_dropout = args.resid_dropout if args.resid_dropout is not None else args.dropout
    return GPT3(**common,
                mlp_expansion=args.mlp_expansion,
                resid_dropout=resid_dropout)


def load_model(args: argparse.Namespace, tokenizer: Tokenizer) -> torch.nn.Module:
    model = build_model(args, tokenizer.get_vocab_size())
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    model = load_model(args, tokenizer)

    if args.strategy == "greedy":
        text = generate_greedy(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    else:
        text = generate(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device,
        )

    print("===== TEXTO GENERADO =====")
    print(text)


if __name__ == "__main__":
    main()
