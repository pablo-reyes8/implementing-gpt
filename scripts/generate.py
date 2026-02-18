#!/usr/bin/env python3
"""Text generation CLI that loads trained checkpoints (baseline + research variants)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.generate_text import generate, generate_greedy
from src.model.gpt_model import GPT2, GPT3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera texto usando un checkpoint entrenado.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--use-ckpt-config", action="store_true", help="Intenta reconstruir la arquitectura desde run_config del checkpoint.")

    parser.add_argument("--model-version", choices=["gpt2", "gpt3"], default="gpt2")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--layernorm-eps", type=float, default=1e-5)
    parser.add_argument("--mlp-expansion", type=int, default=4)
    parser.add_argument("--resid-dropout", type=float, default=None)

    parser.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="layernorm")
    parser.add_argument("--mlp-type", choices=["gelu", "swiglu"], default="gelu")
    parser.add_argument("--pos-encoding", choices=["learned", "rope", "none"], default="learned")
    parser.add_argument("--attention-impl", choices=["manual", "sdpa"], default="manual")
    parser.add_argument("--rope-base", type=int, default=10000)

    parser.add_argument("--prompt", type=str, default="Hola")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--strategy", choices=["topk", "greedy"], default="topk")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
        norm_type=args.norm_type,
        mlp_type=args.mlp_type,
        pos_encoding=args.pos_encoding,
        attention_impl=args.attention_impl,
        rope_base=args.rope_base,
    )
    if args.model_version == "gpt2":
        return GPT2(**common)

    resid_dropout = args.resid_dropout if args.resid_dropout is not None else args.dropout
    return GPT3(
        **common,
        mlp_expansion=args.mlp_expansion,
        resid_dropout=resid_dropout,
    )


def _fill_args_from_run_config(args: argparse.Namespace, run_cfg: dict) -> None:
    keys = [
        "model_version",
        "block_size",
        "n_layer",
        "n_head",
        "d_model",
        "dropout",
        "layernorm_eps",
        "mlp_expansion",
        "resid_dropout",
        "norm_type",
        "mlp_type",
        "pos_encoding",
        "attention_impl",
        "rope_base",
    ]
    for key in keys:
        if key in run_cfg and getattr(args, key, None) is not None:
            setattr(args, key, run_cfg[key])


def load_model(args: argparse.Namespace, tokenizer: Tokenizer) -> torch.nn.Module:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if args.use_ckpt_config and isinstance(checkpoint, dict):
        run_cfg = checkpoint.get("run_config")
        if isinstance(run_cfg, dict):
            # Compatible con checkpoints de train.py (flat) y research/common.py (nested).
            candidate_cfg = dict(run_cfg)
            if isinstance(run_cfg.get("model_cfg"), dict):
                candidate_cfg.update(run_cfg["model_cfg"])
            _fill_args_from_run_config(args, candidate_cfg)

    model = build_model(args, tokenizer.get_vocab_size())
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
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
