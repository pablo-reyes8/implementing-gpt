#!/usr/bin/env python3
"""Compare GPT-2 vs GPT-3 under matched training conditions."""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_registry import dataset_choices
from src.research.common import load_dataloaders, run_single_experiment, write_csv, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corre benchmark controlado GPT-2 vs GPT-3 en mismas condiciones."
    )
    parser.add_argument("--dataset", choices=dataset_choices(), default="small")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--models", nargs="+", default=["gpt2", "gpt3"], choices=["gpt2", "gpt3"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])

    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--layernorm-eps", type=float, default=1e-5)
    parser.add_argument("--mlp-expansion", type=int, default=4)
    parser.add_argument("--resid-dropout", type=float, default=None)
    parser.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="layernorm")
    parser.add_argument("--mlp-type", choices=["gelu", "swiglu"], default="gelu")
    parser.add_argument("--pos-encoding", choices=["learned", "rope", "none"], default="learned")
    parser.add_argument("--attention-impl", choices=["manual", "sdpa"], default="manual")
    parser.add_argument("--rope-base", type=int, default=10000)
    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--beta1", type=float, default=None)
    parser.add_argument("--beta2", type=float, default=None)
    parser.add_argument("--scheduler-type", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--val-checking", action="store_true")
    parser.add_argument("--save-ckpt-every", type=int, default=None)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )

    parser.add_argument("--output-dir", type=str, default="research_runs/compare")
    parser.add_argument("--experiment-name", type=str, default=None)

    return parser.parse_args()


def print_summary(results: list[dict]) -> None:
    if not results:
        return

    print("\n=== Summary by model ===")
    by_model: dict[str, list[dict]] = {}
    for row in results:
        by_model.setdefault(row["model_version"], []).append(row)

    for model_name, rows in by_model.items():
        key = "val_loss_best" if any(r["val_loss_best"] is not None for r in rows) else "train_loss_last"
        vals = [r[key] for r in rows if r[key] is not None]
        mean_val = statistics.mean(vals) if vals else float("nan")
        print(f"{model_name:>5} | runs={len(rows):2d} | {key} mean={mean_val:.4f}")


def main() -> None:
    args = parse_args()

    if args.d_model % args.n_head != 0:
        raise ValueError("d_model debe ser múltiplo de n_head")

    if args.experiment_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"gpt2_vs_gpt3_{args.dataset}_{ts}"

    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("Cargando dataloaders (se reutilizan para todas las corridas)...")
    data_bundle = load_dataloaders(
        args.dataset,
        block_size=args.block_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_cfg = {
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "d_model": args.d_model,
        "dropout": args.dropout,
        "layernorm_eps": args.layernorm_eps,
        "mlp_expansion": args.mlp_expansion,
        "resid_dropout": args.resid_dropout,
        "norm_type": args.norm_type,
        "mlp_type": args.mlp_type,
        "pos_encoding": args.pos_encoding,
        "attention_impl": args.attention_impl,
        "rope_base": args.rope_base,
        "gradient_checkpointing": args.gradient_checkpointing,
    }

    betas = None
    if args.beta1 is not None and args.beta2 is not None:
        betas = (args.beta1, args.beta2)

    train_cfg = {
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "base_lr": args.base_lr,
        "label_smoothing": args.label_smoothing,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "betas": betas,
        "scheduler_type": args.scheduler_type,
        "log_every": args.log_every,
        "val_checking": args.val_checking,
        "save_ckpt_every": args.save_ckpt_every,
    }

    results: list[dict] = []

    for model_version in args.models:
        for seed in args.seeds:
            run_name = f"{model_version}_seed{seed}"
            runtime_cfg = {
                "seed": seed,
                "device": args.device,
                "amp": args.amp,
                "amp_dtype": args.amp_dtype,
                "compile": args.compile,
                "compile_mode": args.compile_mode,
            }
            print(f"\n>>> Running {run_name}")
            row = run_single_experiment(
                run_name=run_name,
                output_dir=experiment_dir,
                model_version=model_version,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                runtime_cfg=runtime_cfg,
                data_bundle=data_bundle,
            )
            results.append(row)

    jsonl_path = experiment_dir / "results.jsonl"
    csv_path = experiment_dir / "results.csv"
    write_jsonl(jsonl_path, results)
    write_csv(csv_path, results)

    print_summary(results)
    print(f"\nResultados guardados en: {jsonl_path}")
    print(f"Resultados tabulares en: {csv_path}")


if __name__ == "__main__":
    main()
