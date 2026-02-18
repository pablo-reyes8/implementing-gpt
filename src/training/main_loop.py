import math
import time

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from src.training.optimizer import create_optimizer_and_scheduler, token_acc
from src.training.scaler import autocast_ctx, make_grad_scaler


def _unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "_orig_mod"):  # torch.compile wrapper
        return model._orig_mod
    return model


TRAINING_PRESETS = {
    "gpt2": {
        "weight_decay": 0.01,
        "betas": (0.9, 0.999),
        "grad_clip": 1.0,
        "scheduler": "cosine",
        "warmup_fn": lambda steps: 2000,
    },
    "gpt3": {
        "weight_decay": 0.1,
        "betas": (0.9, 0.95),
        "grad_clip": 1.0,
        "scheduler": "cosine",
        "warmup_fn": lambda steps: max(10, int(0.01 * max(1, steps))),
    },
}


def _resolve_training_hparams(
    version: str,
    total_steps: int,
    *,
    warmup_steps: int | None,
    weight_decay: float | None,
    grad_clip: float | None,
    betas: tuple[float, float] | None,
    scheduler_type: str | None,
) -> dict:
    key = (version or "gpt2").lower()
    if key not in TRAINING_PRESETS:
        raise ValueError(f"Versión '{version}' no soportada. Usa 'gpt2' o 'gpt3'.")

    preset = TRAINING_PRESETS[key]

    resolved = {}
    resolved["weight_decay"] = weight_decay if weight_decay is not None else preset["weight_decay"]
    resolved["betas"] = betas if betas is not None else preset["betas"]
    resolved["grad_clip"] = grad_clip if grad_clip is not None else preset["grad_clip"]

    if warmup_steps is not None:
        warmup = warmup_steps
    else:
        warmup = preset["warmup_fn"](total_steps)

    warmup = int(max(0, min(warmup, max(0, total_steps))))
    resolved["warmup_steps"] = warmup
    resolved["scheduler"] = scheduler_type or preset.get("scheduler", "none")
    return resolved


def train_gpt_lm(
    model,
    train_loader,
    val_loader=None,
    *,
    epochs: int = 10,
    base_lr: float = 3e-4,
    weight_decay: float | None = None,
    warmup_steps: int | None = None,
    label_smoothing: float = 0.0,
    grad_clip: float | None = None,
    betas: tuple[float, float] | None = None,
    gpt_version: str = "gpt2",
    scheduler_type: str | None = None,
    device: str = "cuda",
    ckpt_path: str = "gptmini_best.pt",
    log_every: int = 100,
    preview_every: int | None = None,
    id2tok_fn=None,
    amp_enabled: bool = True,
    amp_dtype: str = "bf16",
    val_checking: bool = False,
    save_ckpt_every: int | None = None,
    max_steps: int | None = None,
):
    """
    Entrena un modelo GPT decoder-only con soporte de:
      - presets GPT-2/GPT-3
      - warmup + cosine
      - AMP, grad clip y checkpointing por epochs
      - corte temprano por max_steps para comparaciones research
    """
    device = torch.device(device)
    torch.set_float32_matmul_precision("high")
    model.to(device)
    model.train()

    estimated_steps = epochs * len(train_loader)
    total_steps = min(estimated_steps, max_steps) if max_steps is not None else estimated_steps

    training_cfg = _resolve_training_hparams(
        gpt_version,
        total_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        betas=betas,
        scheduler_type=scheduler_type,
    )
    weight_decay = training_cfg["weight_decay"]
    warmup_steps = training_cfg["warmup_steps"]
    grad_clip = training_cfg["grad_clip"]
    betas = training_cfg["betas"]
    scheduler_choice = training_cfg["scheduler"]

    optimizer, scheduler = create_optimizer_and_scheduler(
        model,
        base_lr=base_lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        betas=betas,
        scheduler_type=scheduler_choice,
    )

    ce_train = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    ce_eval = nn.CrossEntropyLoss(label_smoothing=0.0)

    use_scaler = amp_enabled and (amp_dtype.lower() in ("fp16", "float16"))
    scaler = make_grad_scaler(
        device="cuda" if device.type == "cuda" else "cpu",
        enabled=use_scaler,
    )

    best_val = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
        "train_tok_acc": [],
        "val_tok_acc": [],
    }

    global_step = 0

    for epoch in range(1, epochs + 1):
        if max_steps is not None and global_step >= max_steps:
            break

        model.train()
        epoch_loss_sum, epoch_tokens = 0.0, 0
        epoch_acc_sum = 0.0
        t0 = time.time()

        for it, (x_ids, y_ids) in enumerate(train_loader, start=1):
            if max_steps is not None and global_step >= max_steps:
                break

            global_step += 1
            x_ids = x_ids.to(device, non_blocking=True)
            y_ids = y_ids.to(device, non_blocking=True)
            bsz, seq_len = x_ids.shape
            tokens = bsz * seq_len

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(device=device.type, enabled=amp_enabled, dtype=amp_dtype):
                logits, _ = model(x_ids, None)
                _, _, vocab_size = logits.shape
                loss = ce_train(logits.view(bsz * seq_len, vocab_size), y_ids.view(bsz * seq_len))

            if scaler is not None:
                if loss.dim() > 0:
                    loss = loss.mean()

                scaler.scale(loss).backward()
                if grad_clip is not None:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                if loss.dim() > 0:
                    loss = loss.mean()

                loss.backward()
                if grad_clip is not None:
                    clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                acc = token_acc(logits, y_ids)

            epoch_loss_sum += loss.item() * tokens
            epoch_acc_sum += acc * tokens
            epoch_tokens += tokens

            if it % log_every == 0:
                avg_loss = epoch_loss_sum / max(1, epoch_tokens)
                avg_ppl = math.exp(avg_loss)
                avg_acc = epoch_acc_sum / max(1, epoch_tokens)
                tok_per_sec = epoch_tokens / (time.time() - t0 + 1e-9)
                print(
                    f"[Epoch {epoch} | step {it:4d}/{len(train_loader)} | global_step={global_step}] "
                    f"train_loss={avg_loss:.4f}  ppl={avg_ppl:.2f}  "
                    f"tok_acc={avg_acc * 100:.2f}%  tok/s={tok_per_sec:,.0f}"
                )

            if (preview_every is not None) and (id2tok_fn is not None) and (it % preview_every == 0):
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    b0 = 0

                    in_ids = x_ids[b0].tolist()
                    tgt_ids = y_ids[b0].tolist()
                    pred_ids = preds[b0].tolist()

                    max_show = min(80, len(in_ids))
                    in_ids = in_ids[:max_show]
                    tgt_ids = tgt_ids[:max_show]
                    pred_ids = pred_ids[:max_show]

                    ctx = id2tok_fn(in_ids)
                    ref = id2tok_fn(tgt_ids)
                    hyp = id2tok_fn(pred_ids)

                    print("— preview (LM, teacher-forced argmax) —")
                    print("CTX:", repr(ctx))
                    print("REF:", repr(ref))
                    print("HYP:", repr(hyp))

        if epoch_tokens == 0:
            break

        train_loss = epoch_loss_sum / max(1, epoch_tokens)
        train_ppl = math.exp(train_loss)
        train_acc = epoch_acc_sum / max(1, epoch_tokens)

        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["train_tok_acc"].append(train_acc * 100.0)

        if (not val_checking) or (val_loader is None):
            print(
                f"Epoch {epoch} done | "
                f"train_loss={train_loss:.4f}  train_ppl={train_ppl:.2f}  "
                f"train_tok_acc={train_acc * 100:.2f}%"
            )

            if save_ckpt_every is not None and (epoch % save_ckpt_every == 0):
                model_to_save = _unwrap_model(model)
                torch.save(
                    {
                        "model_state": model_to_save.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    ckpt_path,
                )
                print(f"Guardado checkpoint (cada {save_ckpt_every} epochs) -> {ckpt_path}")

            continue

        model.eval()
        val_loss_sum, val_tokens = 0.0, 0
        val_acc_sum = 0.0

        with torch.no_grad():
            for x_ids, y_ids in val_loader:
                x_ids = x_ids.to(device, non_blocking=True)
                y_ids = y_ids.to(device, non_blocking=True)
                bsz, seq_len = x_ids.shape
                tokens = bsz * seq_len

                with autocast_ctx(device=device.type, enabled=amp_enabled, dtype=amp_dtype):
                    logits, _ = model(x_ids, None)
                    vocab_size = logits.size(-1)
                    loss = ce_eval(
                        logits.view(bsz * seq_len, vocab_size),
                        y_ids.view(bsz * seq_len),
                    )

                acc = token_acc(logits, y_ids)

                val_loss_sum += loss.item() * tokens
                val_acc_sum += acc * tokens
                val_tokens += tokens

        val_loss = val_loss_sum / max(1, val_tokens)
        val_ppl = math.exp(val_loss)
        val_acc = val_acc_sum / max(1, val_tokens)

        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["val_tok_acc"].append(val_acc * 100.0)

        print(
            f"Epoch {epoch} done | "
            f"train_loss={train_loss:.4f}  train_ppl={train_ppl:.2f}  train_tok_acc={train_acc * 100:.2f}%  "
            f"val_loss={val_loss:.4f}    val_ppl={val_ppl:.2f}    val_tok_acc={val_acc * 100:.2f}%"
        )

        if val_loss < best_val:
            best_val = val_loss
            model_to_save = _unwrap_model(model)
            torch.save(
                {
                    "model_state": model_to_save.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"Guardado checkpoint (best val_loss={val_loss:.4f}) -> {ckpt_path}")

    history["global_steps"] = global_step
    history["max_steps"] = max_steps
    return history
