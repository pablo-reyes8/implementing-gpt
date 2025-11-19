import pytest
import torch

from src.training.main_loop import _resolve_training_hparams
from src.training.optimizer import WarmupCosineScheduler


def test_resolve_training_hparams_uses_presets():
    cfg = _resolve_training_hparams(
        version="gpt3",
        total_steps=1000,
        warmup_steps=None,
        weight_decay=None,
        grad_clip=None,
        betas=None,
        scheduler_type=None,
    )
    assert cfg["weight_decay"] == 0.1
    assert cfg["betas"] == (0.9, 0.95)
    assert cfg["grad_clip"] == 1.0
    assert cfg["warmup_steps"] == max(10, int(0.01 * 1000))
    assert cfg["scheduler"] == "cosine"


def test_resolve_training_hparams_overrides():
    cfg = _resolve_training_hparams(
        version="gpt2",
        total_steps=100,
        warmup_steps=5,
        weight_decay=0.5,
        grad_clip=0.7,
        betas=(0.8, 0.88),
        scheduler_type="none",
    )
    assert cfg["weight_decay"] == 0.5
    assert cfg["grad_clip"] == 0.7
    assert cfg["betas"] == (0.8, 0.88)
    assert cfg["warmup_steps"] == 5
    assert cfg["scheduler"] == "none"


def test_warmup_cosine_scheduler_warmup_and_decay():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=2, max_steps=4)
    lrs = []
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    assert max(lrs) <= 1e-3 * 1.01
    assert min(lrs) >= -1e-8
    assert any(lr < lrs[0] for lr in lrs), "Scheduler never decayed the LR"
    assert lrs[-1] <= lrs[0]
