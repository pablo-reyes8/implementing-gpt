import torch.nn.functional as F
import torch
import torch.nn as nn
import math

def token_acc(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    logits:  [B, T, V]
    targets: [B, T]
    return: accuracy escalar en [0,1]
    """
    preds = logits.argmax(dim=-1)       # [B, T]
    correct = (preds == targets).sum().item()
    total   = targets.numel()
    return correct / total


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Warmup lineal + decaimiento coseno:
      - durante 'warmup_steps': lr sube lineal desde 0 hasta base_lr
      - después: decae con coseno hasta ~0 en 'max_steps'
    """
    def __init__(self, optimizer, warmup_steps: int, max_steps: int, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self._step_num = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        self._step_num += 1
        step = self._step_num

        if step <= self.warmup_steps:
            # Warmup lineal: 0 -> 1
            scale = step / float(max(1, self.warmup_steps))
        else:
            # Cosine decay de 1 -> 0
            progress = (step - self.warmup_steps) / float(
                max(1, self.max_steps - self.warmup_steps))
            
            # cos(pi * 0) = 1, cos(pi * 1) = -1  ⇒  scale va de 1 -> 0
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [base_lr * scale for base_lr in self.base_lrs]


def create_optimizer_and_scheduler(
    model,
    base_lr: float,
    weight_decay: float,
    warmup_steps: int,
    max_steps: int,
    *,
    betas: tuple[float, float] = (0.9, 0.95),
    scheduler_type: str = "cosine"):
  
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=betas,
        weight_decay=weight_decay)

    scheduler = None
    sched_type = (scheduler_type or "none").lower()
    if sched_type in ("cosine", "warmup_cosine"):
        scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            max_steps=max_steps)
    elif sched_type in ("none", "constant"):
        scheduler = None
    else:
        raise ValueError(f"Scheduler '{scheduler_type}' no soportado.")
    
    return optimizer, scheduler
