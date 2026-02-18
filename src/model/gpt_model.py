import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.model.embeddings import GPT2Embeddings
from src.model.gpt_blocks import GPT2Block, GPT3Block, RMSNorm, build_norm


def init_gptmini_weights(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Linear):
                if name == "lm_head":
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)


def init_gpt3_weights(
    model: nn.Module,
    n_layer: int | None = None,
    *,
    residual_scale: bool = True,
    base_std: float = 0.02,
):
    """
    Inicialización 'GPT-3 style' genérica para modelos decoder-only tipo GPT.

    - Embeddings: N(0, base_std)
    - Linears en bloques residuales: N(0, base_std * scale), bias = 0
    - lm_head: N(0, base_std), bias = 0
    - Norms: gamma = 1, beta = 0 (cuando exista bias)
    """

    if n_layer is None:
        if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
            n_layer = len(model.blocks)
        else:
            n_layer = 1
            residual_scale = False

    scale = 1.0
    if residual_scale:
        scale = 1.0 / math.sqrt(2.0 * n_layer)

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=base_std)
            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                if hasattr(module, "weight") and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                std = base_std if name.endswith("lm_head") or name == "lm_head" else base_std * scale
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class GPT2(nn.Module):
    """
    GPT-2 decoder-only:
      - Embeddings (token + posición)
      - n_layers de GPT2Block
      - Norm final
      - LM head (atado a embeddings de token)
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 4,
        n_head: int = 4,
        d_model: int = 256,
        dropout: float = 0.1,
        layernorm_eps: float = 1e-5,
        norm_type: str = "layernorm",
        mlp_type: str = "gelu",
        pos_encoding: str = "learned",
        attention_impl: str = "manual",
        rope_base: int = 10000,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing
        self.pos_encoding = (pos_encoding or "learned").lower()

        self.emb = GPT2Embeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            block_size=block_size,
            dropout=dropout,
            pos_encoding=self.pos_encoding,
        )

        use_rope = self.pos_encoding == "rope"
        self.blocks = nn.ModuleList(
            [
                GPT2Block(
                    d_model=d_model,
                    num_heads=n_head,
                    block_size=block_size,
                    d_ff=4 * d_model,
                    dropout=dropout,
                    layernorm_eps=layernorm_eps,
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                    attention_impl=attention_impl,
                    use_rope=use_rope,
                    rope_base=rope_base,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = build_norm(norm_type, d_model, eps=layernorm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        init_gptmini_weights(self)
        self.lm_head.weight = self.emb.tok_emb.weight

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, idx: torch.Tensor, targets=None):
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"Secuencia demasiado larga: T={seq_len}, block_size={self.block_size}")

        x = self.emb(idx)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


class GPT3(nn.Module):
    """
    GPT-3 decoder-only inspirado en el paper original.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 24,
        n_head: int = 16,
        d_model: int = 1024,
        dropout: float = 0.1,
        layernorm_eps: float = 1e-5,
        mlp_expansion: int = 4,
        resid_dropout: float | None = None,
        norm_type: str = "layernorm",
        mlp_type: str = "gelu",
        pos_encoding: str = "learned",
        attention_impl: str = "manual",
        rope_base: int = 10000,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing
        self.pos_encoding = (pos_encoding or "learned").lower()

        self.emb = GPT2Embeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            block_size=block_size,
            dropout=dropout,
            pos_encoding=self.pos_encoding,
        )

        use_rope = self.pos_encoding == "rope"
        self.blocks = nn.ModuleList(
            [
                GPT3Block(
                    d_model=d_model,
                    num_heads=n_head,
                    block_size=block_size,
                    expansion=mlp_expansion,
                    dropout=dropout,
                    resid_dropout=resid_dropout,
                    layernorm_eps=layernorm_eps,
                    norm_type=norm_type,
                    mlp_type=mlp_type,
                    attention_impl=attention_impl,
                    use_rope=use_rope,
                    rope_base=rope_base,
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = build_norm(norm_type, d_model, eps=layernorm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        init_gpt3_weights(self, n_layer=n_layer)
        self.lm_head.weight = self.emb.tok_emb.weight

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = enabled

    def forward(self, idx: torch.Tensor, targets=None):
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError(f"Secuencia demasiado larga: T={seq_len}, block_size={self.block_size}")

        x = self.emb(idx)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
