import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import CausalSelfAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)
        return x * self.weight


def build_norm(norm_type: str, d_model: int, eps: float = 1e-5) -> nn.Module:
    name = (norm_type or "layernorm").lower()
    if name == "layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    if name == "rmsnorm":
        return RMSNorm(d_model, eps=eps)
    raise ValueError("norm_type debe ser 'layernorm' o 'rmsnorm'")


class GPT2MLP(nn.Module):
    """
    MLP posición a posición estilo GPT-2:
      Linear(d_model → d_ff) + GELU + Dropout + Linear(d_ff → d_model)
    """

    def __init__(self, d_model: int, d_ff=None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward block usado en variantes modernas de scaling."""

    def __init__(self, d_model: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc_in = nn.Linear(d_model, hidden_size * 2)
        self.fc_out = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.fc_in(x).chunk(2, dim=-1)
        x = value * F.silu(gate)
        x = self.fc_out(x)
        x = self.dropout(x)
        return x


class MLP_gpt3(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expansion * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(expansion * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class GPT2Block(nn.Module):
    """
    Bloque GPT-2:
      x -> LN -> CausalSelfAttention -> +residual
      -> LN -> MLP -> +residual
    Pre-LN (estilo GPT-2 moderno).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        d_ff=None,
        dropout: float = 0.1,
        layernorm_eps: float = 1e-5,
        norm_type: str = "layernorm",
        mlp_type: str = "gelu",
        attention_impl: str = "manual",
        use_rope: bool = False,
        rope_base: int = 10000,
    ):
        super().__init__()
        self.ln_1 = build_norm(norm_type, d_model, eps=layernorm_eps)
        self.ln_2 = build_norm(norm_type, d_model, eps=layernorm_eps)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
            attention_impl=attention_impl,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        hidden_size = d_ff if d_ff is not None else 4 * d_model
        mlp_variant = (mlp_type or "gelu").lower()
        if mlp_variant == "gelu":
            self.mlp = GPT2MLP(d_model=d_model, d_ff=hidden_size, dropout=dropout)
        elif mlp_variant == "swiglu":
            self.mlp = SwiGLUMLP(d_model=d_model, hidden_size=hidden_size, dropout=dropout)
        else:
            raise ValueError("mlp_type debe ser 'gelu' o 'swiglu'")

    def forward(self, x: torch.Tensor):
        normalized_1 = self.ln_1(x)
        attn_output = self.attn(normalized_1)
        x = x + attn_output

        normalized_2 = self.ln_2(x)
        mlp_output = self.mlp(normalized_2)
        x = x + mlp_output

        return x


class GPT3Block(nn.Module):
    """
    Bloque GPT-3:
      LayerNorm -> CausalSelfAttention -> Residual
      LayerNorm -> MLP                -> Residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        expansion: int = 4,
        dropout: float = 0.1,
        layernorm_eps: float = 1e-5,
        resid_dropout: float | None = None,
        norm_type: str = "layernorm",
        mlp_type: str = "gelu",
        attention_impl: str = "manual",
        use_rope: bool = False,
        rope_base: int = 10000,
    ):
        super().__init__()
        self.ln_1 = build_norm(norm_type, d_model, eps=layernorm_eps)
        self.ln_2 = build_norm(norm_type, d_model, eps=layernorm_eps)

        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout,
            attention_impl=attention_impl,
            use_rope=use_rope,
            rope_base=rope_base,
        )

        hidden_size = expansion * d_model
        mlp_variant = (mlp_type or "gelu").lower()
        if mlp_variant == "gelu":
            self.mlp = MLP_gpt3(d_model=d_model, expansion=expansion, dropout=dropout)
        elif mlp_variant == "swiglu":
            self.mlp = SwiGLUMLP(d_model=d_model, hidden_size=hidden_size, dropout=dropout)
        else:
            raise ValueError("mlp_type debe ser 'gelu' o 'swiglu'")

        dropout_value = dropout if resid_dropout is None else resid_dropout
        self.attn_resid = nn.Dropout(dropout_value)
        self.mlp_resid = nn.Dropout(dropout_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attn(self.ln_1(x))
        x = x + self.attn_resid(attn_output)

        mlp_output = self.mlp(self.ln_2(x))
        x = x + self.mlp_resid(mlp_output)

        return x
