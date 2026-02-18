import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None):
    """
    q: (..., Lq, d)
    k: (..., Lk, d)
    v: (..., Lk, dv)
    mask: broadcastable a (..., Lq, Lk)
          - bool: True = BLOQUEAR (poner -inf)
          - float: 1.0 = permitir, 0.0 = bloquear
    Returns:
        output: (..., Lq, dv)
        attn:   (..., Lq, Lk)
    """
    scores = torch.matmul(q, k.transpose(-2, -1))
    dk = q.size(-1)
    scores = scores / dk**0.5

    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(mask, float("-inf"))
        else:
            scores = scores.masked_fill(mask <= 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((-x_odd, x_even), dim=-1)
    return rotated.flatten(-2)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope_cos = rope_cos.to(dtype=q.dtype, device=q.device)
    rope_sin = rope_sin.to(dtype=q.dtype, device=q.device)
    q = (q * rope_cos) + (_rotate_half(q) * rope_sin)
    k = (k * rope_cos) + (_rotate_half(k) * rope_sin)
    return q, k


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        attention_impl: str = "manual",
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model debe ser múltiplo de num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.attention_impl = (attention_impl or "manual").lower()

        if self.attention_impl not in {"manual", "sdpa"}:
            raise ValueError("attention_impl debe ser 'manual' o 'sdpa'")

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)

    def _combine_heads(self, x):
        bsz, heads, seq_len, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, heads * head_dim)

    def forward(self, x_q, x_kv, mask=None, rope_cos=None, rope_sin=None):
        q = self._split_heads(self.w_q(x_q))
        k = self._split_heads(self.w_k(x_kv))
        v = self._split_heads(self.w_v(x_kv))

        if rope_cos is not None and rope_sin is not None:
            q, k = _apply_rope(q, k, rope_cos=rope_cos, rope_sin=rope_sin)

        blocked_mask = None
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            elif mask.dim() != 4:
                raise ValueError(f"Máscara con dims no soportadas: {mask.shape}")

            blocked_mask = mask if mask.dtype == torch.bool else (mask <= 0)

        if self.attention_impl == "manual":
            attn_out, _ = scaled_dot_product_attention(q, k, v, blocked_mask)
        else:
            attn_mask = None
            if blocked_mask is not None:
                attn_mask = torch.zeros_like(blocked_mask, dtype=q.dtype)
                attn_mask = attn_mask.masked_fill(blocked_mask, float("-inf"))

            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=False,
            )

        attn_out = self._combine_heads(attn_out)
        attn_out = self.w_o(attn_out)
        attn_out = self.dropout(attn_out)
        return attn_out


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int,
        dropout: float = 0.1,
        attention_impl: str = "manual",
        use_rope: bool = False,
        rope_base: int = 10000,
    ):
        super().__init__()

        self.mha = MultiHeadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            attention_impl=attention_impl,
        )
        self.block_size = block_size
        self.use_rope = use_rope

        mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

        if self.use_rope:
            d_head = d_model // num_heads
            if d_head % 2 != 0:
                raise ValueError("RoPE requiere d_head par.")

            inv_freq = 1.0 / (
                rope_base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
            )
            positions = torch.arange(block_size, dtype=torch.float32)
            angles = torch.outer(positions, inv_freq)
            rope_cos = torch.repeat_interleave(torch.cos(angles), repeats=2, dim=-1)
            rope_sin = torch.repeat_interleave(torch.sin(angles), repeats=2, dim=-1)
            self.register_buffer("rope_cos_cached", rope_cos, persistent=False)
            self.register_buffer("rope_sin_cached", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, d_model]
        """
        _, t, _ = x.shape
        if t > self.block_size:
            raise ValueError(f"T={t} > block_size={self.block_size}")

        mask = self.causal_mask[:t, :t].unsqueeze(0)
        rope_cos = rope_sin = None

        if self.use_rope:
            rope_cos = self.rope_cos_cached[:t, :].unsqueeze(0).unsqueeze(0)
            rope_sin = self.rope_sin_cached[:t, :].unsqueeze(0).unsqueeze(0)

        out = self.mha(x, x, mask=mask, rope_cos=rope_cos, rope_sin=rope_sin)
        return out
