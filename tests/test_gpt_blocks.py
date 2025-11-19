import pytest
import torch

from src.model.attention import CausalSelfAttention
from src.model.gpt_blocks import GPT3Block


def test_gpt3_block_preserves_shape_and_backprop():
    block = GPT3Block(d_model=32, num_heads=4, block_size=16, dropout=0.0, resid_dropout=0.0)
    x = torch.randn(2, 16, 32, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None


def test_causal_attention_rejects_longer_sequences():
    attn = CausalSelfAttention(d_model=32, num_heads=4, block_size=8)
    x = torch.randn(1, 9, 32)
    with pytest.raises(ValueError):
        attn(x)
