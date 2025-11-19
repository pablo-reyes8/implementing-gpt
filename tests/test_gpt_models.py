import torch

from src.model.gpt_model import GPT2, GPT3


def _dummy_batch(vocab_size: int, block_size: int, batch_size: int = 2):
    x = torch.randint(low=0, high=vocab_size, size=(batch_size, block_size))
    return x


def test_gpt2_forward_and_weight_tying():
    vocab_size = 32
    block_size = 8
    model = GPT2(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, d_model=32)
    x = _dummy_batch(vocab_size, block_size)
    logits, loss = model(x, x)
    assert logits.shape == (x.size(0), block_size, vocab_size)
    assert loss is not None and loss.dim() == 0
    assert model.lm_head.weight.data_ptr() == model.emb.tok_emb.weight.data_ptr()


def test_gpt3_forward_matches_shapes():
    vocab_size = 48
    block_size = 12
    model = GPT3(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=2,
        n_head=4,
        d_model=64,
        dropout=0.0,
        resid_dropout=0.0,
    )
    x = _dummy_batch(vocab_size, block_size)
    logits, loss = model(x, x)
    assert logits.shape == (x.size(0), block_size, vocab_size)
    assert loss is not None and loss.dim() == 0
    assert model.lm_head.weight.data_ptr() == model.emb.tok_emb.weight.data_ptr()
