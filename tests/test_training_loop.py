import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.gpt_model import GPT2
from src.training.main_loop import train_gpt_lm


def _make_loader(vocab_size: int, block_size: int, num_batches: int = 4):
    seqs = torch.randint(0, vocab_size, (num_batches * 2, block_size))
    dataset = TensorDataset(seqs, seqs)
    return DataLoader(dataset, batch_size=2)


def test_train_gpt_lm_runs_on_cpu(tmp_path):
    vocab_size = 32
    block_size = 8
    train_loader = _make_loader(vocab_size, block_size)

    model = GPT2(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=1,
        n_head=2,
        d_model=32,
        dropout=0.0,
    )

    ckpt_path = tmp_path / "tiny_model.pt"
    history = train_gpt_lm(
        model,
        train_loader,
        val_loader=None,
        epochs=1,
        base_lr=1e-3,
        device="cpu",
        ckpt_path=str(ckpt_path),
        log_every=10,
        amp_enabled=False,
        gpt_version="gpt2",
        save_ckpt_every=1,
    )

    assert len(history["train_loss"]) == 1
    assert ckpt_path.exists()
