import torch
import torch.nn as nn


class GPT2Embeddings(nn.Module):
    """
    Embeddings estilo GPT:
      - token embeddings aprendidos
      - positional embeddings aprendidos (opcional)
      - dropout opcional

    input_ids: LongTensor [B, T]
    return:    FloatTensor [B, T, d_model]
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        block_size: int,
        dropout: float = 0.1,
        pos_encoding: str = "learned",
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size
        self.d_model = d_model
        self.pos_encoding = (pos_encoding or "learned").lower()

        if self.pos_encoding not in {"learned", "rope", "none"}:
            raise ValueError("pos_encoding debe ser 'learned', 'rope' o 'none'")

        self.pos_emb = (
            nn.Embedding(block_size, d_model)
            if self.pos_encoding == "learned"
            else None
        )

    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: [B, T] con IDs de tokens.
        """
        bsz, seq_len = input_ids.shape
        if seq_len > self.block_size:
            raise ValueError(f"Secuencia T={seq_len} > block_size={self.block_size}")

        tok = self.tok_emb(input_ids)  # [B, T, d_model]

        if self.pos_emb is not None:
            device = input_ids.device
            pos_ids = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
            pos = self.pos_emb(pos_ids)
            tok = tok + pos

        tok = self.dropout(tok)
        return tok
