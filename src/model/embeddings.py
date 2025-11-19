import torch
import torch.nn as nn

class GPT2Embeddings(nn.Module):
    """
    Embeddings estilo GPT-2:
      - token embeddings aprendidos
      - positional embeddings aprendidos
      - dropout opcional

    input_ids: LongTensor [B, T]
    return:    FloatTensor [B, T, d_model]
    """
    def __init__(self, vocab_size: int, d_model: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size
        self.d_model = d_model

    def forward(self, input_ids: torch.Tensor):
        """
        input_ids: [B, T] con IDs de tokens.
        """
        
        B, T = input_ids.shape
        if T > self.block_size:
            raise ValueError(f"Secuencia T={T} > block_size={self.block_size}")

        # posiciones [0, 1, ..., T-1]
        device = input_ids.device
        pos_ids = torch.arange(0, T, device=device).unsqueeze(0)  # [1, T]
        pos_ids = pos_ids.expand(B, T)  

        tok = self.tok_emb(input_ids)  # [B, T, d_model]
        pos = self.pos_emb(pos_ids)   
        x = tok + pos                  # [B, T, d_model]

        x = self.dropout(x)
        return x
    
