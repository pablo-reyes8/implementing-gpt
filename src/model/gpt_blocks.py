import torch.nn.functional as F
import torch
import torch.nn as nn
from src.model.attention import *

class GPT2MLP(nn.Module):
    """
    MLP posición a posición estilo GPT-2:
      Linear(d_model → d_ff) + GELU + Dropout + Linear(d_ff → d_model)
    """
    def __init__(self, d_model: int, d_ff = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # típico en GPT

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
    def __init__(self, d_model: int, num_heads: int, block_size: int,
                 d_ff = None, dropout: float = 0.1, layernorm_eps: float = 1e-5):
      
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.ln_2 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.attn = CausalSelfAttention(d_model=d_model, num_heads=num_heads,
                                        block_size=block_size, dropout=dropout)
        
        self.mlp  = GPT2MLP(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor):
      # Pre-LN + atención causal
      normalized_1 = self.ln_1(x)
      attn_output = self.attn(normalized_1)
      x = x + attn_output # residual conection
      
      # Pre-LN + MLP
      normalized_2 = self.ln_2(x)
      mlp_output = self.mlp(normalized_2)
      x = x + mlp_output # residual conection
    
      return x
    

class GPT3Block(nn.Module):
    """
    Bloque GPT-3:
      LayerNorm -> CausalSelfAttention -> Residual
      LayerNorm -> MLP_gpt3          -> Residual
    """
    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 block_size: int,
                 expansion: int = 4,
                 dropout: float = 0.1,
                 layernorm_eps: float = 1e-5,
                 resid_dropout: float | None = None):

        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.ln_2 = nn.LayerNorm(d_model, eps=layernorm_eps)

        self.attn = CausalSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            block_size=block_size,
            dropout=dropout)

        self.mlp = MLP_gpt3(d_model=d_model, expansion=expansion, dropout=dropout)

        dropout_value = dropout if resid_dropout is None else resid_dropout
        self.attn_resid = nn.Dropout(dropout_value)
        self.mlp_resid = nn.Dropout(dropout_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.attn(self.ln_1(x))
        x = x + self.attn_resid(attn_output)

        mlp_output = self.mlp(self.ln_2(x))
        x = x + self.mlp_resid(mlp_output)

        return x