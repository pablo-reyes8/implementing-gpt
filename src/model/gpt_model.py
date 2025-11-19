import torch.nn.functional as F
import torch
import torch.nn as nn
from src.model.attention import *
from src.model.gpt_blocks import *
from src.model.embeddings import *
import math

def init_gptmini_weights(model):
    with torch.no_grad():
        for name, module in model.named_modules():
            # Embeddings 
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # Linears
            elif isinstance(module, nn.Linear):
                if name == "lm_head":
                    # pesos pequeños
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                else:
                    # Xavier 
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)


def init_gpt3_weights(
    model: nn.Module,
    n_layer: int | None = None,
    *,
    residual_scale: bool = True,
    base_std: float = 0.02):
    """
    Inicialización 'GPT-3 style' genérica para modelos decoder-only tipo GPT.

    - Embeddings: N(0, base_std)
    - Linears en bloques residuales: N(0, base_std * scale), bias = 0
    - lm_head: N(0, base_std), bias = 0
    - LayerNorm: gamma = 1, beta = 0

    Donde:
      scale = 1 / sqrt(2 * n_layer) si residual_scale=True
            = 1                    si residual_scale=False

    * n_layer:
        - Si no se pasa, intenta inferirlo de `len(model.blocks)` si existe.
        - Si no, asume 1 (no escala).
    """

    if n_layer is None:
        if hasattr(model, "blocks") and isinstance(model.blocks, nn.ModuleList):
            n_layer = len(model.blocks)
        else:
            # fallback seguro: sin escalado
            n_layer = 1
            residual_scale = False

    scale = 1.0
    if residual_scale:
        scale = 1.0 / math.sqrt(2.0 * n_layer)

    with torch.no_grad():
        for name, module in model.named_modules():

            # ---- Embeddings ----
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=base_std)

            # ---- LayerNorm ----
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

            # ---- Linears ----
            elif isinstance(module, nn.Linear):
                if name.endswith("lm_head") or name == "lm_head":
                    std = base_std
                else:
                    std = base_std * scale

                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class GPT2(nn.Module):
    """
    GPT-2 'mini' decoder-only:
      - Embeddings (token + posición aprendida)
      - n_layers de GPT2Block
      - LayerNorm final
      - LM head (atado a los embeddings de token)
    """
    
    def __init__(self,
                 vocab_size: int,
                 block_size: int,
                 n_layer: int = 4,
                 n_head: int = 4,
                 d_model: int = 256,
                 dropout: float = 0.1,
                 layernorm_eps: float = 1e-5):
        
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model    = d_model

        self.emb = GPT2Embeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            block_size=block_size,
            dropout=dropout)

        self.blocks = nn.ModuleList([
            GPT2Block(
                d_model=d_model,
                num_heads=n_head,
                block_size=block_size,
                d_ff=4*d_model,
                dropout=dropout,
                layernorm_eps=layernorm_eps,) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(d_model, eps=layernorm_eps)

        # LM head: proyecta representaciones a logits de vocabulario
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: compartir pesos con embedding de tokens
        init_gptmini_weights(self)
        self.lm_head.weight = self.emb.tok_emb.weight

    def forward(self, idx: torch.Tensor, targets = None):
      """
      idx:     [B, T] con IDs de tokens de entrada
      targets: [B, T] con IDs objetivo (shifted) o None

      Returns:
        logits: [B, T, vocab_size]
        loss:   escalar (si targets no es None), sino None
      """
      B, T = idx.shape
      if T > self.block_size:
          raise ValueError(f"Secuencia demasiado larga: T={T}, block_size={self.block_size}")

      # Embeddings token + posición
      x = self.emb(idx)  # [B, T, d_model]

      # Pasar por los bloques GPT-2
      for block in self.blocks:
          x = block(x)    # [B, T, d_model]

      # LayerNorm final
      x = self.ln_f(x)   # [B, T, d_model]

      # LM head -> logits
      logits = self.lm_head(x)  # [B, T, vocab_size]

      loss = None
      if targets is not None:
          # Cross-entropy autoregresiva
          logits_flat  = logits.view(-1, self.vocab_size)
          targets_flat = targets.view(-1)
          # Opcional: ignorar padding tokens si usas ignore_index
          loss = F.cross_entropy(logits_flat, targets_flat)

      return logits, loss


class GPT3(nn.Module):
    """
    GPT-3 decoder-only inspirado en el paper original:
      - Embeddings aprendidos compartidos con lm_head
      - Bloques GPT3Block con MLP_gpt3 + dropout residual
      - LayerNorm final
    """
    def __init__(self,
                 vocab_size: int,
                 block_size: int,
                 n_layer: int = 24,
                 n_head: int = 16,
                 d_model: int = 1024,
                 dropout: float = 0.1,
                 layernorm_eps: float = 1e-5,
                 mlp_expansion: int = 4,
                 resid_dropout: float | None = None):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model

        self.emb = GPT2Embeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            block_size=block_size,
            dropout=dropout)

        self.blocks = nn.ModuleList([
            GPT3Block(
                d_model=d_model,
                num_heads=n_head,
                block_size=block_size,
                expansion=mlp_expansion,
                dropout=dropout,
                resid_dropout=resid_dropout,
                layernorm_eps=layernorm_eps,) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(d_model, eps=layernorm_eps)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        init_gpt3_weights(self)
        self.lm_head.weight = self.emb.tok_emb.weight #tying

    def forward(self, idx: torch.Tensor, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Secuencia demasiado larga: T={T}, block_size={self.block_size}")

        x = self.emb(idx)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.view(-1, self.vocab_size)
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss
