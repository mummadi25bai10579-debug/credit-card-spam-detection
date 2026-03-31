"""
deep_models/ft_transformer.py
------------------------------
FT-Transformer  (Feature Tokenizer + Transformer)
Paper: "Revisiting Deep Learning Models for Tabular Data" — Gorishniy et al., 2021

Idea in plain English
---------------------
Every feature — numerical or categorical — gets turned into a small embedding
vector (the "token").  Then a standard Transformer encoder runs attention over
all those feature-tokens at once.  The [CLS] token that sits at position-0 is
read off at the end and passed to a linear head for classification.

Why it works well
-----------------
- Attention lets the model learn interactions between ANY pair of features
  without us having to hand-craft those interactions.
- Shared embedding dimension means numerical and categorical features live
  in the same space and interact on equal footing.

Architecture
------------
  Input (B, F)
    └─► FeatureTokenizer  →  (B, F+1, D)   ← +1 for the [CLS] token
          └─► N × TransformerEncoderLayer
                └─► CLS token  →  LayerNorm  →  Linear  →  logit
"""

import torch
import torch.nn as nn
from .base_trainer import base_fit


# ─────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Projects each scalar feature to a D-dimensional vector with a simple
    per-feature linear layer.  Also prepends a learnable [CLS] token.
    """
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        # one weight + bias per feature → (n_features, d_model)
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

        # the [CLS] token is a learnable embedding (batch will expand it)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  →  (B, F, D)
        tokens = x.unsqueeze(-1) * self.weight + self.bias   # broadcast
        # prepend CLS:  (1,1,D) → (B,1,D) → cat → (B, F+1, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class FTTransformerModel(nn.Module):
    """
    Full FT-Transformer.

    Parameters
    ----------
    n_features  : number of input features (F)
    d_model     : embedding / hidden dimension (default 128)
    n_heads     : number of attention heads (default 8)
    n_layers    : number of Transformer encoder layers (default 3)
    dropout     : dropout rate applied in attention + FFN
    """
    def __init__(
        self,
        n_features : int,
        d_model    : int = 128,
        n_heads    : int = 8,
        n_layers   : int = 3,
        dropout    : float = 0.1,
    ):
        super().__init__()

        self.tokenizer = FeatureTokenizer(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,   # standard 4× FFN expansion
            dropout         = dropout,
            activation      = 'gelu',
            batch_first     = True,          # (B, S, D) — much less confusing
            norm_first      = True,          # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm        = nn.LayerNorm(d_model)
        self.head        = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens   = self.tokenizer(x)               # (B, F+1, D)
        encoded  = self.transformer(tokens)         # (B, F+1, D)
        cls_out  = self.norm(encoded[:, 0, :])      # (B, D)  ← CLS position
        return self.head(cls_out)                   # (B, 1)


# ─────────────────────────────────────────────
# Public training function
# ─────────────────────────────────────────────

def train_ft_transformer(X, y, **kwargs) -> dict:
    """
    Build + train an FT-Transformer on tabular fraud data.

    Extra kwargs are forwarded to base_fit() so you can pass
    epochs, lr, batch_size, patience, device, etc. from train_model.py.
    """
    n_features = X.shape[1]

    model = FTTransformerModel(
        n_features = n_features,
        d_model    = 128,
        n_heads    = 8,
        n_layers   = 3,
        dropout    = 0.1,
    )

    return base_fit(model, X, y, model_name="FT-Transformer", **kwargs)
