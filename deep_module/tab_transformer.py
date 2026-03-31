"""
deep_models/tab_transformer.py
--------------------------------
TabTransformer
Paper: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings" — Huang et al., 2020

Idea in plain English
---------------------
Only the CATEGORICAL features get turned into embedding tokens and processed
by a Transformer.  Numerical features are just standardised and kept as-is.
At the end both halves are concatenated and fed into a small MLP head.

This is simpler than FT-Transformer (which tokenizes everything) but it was
one of the first papers to show Transformers could beat tree models on tabular
data when there are many categorical columns.

In this fraud dataset all 30 features are numerical (V1-V28 + Amount + Time).
We treat the first 10 features as "pseudo-categorical" embeddings so the
architecture still runs and you can see the full code path.  In a real project
you'd pass in the categorical column indices explicitly.

Architecture
------------
  Categorical features (B, C)
    └─► Embedding table  →  (B, C, D)
          └─► N × TransformerEncoderLayer  →  (B, C, D)
                └─► flatten  →  (B, C*D)

  Numerical features (B, N)  ──────────────────────────────────────┐
                                                                    ▼
                                                     concat  →  (B, C*D + N)
                                                       └─► MLP  →  logit
"""

import torch
import torch.nn as nn
from .base_trainer import base_fit


# ─────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────

class CategoricalEmbedder(nn.Module):
    """
    Treats each of the `n_cat` features as if it were a discrete token by
    projecting each scalar value through a per-feature linear layer (same as
    FT-Transformer's FeatureTokenizer but without the CLS token).
    """
    def __init__(self, n_cat: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_cat, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_cat, d_model))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)  →  (B, C, D)
        return x.unsqueeze(-1) * self.weight + self.bias


class TabTransformerModel(nn.Module):
    """
    TabTransformer.

    Parameters
    ----------
    n_features  : total number of input features
    n_cat       : how many of those features get the Transformer treatment
    d_model     : embedding size for categorical tokens
    n_heads     : attention heads
    n_layers    : Transformer encoder layers
    dropout     : dropout rate
    """
    def __init__(
        self,
        n_features : int,
        n_cat      : int  = 10,
        d_model    : int  = 32,
        n_heads    : int  = 4,
        n_layers   : int  = 3,
        dropout    : float = 0.1,
    ):
        super().__init__()
        self.n_cat = n_cat
        n_num      = n_features - n_cat   # leftover numerical features

        # ── categorical branch ────────────────────────────────────────────
        self.embedder = CategoricalEmbedder(n_cat, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * 4,
            dropout         = dropout,
            activation      = 'relu',
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── MLP head  ─────────────────────────────────────────────────────
        cat_out_dim = n_cat * d_model
        mlp_in_dim  = cat_out_dim + n_num

        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_in_dim),
            nn.Linear(mlp_in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cat = x[:, :self.n_cat]         # (B, n_cat)
        x_num = x[:, self.n_cat:]         # (B, n_num)

        # categorical path
        cat_tokens   = self.embedder(x_cat)           # (B, n_cat, D)
        cat_encoded  = self.transformer(cat_tokens)   # (B, n_cat, D)
        cat_flat     = cat_encoded.flatten(1)          # (B, n_cat * D)

        # merge and classify
        combined = torch.cat([cat_flat, x_num], dim=1)  # (B, n_cat*D + n_num)
        return self.mlp(combined)                        # (B, 1)


# ─────────────────────────────────────────────
# Public training function
# ─────────────────────────────────────────────

def train_tab_transformer(X, y, **kwargs) -> dict:
    """
    Build + train a TabTransformer on tabular fraud data.
    """
    n_features = X.shape[1]
    n_cat      = min(10, n_features // 2)   # sensible default

    model = TabTransformerModel(
        n_features = n_features,
        n_cat      = n_cat,
        d_model    = 32,
        n_heads    = 4,
        n_layers   = 3,
        dropout    = 0.1,
    )

    return base_fit(model, X, y, model_name="TabTransformer", **kwargs)
