"""
deep_models/resnet_mlp.py
--------------------------
ResNet-style MLP for tabular data
Reference: "Revisiting Deep Learning Models for Tabular Data" — Gorishniy et al., 2021
           (where it's called simply "ResNet")

Idea in plain English
---------------------
A stack of residual blocks, each of which looks like:

    x  ──► BN → Linear → ReLU → Dropout → Linear → Dropout
    │                                                  │
    └──────────────── (skip connection) ───────────────┘
                                │
                               (+)  → output of block

The skip connection means the gradient can flow directly from the loss back to
every block without vanishing.  This lets us go deeper than a plain MLP.

A 1-D "downsampling" linear projects x to the right width if the block changes
the hidden dimension (same idea as ResNet's 1×1 convolution shortcut).

Architecture
------------
  Input (B, F)
    └─► Linear → (B, hidden)        ← input projection to hidden dim
          └─► N × ResidualBlock     ← each block: BN-Lin-ReLU-Drop-Lin-Drop + skip
                └─► BN → Linear → logit
"""

import torch
import torch.nn as nn
from .base_trainer import base_fit


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    One residual block:
      - BatchNorm before the first linear (pre-norm style)
      - Two linear layers with ReLU in between
      - Dropout after each linear
      - Skip connection with optional projection if in_dim ≠ out_dim
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()

        self.block = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim,     hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout),
        )

        # project the skip if dimensions change
        self.skip = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


# ─────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────

class ResNetMLP(nn.Module):
    """
    ResNet-style MLP.

    Parameters
    ----------
    n_features  : number of input features
    hidden_dim  : width of every residual block (default 256)
    n_blocks    : how many residual blocks to stack (default 4)
    dropout     : dropout probability inside each block
    """
    def __init__(
        self,
        n_features : int,
        hidden_dim : int   = 256,
        n_blocks   : int   = 4,
        dropout    : float = 0.1,
    ):
        super().__init__()

        # ── input projection ─────────────────────────────────────────────
        # Lift the raw feature vector to the model's hidden dimension.
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # ── residual blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            ResidualBlock(
                in_dim     = hidden_dim,
                hidden_dim = hidden_dim * 2,   # expand then contract (bottleneck-lite)
                out_dim    = hidden_dim,
                dropout    = dropout,
            )
            for _ in range(n_blocks)
        ])

        # ── classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)          # (B, hidden_dim)
        for block in self.blocks:
            h = block(h)                # (B, hidden_dim)
        return self.head(h)             # (B, 1)


# ─────────────────────────────────────────────
# Public training function
# ─────────────────────────────────────────────

def train_resnet_mlp(X, y, **kwargs) -> dict:
    """
    Build + train a ResNet-style MLP on tabular fraud data.

    This is often the strongest baseline among deep models — don't underestimate it!
    """
    model = ResNetMLP(
        n_features = X.shape[1],
        hidden_dim = 256,
        n_blocks   = 4,
        dropout    = 0.1,
    )
    return base_fit(model, X, y, model_name="ResNet-MLP", **kwargs)
