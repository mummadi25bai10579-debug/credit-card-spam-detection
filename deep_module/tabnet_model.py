"""
deep_models/tabnet_model.py
----------------------------
TabNet  (from scratch, no pytorch-tabnet dependency)
Paper: "TabNet: Attentive Interpretable Tabular Learning" — Arik & Pfister, Google, 2019

Idea in plain English
---------------------
TabNet processes data in sequential "steps".  At each step a soft attention
mask decides WHICH features to look at (like a differentiable feature
selector), and a small FC block processes only those features.  The masks
from all steps are summed into feature importances you can actually visualise.

This gives TabNet two nice properties:
  1. It's interpretable — you can see which features drove each decision.
  2. It does implicit feature selection, so it doesn't need heavy pre-processing.

Architecture per step
---------------------
  Input (B, F)
    └─► BN  →  prior_scales * attention_transformer(H)  →  soft mask M_i
          └─► M_i ⊙ x  →  feature_transformer(x_masked)
                └─► split(h_a, h_d)
                      h_a → update prior_scales (suppress already-used features)
                      h_d → accumulate into output

  final output: sum of h_d across all steps  →  Linear  →  logit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import base_fit


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class GLU(nn.Module):
    """
    Gated Linear Unit:  splits the last dim in half and gates one half with
    the other.  TabNet uses this instead of plain ReLU for richer activations.
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = x.size(-1) // 2
        return x[:, :half] * torch.sigmoid(x[:, half:])


class FeatureTransformer(nn.Module):
    """
    Two FC-BN-GLU blocks with a skip connection around the second block.
    Shared across all steps (shared_fc) + one step-specific block (step_fc).
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # shared part — trained once and reused every step (cheaper)
        self.shared_fc = nn.Sequential(
            nn.Linear(in_dim,  out_dim * 2, bias=False),
            nn.BatchNorm1d(out_dim * 2),
            GLU(),
        )
        # step-specific part — each step gets its own weights
        self.step_fc = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2, bias=False),
            nn.BatchNorm1d(out_dim * 2),
            GLU(),
        )
        self.scale = (0.5 ** 0.5)  # √0.5 residual scaling from the paper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared_fc(x)
        return (h + self.step_fc(h)) * self.scale


class AttentionTransformer(nn.Module):
    """
    Produces the soft feature-selection mask for one step.
    prior_scales penalises features that were already selected in earlier steps.
    """
    def __init__(self, in_dim: int, n_features: int, momentum: float = 0.02):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_features, bias=False)
        self.bn = nn.BatchNorm1d(n_features, momentum=momentum)

    def forward(self, h: torch.Tensor, prior_scales: torch.Tensor) -> torch.Tensor:
        # h: (B, in_dim),  prior_scales: (B, F)
        a = self.bn(self.fc(h))
        a = a * prior_scales           # suppress already-used features
        return F.softmax(a, dim=-1)    # (B, F) soft mask, sums to 1


# ─────────────────────────────────────────────
# Full TabNet model
# ─────────────────────────────────────────────

class TabNetModel(nn.Module):
    """
    TabNet classifier.

    Parameters
    ----------
    n_features  : number of input features (F)
    n_steps     : number of sequential attention steps (default 5)
    n_d         : width of the decision branch  (default 32)
    n_a         : width of the attention branch (default 32)
    gamma       : sparsity coefficient — how quickly prior_scales decay (default 1.3)
    momentum    : batch-norm momentum used in attention transformers
    """
    def __init__(
        self,
        n_features : int,
        n_steps    : int   = 5,
        n_d        : int   = 32,
        n_a        : int   = 32,
        gamma      : float = 1.3,
        momentum   : float = 0.02,
    ):
        super().__init__()
        self.n_steps    = n_steps
        self.gamma      = gamma
        self.n_features = n_features

        self.initial_bn = nn.BatchNorm1d(n_features, momentum=momentum)

        # One feature-transformer + one attention-transformer per step
        self.feat_transformers = nn.ModuleList([
            FeatureTransformer(n_features, n_d + n_a) for _ in range(n_steps)
        ])
        self.att_transformers = nn.ModuleList([
            AttentionTransformer(n_a, n_features, momentum) for _ in range(n_steps)
        ])

        self.final_head = nn.Linear(n_d, 1)
        self.n_d = n_d
        self.n_a = n_a

    def forward(self, x: torch.Tensor):
        B = x.size(0)
        x = self.initial_bn(x)

        # prior_scales starts at 1 everywhere → all features equally available
        prior_scales = torch.ones(B, self.n_features, device=x.device)

        # h_a is the "attention buffer" — seeded with zeros
        h_a = torch.zeros(B, self.n_a, device=x.device)

        # accumulate decision outputs across all steps
        total_output = torch.zeros(B, self.n_d, device=x.device)

        for step in range(self.n_steps):
            # ── attention mask for this step ────────────────────────────
            M = self.att_transformers[step](h_a, prior_scales)   # (B, F)

            # update prior_scales so the next step looks at *other* features
            prior_scales = prior_scales * (self.gamma - M)

            # ── feature transformer on masked input ────────────────────
            x_masked = M * x                                      # (B, F)
            h = self.feat_transformers[step](x_masked)            # (B, n_d+n_a)

            # split into decision (h_d) and attention (h_a) branches
            h_d = F.relu(h[:, :self.n_d])
            h_a =        h[:, self.n_d:]

            total_output += h_d

        return self.final_head(total_output)   # (B, 1)


# ─────────────────────────────────────────────
# Public training function
# ─────────────────────────────────────────────

def train_tabnet(X, y, **kwargs) -> dict:
    """
    Build + train a TabNet on tabular fraud data.
    """
    model = TabNetModel(
        n_features = X.shape[1],
        n_steps    = 5,
        n_d        = 32,
        n_a        = 32,
        gamma      = 1.3,
    )
    return base_fit(model, X, y, model_name="TabNet", **kwargs)
