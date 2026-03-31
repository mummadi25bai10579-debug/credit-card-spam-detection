"""
deep_models/node_model.py
--------------------------
NODE  (Neural Oblivious Decision Ensembles)
Paper: "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data"
       — Popov et al., Yandex, ICLR 2020

Idea in plain English
---------------------
Gradient-boosted decision trees (XGBoost, LightGBM) dominate tabular data
because they naturally learn split conditions on individual features.
NODE makes this differentiable so we can train it end-to-end with backprop.

Key concept — "Oblivious Decision Tree" (ODT)
An oblivious tree asks the SAME question at every node on the same depth.
So depth-3 tree has 3 feature thresholds and 8 leaves.
This constraint makes the tree cheaper to compute AND easier to learn.

How NODE makes it differentiable
1. Instead of hard splits (left / right), use a smooth sigmoid gate.
2. The probability of being in each leaf is the product of those gates.
3. Leaf values are learnable parameters.

A NODE layer stacks many such ODTs (a "forest") side by side,
and multiple NODE layers can be stacked like a deep network.

Architecture
------------
  Input (B, F)
    └─► NODELayer_1  →  (B, n_trees * tree_dim)
          └─► NODELayer_2  →  ...
                └─► mean over trees  →  Linear  →  logit
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import base_fit


# ─────────────────────────────────────────────
# Differentiable oblivious split
# ─────────────────────────────────────────────

class ObliviousDecisionTree(nn.Module):
    """
    One differentiable oblivious decision tree.

    Parameters
    ----------
    n_features : F — number of input features
    depth      : how many split levels (creates 2^depth leaves)
    tree_dim   : output width per tree (number of outputs per leaf)
    """
    def __init__(self, n_features: int, depth: int = 4, tree_dim: int = 1):
        super().__init__()
        self.depth    = depth
        self.tree_dim = tree_dim
        n_leaves      = 2 ** depth

        # ── learnable parameters ──────────────────────────────────────────
        # which feature each of the `depth` splits applies to
        self.feature_selector = nn.Parameter(torch.empty(depth, n_features))
        nn.init.uniform_(self.feature_selector, -0.5, 0.5)

        # threshold for each split (one scalar per depth level)
        self.thresholds = nn.Parameter(torch.zeros(depth))

        # log-scale temperature per split for the sigmoid gate
        self.log_temp = nn.Parameter(torch.zeros(depth))

        # output value per leaf, per output dim
        self.leaf_values = nn.Parameter(torch.empty(n_leaves, tree_dim))
        nn.init.normal_(self.leaf_values, std=0.01)

        # precompute which leaf gets +1 or -1 at each depth
        # shape: (depth, n_leaves)
        indices = torch.arange(n_leaves)
        signs   = torch.stack([
            ((indices >> d) & 1).float() * 2 - 1   # +1 or -1
            for d in range(depth)
        ])
        self.register_buffer('signs', signs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── choose features: soft mixture via softmax over feature scores
        selector_probs = F.softmax(self.feature_selector, dim=-1)  # (depth, F)
        selected = torch.einsum('bi,di->bd', x, selector_probs)    # (B, depth)

        # ── compute smooth gate for each split ───────────────────────────
        temp  = torch.exp(self.log_temp).clamp(min=0.01)            # (depth,)
        gates = torch.sigmoid((selected - self.thresholds) * temp)  # (B, depth)

        # ── accumulate leaf probability ───────────────────────────────────
        # For each leaf: product of gate or (1-gate) across depth levels.
        # We do this in log-space for numerical stability.
        # signs[d, leaf] = +1 means we go "right" at depth d for this leaf.
        # gate^{+1} = gate,  gate^{-1} via 1-gate.
        # In log: log_prob[leaf] = Σ_d signs[d,leaf] * log(gate[d])
        #       but we need p(left) = 1-gate and p(right) = gate.
        # Easier:  p(leaf) = Π gate[d]^{I(leaf goes right at d)} * (1-gate[d])^{...}

        # gates: (B, depth) → (B, depth, 1) broadcast with signs (depth, n_leaves)
        g = gates.unsqueeze(-1)                                  # (B, depth, 1)
        # for each leaf, decide gate or 1-gate
        leaf_gates = torch.where(
            self.signs.unsqueeze(0) > 0,   # (1, depth, n_leaves) right?
            g.expand(-1, -1, 2**self.depth),
            1 - g.expand(-1, -1, 2**self.depth),
        )                                                        # (B, depth, n_leaves)

        # product over depth dimension in log space then exp
        log_leaf_prob = leaf_gates.clamp(min=1e-8).log().sum(dim=1)   # (B, n_leaves)
        leaf_prob     = log_leaf_prob.exp()                             # (B, n_leaves)

        # ── weighted sum over leaf values ─────────────────────────────────
        output = torch.einsum('bl,lo->bo', leaf_prob, self.leaf_values)  # (B, tree_dim)
        return output


# ─────────────────────────────────────────────
# NODE layer = ensemble of trees
# ─────────────────────────────────────────────

class NODELayer(nn.Module):
    """
    A layer of n_trees oblivious decision trees, each seeing a linear
    projection of the full input (so trees can attend to all features).
    Outputs are concatenated: (B, n_trees * tree_dim).
    """
    def __init__(
        self,
        n_features : int,
        n_trees    : int = 128,
        depth      : int = 4,
        tree_dim   : int = 1,
    ):
        super().__init__()
        self.trees = nn.ModuleList([
            ObliviousDecisionTree(n_features, depth, tree_dim)
            for _ in range(n_trees)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # each tree: (B, tree_dim) → stack → (B, n_trees, tree_dim) → flatten
        tree_outputs = torch.stack([t(x) for t in self.trees], dim=1)  # (B, n_trees, D)
        return tree_outputs.flatten(1)                                   # (B, n_trees*D)


# ─────────────────────────────────────────────
# Full NODE model
# ─────────────────────────────────────────────

class NODEModel(nn.Module):
    """
    Stacked NODE layers followed by a linear head.

    Parameters
    ----------
    n_features  : F — raw input width
    n_layers    : how many NODE layers to stack (default 2)
    n_trees     : trees per layer (default 128)
    depth       : tree depth (default 4  → 16 leaves)
    tree_dim    : output dim per tree (default 1)
    """
    def __init__(
        self,
        n_features : int,
        n_layers   : int = 2,
        n_trees    : int = 128,
        depth      : int = 4,
        tree_dim   : int = 1,
    ):
        super().__init__()
        layer_out = n_trees * tree_dim

        layers = []
        in_dim = n_features
        for _ in range(n_layers):
            layers.append(NODELayer(in_dim, n_trees, depth, tree_dim))
            # each layer's output gets concatenated with the original input
            # (dense connectivity, same idea as DenseNet)
            in_dim += layer_out

        self.layers = nn.ModuleList(layers)
        self.head   = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            out = layer(h)
            h   = torch.cat([h, out], dim=1)   # dense connection
        return self.head(h)


# ─────────────────────────────────────────────
# Public training function
# ─────────────────────────────────────────────

def train_node(X, y, **kwargs) -> dict:
    """
    Build + train a NODE model on tabular fraud data.

    NODE is slower than the other models (lots of small tree operations)
    so we default to fewer epochs and a smaller ensemble.
    """
    # NODE is memory-heavy; use smaller config if GPU not available
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_trees = 128 if device.type == 'cuda' else 64

    model = NODEModel(
        n_features = X.shape[1],
        n_layers   = 2,
        n_trees    = n_trees,
        depth      = 4,
        tree_dim   = 1,
    )

    # NODE needs a slightly lower LR and more epochs to converge
    kwargs.setdefault('epochs',  80)
    kwargs.setdefault('lr',      5e-4)
    kwargs.setdefault('patience', 15)

    return base_fit(model, X, y, model_name="NODE", device=device, **kwargs)
