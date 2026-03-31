"""
deep_models/base_trainer.py
---------------------------
Everything that all five deep models share:
  - TabularDataset  : wraps a numpy array pair into a PyTorch Dataset
  - EarlyStopping   : stops training when val-loss stops improving
  - evaluate_model  : computes AUC, F1, precision, recall on a DataLoader
  - base_fit        : one standard training loop used by every architecture

Keep this file simple.  Individual architecture files only define the
nn.Module and then call base_fit() to do the actual training.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TabularDataset(Dataset):
    """
    Plain wrapper that takes numpy X and y arrays and makes them look
    like a PyTorch dataset.  Nothing fancy needed for tabular data.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# Early Stopping
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Watches validation loss and stops training if it hasn't improved
    for `patience` consecutive epochs.  Also saves the best weights
    so we restore them at the end.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_loss  = float('inf')
        self.counter    = 0
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            # deep-copy the weights so we can restore later
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ─────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────

def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """
    Runs the model on every batch in `loader` and returns a dict with
    auc, f1, precision, recall, and average loss.
    """
    model.eval()
    criterion = nn.BCEWithLogitsLoss()

    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).squeeze(-1)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item()

            all_logits.append(logits.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    logits_np = np.concatenate(all_logits)
    labels_np = np.concatenate(all_labels)
    probs_np  = 1 / (1 + np.exp(-logits_np))   # sigmoid
    preds_np  = (probs_np >= 0.5).astype(int)

    return {
        'loss'      : total_loss / len(loader),
        'auc'       : roc_auc_score(labels_np, probs_np),
        'f1'        : f1_score(labels_np, preds_np, zero_division=0),
        'precision' : precision_score(labels_np, preds_np, zero_division=0),
        'recall'    : recall_score(labels_np, preds_np, zero_division=0),
    }


# ─────────────────────────────────────────────
# Shared training loop
# ─────────────────────────────────────────────

def base_fit(
    model      : nn.Module,
    X          : np.ndarray,
    y          : np.ndarray,
    model_name : str,
    *,
    epochs     : int   = 50,
    batch_size : int   = 512,
    lr         : float = 1e-3,
    patience   : int   = 10,
    val_frac   : float = 0.15,
    device     : torch.device | None = None,
) -> dict:
    """
    One standard training loop shared by every architecture in this folder.

    Steps
    -----
    1. Scale features with StandardScaler
    2. Split into train / validation
    3. Build DataLoaders
    4. Train with AdamW + cosine LR schedule + early stopping
    5. Restore best weights and return evaluation metrics

    Returns a dict with the best val metrics plus 'scaler' so predict
    can inverse-transform features later.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 1. scale ──────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 2. split ──────────────────────────────────────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_scaled, y, test_size=val_frac, random_state=42, stratify=y
    )

    # ── 3. loaders ────────────────────────────────────────────────────────
    train_loader = DataLoader(
        TabularDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TabularDataset(X_val, y_val),
        batch_size=batch_size * 2,
        shuffle=False,
    )

    # ── 4. optimizer + scheduler ──────────────────────────────────────────
    model     = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss(
        # fraud is rare → up-weight positive class
        pos_weight=torch.tensor([y.sum() / max(1, (1 - y).sum())]).to(device) * 3
    )
    stopper = EarlyStopping(patience=patience)

    # ── 5. train ──────────────────────────────────────────────────────────
    print(f"\n  Training {model_name} on {device}  "
          f"({X_tr.shape[0]:,} train | {X_val.shape[0]:,} val)")
    print(f"  {'Epoch':>5}  {'Train Loss':>11}  {'Val Loss':>9}  {'Val AUC':>8}")
    print(f"  {'─'*5}  {'─'*11}  {'─'*9}  {'─'*8}")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch).squeeze(-1)
            loss   = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # evaluate every 5 epochs so the console isn't flooded
        if epoch % 5 == 0 or epoch == 1:
            val_metrics = evaluate_model(model, val_loader, device)
            avg_train   = train_loss / len(train_loader)
            print(f"  {epoch:>5}  {avg_train:>11.4f}  "
                  f"{val_metrics['loss']:>9.4f}  {val_metrics['auc']:>8.4f}")

            if stopper(val_metrics['loss'], model):
                print(f"  ↳ early stopping at epoch {epoch}")
                break

    stopper.restore_best(model)
    elapsed = time.time() - t0

    # final validation pass with best weights
    final_metrics = evaluate_model(model, val_loader, device)
    print(f"\n  ✓ {model_name} finished in {elapsed:.1f}s")
    print(f"    AUC={final_metrics['auc']:.4f}  "
          f"F1={final_metrics['f1']:.4f}  "
          f"Precision={final_metrics['precision']:.4f}  "
          f"Recall={final_metrics['recall']:.4f}")

    final_metrics['scaler'] = scaler
    final_metrics['model']  = model
    return final_metrics
