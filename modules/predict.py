"""
modules/predict.py
-------------------
Loads whichever model won training (classical .pkl  or  deep .pt)
and exposes two prediction functions:

  predict_single(transaction_dict, threshold)  ->  one decision
  predict_batch(df, model, threshold, log_to_db)  ->  DataFrame with risk labels
"""

import os
import pickle
import inspect
import numpy as np
import pandas as pd

from modules.data_loader import get_features_and_labels
from modules.db_setup    import log_prediction

MODELS_DIR = "models"
_ARCHITECTURE_REGISTRY: dict = {}


def _require_torch():
    try:
        import torch
        return torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required only for deep (.pt) models. "
            "Use a classical .pkl best_model or install a PyTorch build "
            "compatible with your Python runtime."
        ) from exc


def _register_architectures():
    global _ARCHITECTURE_REGISTRY
    if _ARCHITECTURE_REGISTRY:
        return
    from deep_module.ft_transformer  import FTTransformerModel
    from deep_module.tab_transformer import TabTransformerModel
    from deep_module.tabnet_model    import TabNetModel
    from deep_module.resnet_mlp      import ResNetMLP
    from deep_module.node_model      import NODEModel
    _ARCHITECTURE_REGISTRY = {
        'FT_Transformer' : FTTransformerModel,
        'TabTransformer' : TabTransformerModel,
        'TabNet'         : TabNetModel,
        'ResNet_MLP'     : ResNetMLP,
        'NODE'           : NODEModel,
    }


# ── Deep model wrapper ────────────────────────────────────────────────────────

class DeepModelWrapper:
    def __init__(self, model, scaler, arch_name):
        torch = _require_torch()
        self.model     = model
        self.scaler    = scaler
        self.arch_name = arch_name
        self.device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device).eval()

    def predict_proba(self, X):
        torch = _require_torch()
        X_sc = self.scaler.transform(X)
        X_t  = torch.tensor(X_sc, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t).squeeze(-1).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        return np.stack([1 - probs, probs], axis=1)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    is_deep_model = True


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(name='best_model'):
    pkl_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    pt_path  = os.path.join(MODELS_DIR, f"{name}.pt")

    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)

    elif os.path.exists(pt_path):
        torch = _require_torch()
        _register_architectures()
        checkpoint = torch.load(pt_path, map_location='cpu')
        arch_name  = checkpoint['type']
        scaler     = checkpoint['scaler']
        weights    = checkpoint['weights']
        n_features = _infer_n_features(arch_name, weights)
        ModelClass = _ARCHITECTURE_REGISTRY.get(arch_name)
        if ModelClass is None:
            raise ValueError(f"Unknown deep architecture: {arch_name}")
        model = ModelClass(n_features=n_features)
        model.load_state_dict(weights)
        return DeepModelWrapper(model, scaler, arch_name)

    else:
        raise FileNotFoundError(
            f"No model found at {pkl_path} or {pt_path}. Run training first."
        )


def _infer_n_features(arch_name, weights):
    key_map = {
        'FT_Transformer' : 'tokenizer.weight',
        'TabTransformer' : 'embedder.weight',
        'TabNet'         : 'initial_bn.weight',
        'ResNet_MLP'     : 'input_proj.weight',
        'NODE'           : 'layers.0.trees.0.feature_selector',
    }
    key = key_map.get(arch_name)
    if key and key in weights:
        t = weights[key]
        return t.shape[-1] if t.dim() == 2 else t.shape[0]
    for k, v in weights.items():
        if 'weight' in k and v.dim() == 2 and v.shape[1] in range(5, 100):
            return v.shape[1]
    raise ValueError(f"Could not infer n_features for {arch_name}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_label(prob, threshold):
    if prob >= threshold:
        return "HIGH"
    elif prob >= threshold * 0.6:
        return "MEDIUM"
    return "LOW"


def _safe_log(i, row, actual):
    """
    Call log_prediction() regardless of whether your db_setup uses
    'transaction_id', 'transaction_ref', or something else entirely.
    We inspect the real signature at runtime so it never mismatches.
    """
    sig         = inspect.signature(log_prediction)
    param_names = list(sig.parameters.keys())

    # find the ID-like first parameter
    id_param = next(
        (p for p in param_names if any(w in p.lower() for w in ('id', 'ref', 'idx'))),
        param_names[0]
    )

    kwargs = {id_param: int(i)}

    # map common name variants to what the caller provides
    name_map = {
        'fraud_prob'  : float(row['fraud_prob']),
        'fraud_score' : float(row['fraud_prob']),
        'score'       : float(row['fraud_prob']),
        'predicted'   : int(row['prediction']),
        'prediction'  : int(row['prediction']),
        'pred'        : int(row['prediction']),
        'actual'      : actual,
        'true_label'  : actual,
        'label'       : actual,
        'risk_level'  : row['risk_level'],
        'risk'        : row['risk_level'],
    }

    for param in param_names:
        if param == id_param:
            continue
        if param in name_map:
            val = name_map[param]
            # skip optional None params if signature doesn't accept None
            if val is not None:
                kwargs[param] = val

    try:
        log_prediction(**kwargs)
    except Exception:
        pass   # logging must never crash predictions


# ── Public API ────────────────────────────────────────────────────────────────

def predict_single(transaction, threshold=0.4, model=None):
    if model is None:
        model = load_model('best_model')

    df_row  = pd.DataFrame([transaction])
    X, _, _ = get_features_and_labels(df_row)
    prob     = model.predict_proba(X)[0, 1]
    is_fraud = prob >= threshold

    print(f"  Fraud probability : {prob:.4f}")
    print(f"  Threshold         : {threshold}")
    print(f"  Risk level        : {_risk_label(prob, threshold)}")

    return {
        'prob'       : float(prob),
        'is_fraud'   : bool(is_fraud),
        'risk_level' : _risk_label(prob, threshold),
        'threshold'  : threshold,
    }


def predict_batch(df, model=None, threshold=0.4, log_to_db=True):
    if model is None:
        model = load_model('best_model')

    X, y_true, _ = get_features_and_labels(df)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    results               = df.copy()
    results['fraud_prob'] = probs
    results['prediction'] = preds
    results['risk_level'] = [_risk_label(p, threshold) for p in probs]

    if log_to_db:
        for i, row in results.iterrows():
            actual = int(y_true[i]) if i < len(y_true) else None
            _safe_log(i, row, actual)

    return results
