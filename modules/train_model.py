"""
modules/train_model.py
-----------------------
Trains all classical ML models and (optionally) all five deep learning models,
then saves the best one as `best_model`.

Classical models   : Logistic Regression, Random Forest, XGBoost,
                     Isolation Forest, Voting Ensemble (RF + XGB)

Deep models        : FT-Transformer, TabTransformer, TabNet,
                     ResNet-MLP, NODE
                     (enabled when train_all(include_deep=True) is called)

The winner is decided by validation AUC.
"""

import os
import sys
import pickle
import numpy as np
import importlib
import matplotlib
matplotlib.use('Agg')    # no display needed on a server
import matplotlib.pyplot as plt

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier, VotingClassifier
from sklearn.metrics        import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score
)
from sklearn.preprocessing  import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost                import XGBClassifier
from sklearn.ensemble       import IsolationForest

from modules.data_loader import load_raw_data, get_features_and_labels
from modules.db_setup    import get_connection, log_model_run


# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODELS_DIR = "models"
PLOTS_DIR  = "plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)


def _require_torch():
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Deep training requested but PyTorch is not installed. "
            "Install it with: pip install torch"
        ) from exc


# ─────────────────────────────────────────────
# Save / Load helpers
# ─────────────────────────────────────────────

def save_model(model, name: str):
    """Pickle a classical sklearn / xgb model to models/<name>.pkl"""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"    saved → {path}")
    return path


def save_deep_model(state: dict, name: str):
    """
    For deep models we save a small dict containing:
      - 'weights' : the PyTorch state_dict
      - 'scaler'  : the fitted StandardScaler
      - 'type'    : architecture name (so predict.py knows how to reload)
    """
    torch = _require_torch()

    path = os.path.join(MODELS_DIR, f"{name}.pt")
    torch.save({
        'weights' : state['model'].state_dict(),
        'scaler'  : state['scaler'],
        'type'    : name,
        'config'  : state.get('config', {}),
    }, path)
    print(f"    saved → {path}")


def save_best(model_or_state, name: str, is_deep: bool = False):
    """Save whatever won the competition as best_model.*"""
    if is_deep:
        torch = _require_torch()
        path = os.path.join(MODELS_DIR, "best_model.pt")
        torch.save({
            'weights' : model_or_state['model'].state_dict(),
            'scaler'  : model_or_state['scaler'],
            'type'    : name,
        }, path)
        print(f"\n  🏆  Best model: {name}  →  models/best_model.pt")
    else:
        path = os.path.join(MODELS_DIR, "best_model.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model_or_state, f)
        print(f"\n  🏆  Best model: {name}  →  models/best_model.pkl")


# ─────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Legit', 'Fraud'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Legit', 'Fraud'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — {model_name}')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"cm_{model_name.replace(' ', '_')}.png"), dpi=120)
    plt.close()


def plot_precision_recall(y_true, y_scores, model_name: str = "All"):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, lw=2, color='steelblue')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall — {model_name}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"precision_recall_{model_name.replace(' ', '_')}.png"), dpi=120)
    plt.close()


def plot_feature_importance(model, feature_names: list, model_name: str):
    try:
        importances = (
            model.feature_importances_
            if hasattr(model, 'feature_importances_')
            else None
        )
        if importances is None:
            return
        idx = np.argsort(importances)[-15:]   # top 15
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([feature_names[i] for i in idx], importances[idx], color='steelblue')
        ax.set_title(f'Feature Importance — {model_name}')
        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOTS_DIR, f"feature_importance_{model_name.replace(' ', '_')}.png"),
            dpi=120
        )
        plt.close()
    except Exception:
        pass


# ─────────────────────────────────────────────
# Classical model training
# ─────────────────────────────────────────────

def train_classical_models(X_train, X_test, y_train, y_test, feature_names) -> dict:
    """
    Trains all five classical models, plots their confusion matrices,
    saves them to disk, and returns a dict of {name: auc_score}.
    """
    scaler    = StandardScaler()
    X_tr_sc   = scaler.fit_transform(X_train)
    X_te_sc   = scaler.transform(X_test)
    disable_xgb = (sys.version_info >= (3, 14)) or (os.getenv("DISABLE_XGBOOST", "0") == "1")
    if disable_xgb:
        print("  Note: Skipping XGBoost/Voting(RF+XGB) on this runtime for stability.")

    # ── define models ──────────────────────────────────────────────────────
    rf  = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                  random_state=42, n_jobs=-1)
    # Use a conservative config that avoids QuantileDMatrix instability on some setups.
    xgb = XGBClassifier(
        n_estimators=120,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        eval_metric='logloss',
        tree_method='approx',
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    lr  = LogisticRegression(max_iter=1000, class_weight='balanced',
                              random_state=42)
    iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    models_to_train = {
        'Random_Forest'        : (rf,  X_train, X_test,  True),
        'Logistic_Regression'  : (lr,  X_tr_sc, X_te_sc, True),
        'Isolation_Forest'     : (iso, X_train, X_test,  False),
    }
    if not disable_xgb:
        ens = VotingClassifier(
            estimators=[('rf', rf), ('xgb', xgb)],
            voting='soft'
        )
        models_to_train['XGBoost'] = (xgb, X_train, X_test, True)
        models_to_train['Voting_Ensemble_RFXGB'] = (ens, X_train, X_test, True)

    results = {}
    conn = get_connection()
    train_samples = int(len(y_train))
    test_samples = int(len(y_test))
    fraud_in_test = int((y_test == 1).sum())

    for name, (clf, X_tr, X_te, has_proba) in models_to_train.items():
        print(f"  Training {name} ...", end='  ')

        try:
            if name == 'Isolation_Forest':
                clf.fit(X_tr)
                preds  = (clf.predict(X_te) == -1).astype(int)
                scores = -clf.score_samples(X_te)
                auc    = roc_auc_score(y_test, scores)
            else:
                clf.fit(X_tr, y_train)
                preds  = clf.predict(X_te)
                scores = clf.predict_proba(X_te)[:, 1]
                auc    = roc_auc_score(y_test, scores)
        except KeyboardInterrupt:
            if name in ('XGBoost', 'Voting_Ensemble_RFXGB'):
                print("skipped (KeyboardInterrupt during XGBoost backend)")
                continue
            raise
        except Exception as exc:
            print(f"skipped ({exc.__class__.__name__}: {exc})")
            continue

        print(f"AUC={auc:.4f}")
        model_path = save_model(clf, name)
        metrics = {
            'accuracy': float(accuracy_score(y_test, preds)),
            'precision': float(precision_score(y_test, preds, zero_division=0)),
            'recall': float(recall_score(y_test, preds, zero_division=0)),
            'f1': float(f1_score(y_test, preds, zero_division=0)),
            'roc_auc': float(auc),
            'avg_precision': float(average_precision_score(y_test, scores)),
        }
        log_model_run(
            conn,
            name,
            metrics,
            train_samples,
            test_samples,
            fraud_in_test,
            model_path,
        )

        plot_confusion_matrix(y_test, preds, name)
        plot_feature_importance(clf, feature_names, name)
        results[name] = (auc, clf)

    conn.commit()
    conn.close()
    return results


# ─────────────────────────────────────────────
# Deep model training
# ─────────────────────────────────────────────

def train_deep_models(X_train, y_train) -> dict:
    """
    Trains all five deep models and returns {name: (auc, state_dict)}.
    Uses the entire training split — base_fit() handles its own val split.
    """
    _require_torch()

    # lazy imports so the rest of the pipeline works even without torch
    from deep_module.ft_transformer  import train_ft_transformer
    from deep_module.tab_transformer import train_tab_transformer
    from deep_module.tabnet_model    import train_tabnet
    from deep_module.resnet_mlp      import train_resnet_mlp
    from deep_module.node_model      import train_node

    deep_trainers = {
        'FT_Transformer'  : train_ft_transformer,
        'TabTransformer'  : train_tab_transformer,
        'TabNet'          : train_tabnet,
        'ResNet_MLP'      : train_resnet_mlp,
        'NODE'            : train_node,
    }

    results = {}
    for name, trainer_fn in deep_trainers.items():
        print(f"\n  ── {name} ──")
        state = trainer_fn(
            X_train, y_train,
            epochs     = 50,
            batch_size = 512,
            lr         = 1e-3,
            patience   = 10,
        )
        save_deep_model(state, name)
        results[name] = (state['auc'], state)

    return results


# ─────────────────────────────────────────────
# Master entry point
# ─────────────────────────────────────────────

def train_all(include_deep: bool = False) -> tuple[str, float]:
    """
    Full training pipeline.

    Parameters
    ----------
    include_deep : if True, also trains the five PyTorch deep models.

    Returns
    -------
    (best_model_name, best_auc)
    """
    print("\n  Loading data ...")
    df = load_raw_data()
    X, y, feature_names = get_features_and_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"  Fraud rate: {y.mean()*100:.2f}%")

    # ── classical ─────────────────────────────────────────────────────────
    print("\n  ── Classical Models ──")
    classical_results = train_classical_models(X_train, X_test, y_train, y_test, feature_names)

    best_name  = max(classical_results, key=lambda k: classical_results[k][0])
    best_auc   = classical_results[best_name][0]
    best_model = classical_results[best_name][1]
    best_deep  = False

    # ── deep  ─────────────────────────────────────────────────────────────
    if include_deep:
        print("\n  ── Deep Models ──")
        deep_results = train_deep_models(X_train, y_train)

        for name, (auc, state) in deep_results.items():
            if auc > best_auc:
                best_auc   = auc
                best_name  = name
                best_model = state
                best_deep  = True

    # ── save the winner ───────────────────────────────────────────────────
    save_best(best_model, best_name, is_deep=best_deep)

    # final PR curve (best classical model for quick reference)
    best_classical_name = max(classical_results, key=lambda k: classical_results[k][0])
    best_clf = classical_results[best_classical_name][1]
    try:
        scaler   = StandardScaler().fit(X_train)
        X_te_sc  = scaler.transform(X_test)
        X_te_use = X_test if hasattr(best_clf, 'feature_importances_') else X_te_sc
        if hasattr(best_clf, 'predict_proba'):
            scores = best_clf.predict_proba(X_te_use)[:, 1]
            plot_precision_recall(y_test, scores, best_classical_name)
    except Exception:
        pass

    print(f"\n  ── Summary ──")
    print(f"  Classical winners:")
    for name, (auc, _) in sorted(classical_results.items(), key=lambda x: -x[1][0]):
        marker = " ←" if name == best_classical_name else ""
        print(f"    {name:<30}  AUC={auc:.4f}{marker}")
    if include_deep:
        print(f"  Deep winners:")
        for name, (auc, _) in sorted(deep_results.items(), key=lambda x: -x[1][0]):
            marker = " ←" if name == best_name and best_deep else ""
            print(f"    {name:<30}  AUC={auc:.4f}{marker}")

    return best_name, best_auc
