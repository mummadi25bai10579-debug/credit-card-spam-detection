# Credit Card Fraud Detection System

A complete end-to-end fraud detection pipeline built with Python, SQLite, scikit-learn, XGBoost, and PyTorch. The system trains both classical machine learning models and five state-of-the-art deep learning architectures on tabular transaction data, automatically picks the best performer, and generates reports with predictions logged to a local database.

---

## Project Structure

```
credit card fake detection/
│
├── dataset/
│   └── creditcard.csv               ← raw transaction data (download separately)
│
├── models/                          ← saved model files after training
│   ├── best_model.pkl / .pt         ← winner of all trained models
│   ├── Random_Forest.pkl
│   ├── XGBoost.pkl
│   ├── Logistic_Regression.pkl
│   ├── Isolation_Forest.pkl
│   ├── Voting_Ensemble_RFXGB.pkl
│   ├── FT_Transformer.pt            ← saved only when --deep is used
│   ├── TabTransformer.pt
│   ├── TabNet.pt
│   ├── ResNet_MLP.pt
│   └── NODE.pt
│
├── modules/                         ← core pipeline code
│   ├── __init__.py
│   ├── data_loader.py               ← loads CSV, splits features and labels
│   ├── db_setup.py                  ← SQLite table creation and logging
│   ├── train_model.py               ← trains all models, saves best
│   ├── predict.py                   ← batch and single-transaction prediction
│   └── report.py                    ← generates summary reports
│
├── deep_models/                     ← five PyTorch architectures (new)
│   ├── __init__.py
│   ├── base_trainer.py              ← shared training loop for all deep models
│   ├── ft_transformer.py            ← FT-Transformer
│   ├── tab_transformer.py           ← TabTransformer
│   ├── tabnet_model.py              ← TabNet (built from scratch)
│   ├── resnet_mlp.py                ← ResNet-style MLP
│   └── node_model.py                ← NODE (differentiable decision trees)
│
├── plots/                           ← confusion matrices, feature importance, PR curves
├── reports/                         ← exported CSV fraud alert reports
├── main.py                          ← single entry point for everything
└── requirements.txt
```

---

## Quickstart

### 1. Get the dataset

Download `creditcard.csv` from Kaggle and place it in the `dataset/` folder:
[https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 2. Install dependencies

```powershell
pip install -r requirements.txt

# PyTorch (CPU — smaller and works for most machines)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run the pipeline

```powershell
# Full pipeline — classical models only (fast, ~2–5 minutes)
python main.py

# Full pipeline — classical + all 5 deep learning models
python main.py --deep

# Single transaction demo
python main.py --demo
```

---

## All Command-Line Options

| Flag | What it does |
|---|---|
| *(none)* | Full pipeline: train → predict → report (classical only) |
| `--deep` | Also train all 5 deep learning models alongside classical ones |
| `--train-only` | Only train models, skip predictions and reports |
| `--predict-only` | Only run predictions (model must already be trained) |
| `--report-only` | Only generate reports (predictions must already be logged) |
| `--demo` | Run one sample transaction through the system and print the decision |
| `--threshold 0.4` | Set the fraud probability threshold (default: 0.4) |
| `--sample-size 1000` | Number of transactions to predict on (default: 1000) |

---

## Models

### Classical Models (always trained)

| Model | Notes |
|---|---|
| Random Forest | 200 trees, balanced class weights |
| XGBoost | 200 estimators, scale_pos_weight=10 for fraud imbalance |
| Logistic Regression | L2 regularised, balanced class weights, scaled input |
| Isolation Forest | Unsupervised anomaly detection, contamination=0.01 |
| Voting Ensemble (RF + XGB) | Soft voting over Random Forest and XGBoost |

### Deep Learning Models (enabled with `--deep`)

| Model | Core idea |
|---|---|
| **FT-Transformer** | Every feature becomes an embedding token; Transformer attends over all features; CLS token gives the final prediction |
| **TabTransformer** | Only the first N features are tokenised and processed by a Transformer; numerical features are concatenated before the MLP head |
| **TabNet** | Sequential soft attention steps select which features to look at; interpretable feature importance masks are a natural by-product |
| **ResNet-MLP** | Plain MLP with residual skip connections; each block is BN → Linear → ReLU → Dropout → Linear → skip; often the strongest deep baseline |
| **NODE** | Stacked layers of differentiable oblivious decision trees; bridges XGBoost-style tree inductive bias with end-to-end gradient training |

All deep models share the same training loop in `deep_models/base_trainer.py`:
- AdamW optimiser with cosine annealing learning rate schedule
- BCEWithLogitsLoss with positive class up-weighting for imbalanced fraud data
- Early stopping that restores the best validation weights automatically
- Gradient clipping at 1.0 to prevent exploding gradients

The model with the highest validation AUC across all trained models (classical + deep) is saved as `best_model` and used for all subsequent predictions.

---

## How Predictions Work

`predict.py` loads `best_model.pkl` (classical) or `best_model.pt` (deep) automatically. Deep models are wrapped in `DeepModelWrapper` which exposes a standard `.predict_proba()` interface so the rest of the pipeline doesn't need to know about PyTorch.

Every transaction is assigned a risk level:

| Risk Level | Condition |
|---|---|
| `HIGH` | `fraud_prob >= threshold` |
| `MEDIUM` | `fraud_prob >= threshold × 0.6` |
| `LOW` | everything else |

---

## Database

All model runs and predictions are logged to a local SQLite database via `modules/db_setup.py`. Tables are created automatically on first run. This lets `report.py` query prediction history, flag high-risk transactions, and export fraud alert CSVs.

---

## Outputs After Running

```
plots/
  cm_Random_Forest.png                 ← confusion matrix per model
  cm_XGBoost.png
  cm_Logistic_Regression.png
  cm_Isolation_Forest.png
  cm_Voting_Ensemble_RFXGB.png
  feature_importance_Random_Forest.png ← top 15 features
  feature_importance_XGBoost.png
  precision_recall_<best_model>.png    ← PR curve for best classical model

reports/
  fraud_alerts_<date>.csv              ← exported high-risk transactions
```

---

## Requirements

```
python        >= 3.10
numpy         >= 1.24
pandas        >= 2.0
scikit-learn  >= 1.3
xgboost       >= 2.0
matplotlib    >= 3.7
torch         >= 2.0      (only needed for --deep)
```

---

## Common Errors and Fixes

**`ModuleNotFoundError: No module named 'torch'`**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**`cannot import name 'get_features_and_labels'`**
Replace `modules/data_loader.py` with the updated version from this repo which includes that function.

**PowerShell execution policy error when activating venv**
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned


**`FileNotFoundError: dataset/creditcard.csv`**
Download the dataset from Kaggle (link above) and place it in the `dataset/` folder.
