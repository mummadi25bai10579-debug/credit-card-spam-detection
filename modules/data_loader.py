"""
modules/data_loader.py
-----------------------
Handles loading the raw creditcard.csv dataset and preparing
features + labels for model training and prediction.

Functions
---------
load_raw_data()              → pd.DataFrame   (raw CSV)
get_features_and_labels(df)  → (X, y, feature_names)
"""

import os
import numpy as np
import pandas as pd

DATASET_PATH = os.path.join("dataset", "creditcard.csv")

# These columns will never be used as features
_LABEL_COL   = "Class"
_DROP_COLS   = {"Class"}   # add more here if needed e.g. {"Class", "id"}


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw creditcard.csv from the dataset/ folder.
    Raises a clear error if the file isn't found.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'.\n"
            "Please download creditcard.csv from Kaggle and place it in the dataset/ folder.\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )

    df = pd.read_csv(DATASET_PATH)
    print(f"  Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def get_features_and_labels(df: pd.DataFrame):
    """
    Split a DataFrame into feature matrix X, label vector y,
    and the list of feature column names.

    Works whether or not the 'Class' column is present:
    - If present  → y = df['Class'],  X = everything else
    - If absent   → y = zeros array  (prediction mode, no ground truth)

    Returns
    -------
    X             : np.ndarray  shape (n_samples, n_features)
    y             : np.ndarray  shape (n_samples,)   int dtype
    feature_names : list[str]
    """
    # figure out which columns are features
    feature_cols = [c for c in df.columns if c not in _DROP_COLS]

    X = df[feature_cols].values.astype(np.float32)

    if _LABEL_COL in df.columns:
        y = df[_LABEL_COL].values.astype(int)
    else:
        # prediction mode — caller doesn't have ground truth
        y = np.zeros(len(df), dtype=int)

    return X, y, feature_cols
