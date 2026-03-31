"""
db_setup.py
-----------
Handles all database creation and table setup for the fraud detection system.
Think of this as the 'foundation layer' — run it once before anything else.
"""

import sqlite3
import os

# Where we store our database file
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'fraud_detection.db')


def get_connection():
    """
    Returns a live connection to our SQLite database.
    We enable WAL mode so reads don't block writes — useful when logging predictions
    while training is still happening in the background.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row  # lets us access columns by name, not just index
    return conn


def create_tables():
    """
    Sets up all necessary tables if they don't already exist.
    We have four main tables:
      1. transactions     — raw transaction records from the dataset
      2. fraud_alerts     — flagged suspicious transactions with reasons
      3. model_runs       — a history of every training session
      4. prediction_logs  — every prediction made, with confidence scores
    """
    conn = get_connection()
    cursor = conn.cursor()

    # --- Table 1: Raw Transactions ---
    # Stores every transaction we load from the CSV dataset
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_ref TEXT UNIQUE,         -- unique ID we generate
            time_seconds    REAL,                -- seconds elapsed since first transaction
            amount          REAL,                -- transaction amount in euros
            true_label      INTEGER,             -- 0 = legit, 1 = fraud (ground truth)
            loaded_at       TEXT DEFAULT (datetime('now'))
        )
    """)

    # --- Table 2: Fraud Alerts ---
    # When our model predicts fraud, we log it here with details
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fraud_alerts (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_ref TEXT,
            predicted_label INTEGER,             -- what our model said
            confidence      REAL,                -- probability score (0.0 to 1.0)
            model_used      TEXT,                -- which model made this call
            risk_level      TEXT,                -- LOW / MEDIUM / HIGH / CRITICAL
            amount          REAL,
            flagged_at      TEXT DEFAULT (datetime('now')),
            reviewed        INTEGER DEFAULT 0,   -- 0 = not reviewed yet, 1 = reviewed
            FOREIGN KEY (transaction_ref) REFERENCES transactions(transaction_ref)
        )
    """)

    # --- Table 3: Model Training Runs ---
    # Every time we train a model, we record how it performed
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT,
            accuracy        REAL,
            precision_score REAL,
            recall_score    REAL,
            f1_score        REAL,
            roc_auc         REAL,
            avg_precision   REAL,               -- AUC-PR, better metric for imbalanced data
            train_samples   INTEGER,
            test_samples    INTEGER,
            fraud_in_test   INTEGER,
            trained_at      TEXT DEFAULT (datetime('now')),
            model_path      TEXT                -- where the .pkl file is saved
        )
    """)

    # --- Table 4: Prediction Logs ---
    # A full audit trail of every single prediction we make
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_ref TEXT,
            model_name      TEXT,
            prediction      INTEGER,            -- 0 or 1
            confidence      REAL,
            correct         INTEGER,            -- 1 if prediction matched true label, else 0
            predicted_at    TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] All tables are ready.")


def insert_transaction(conn, transaction_ref, time_seconds, amount, true_label):
    """
    Inserts a single transaction record. Uses INSERT OR IGNORE so duplicate
    refs don't cause crashes when we re-run the loader.
    """
    conn.execute("""
        INSERT OR IGNORE INTO transactions (transaction_ref, time_seconds, amount, true_label)
        VALUES (?, ?, ?, ?)
    """, (transaction_ref, time_seconds, amount, true_label))


def insert_fraud_alert(conn, transaction_ref, predicted_label, confidence, model_used, amount):
    """
    Logs a fraud alert. Automatically determines the risk level based on confidence:
      - CRITICAL  : >= 0.90
      - HIGH      : >= 0.75
      - MEDIUM    : >= 0.60
      - LOW       : anything below that
    """
    if confidence >= 0.90:
        risk = "CRITICAL"
    elif confidence >= 0.75:
        risk = "HIGH"
    elif confidence >= 0.60:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    conn.execute("""
        INSERT INTO fraud_alerts (transaction_ref, predicted_label, confidence, model_used, risk_level, amount)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (transaction_ref, predicted_label, confidence, model_used, risk, amount))


def log_model_run(conn, model_name, metrics: dict, train_samples, test_samples, fraud_in_test, model_path):
    """
    Saves a training run record so we can compare models over time.
    `metrics` should be a dict with keys: accuracy, precision, recall, f1, roc_auc, avg_precision
    """
    conn.execute("""
        INSERT INTO model_runs
            (model_name, accuracy, precision_score, recall_score, f1_score, roc_auc,
             avg_precision, train_samples, test_samples, fraud_in_test, model_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        model_name,
        metrics.get('accuracy'),
        metrics.get('precision'),
        metrics.get('recall'),
        metrics.get('f1'),
        metrics.get('roc_auc'),
        metrics.get('avg_precision'),
        train_samples,
        test_samples,
        fraud_in_test,
        model_path
    ))


def log_prediction(conn, transaction_ref, model_name, prediction, confidence, correct):
    """Appends one row to the prediction audit log."""
    conn.execute("""
        INSERT INTO prediction_logs (transaction_ref, model_name, prediction, confidence, correct)
        VALUES (?, ?, ?, ?, ?)
    """, (transaction_ref, model_name, prediction, confidence, correct))


if __name__ == "__main__":
    create_tables()
