"""
report.py
---------
Pulls data from the database and generates human-readable fraud reports.
These are plain text / CSV reports — no UI, no web server, just clean output
you can read in the terminal or save to a file.

Reports we generate:
  1. Daily Fraud Summary       — how many frauds today, total amounts, risk breakdown
  2. Model Performance Report  — compare all trained models side by side
  3. High-Risk Alert Report    — all CRITICAL and HIGH risk transactions
  4. Prediction Accuracy Report — how accurate have our predictions been
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta

from modules.db_setup import get_connection

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'reports')


def ensure_reports_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def fraud_summary_report(days_back=1):
    """
    Generates a summary of fraud alerts from the last N days.
    Perfect for a daily operations digest.
    """
    conn = get_connection()
    since = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d %H:%M:%S')

    # Total alerts
    total = conn.execute(
        "SELECT COUNT(*) FROM fraud_alerts WHERE flagged_at >= ?", (since,)
    ).fetchone()[0]

    # By risk level
    risk_breakdown = conn.execute("""
        SELECT risk_level, COUNT(*) as count, SUM(amount) as total_amount,
               AVG(confidence) as avg_confidence
        FROM fraud_alerts
        WHERE flagged_at >= ?
        GROUP BY risk_level
        ORDER BY CASE risk_level
            WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2
            WHEN 'MEDIUM' THEN 3 WHEN 'LOW' THEN 4 END
    """, (since,)).fetchall()

    # Highest confidence fraud
    top_alerts = conn.execute("""
        SELECT transaction_ref, amount, confidence, risk_level, flagged_at
        FROM fraud_alerts
        WHERE flagged_at >= ?
        ORDER BY confidence DESC
        LIMIT 10
    """, (since,)).fetchall()

    conn.close()

    # Build the report text
    lines = []
    lines.append("=" * 60)
    lines.append(f"  FRAUD DETECTION — DAILY SUMMARY REPORT")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Period    : Last {days_back} day(s)")
    lines.append("=" * 60)
    lines.append(f"\n  Total Fraud Alerts : {total}")
    lines.append("\n  Risk Level Breakdown:")
    lines.append(f"  {'Risk':<12} {'Count':>8} {'Total Amount':>15} {'Avg Confidence':>16}")
    lines.append(f"  {'-'*55}")

    for row in risk_breakdown:
        amt_str = f"€{row['total_amount']:,.2f}" if row['total_amount'] else "€0.00"
        lines.append(
            f"  {row['risk_level']:<12} {row['count']:>8} {amt_str:>15} {row['avg_confidence']:>15.4f}"
        )

    lines.append("\n  Top 10 Highest Confidence Fraud Alerts:")
    lines.append(f"  {'Ref':<15} {'Amount':>10} {'Confidence':>12} {'Risk':<12} {'Flagged At'}")
    lines.append(f"  {'-'*65}")

    for row in top_alerts:
        lines.append(
            f"  {row['transaction_ref']:<15} €{row['amount']:>9,.2f} "
            f"{row['confidence']:>11.4f} {row['risk_level']:<12} {row['flagged_at']}"
        )

    report_text = "\n".join(lines)
    print(report_text)

    # Save to file
    ensure_reports_dir()
    filename = f"fraud_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[REPORT] Saved → {path}")

    return report_text


def model_performance_report():
    """
    Shows all trained model runs side by side so you can easily compare.
    Most recent run per model is shown.
    """
    conn = get_connection()

    # Get latest run for each model
    rows = conn.execute("""
        SELECT model_name, accuracy, precision_score, recall_score,
               f1_score, roc_auc, avg_precision, train_samples,
               test_samples, fraud_in_test, trained_at
        FROM model_runs
        WHERE id IN (
            SELECT MAX(id) FROM model_runs GROUP BY model_name
        )
        ORDER BY avg_precision DESC
    """).fetchall()

    conn.close()

    lines = []
    lines.append("=" * 75)
    lines.append("  MODEL PERFORMANCE COMPARISON REPORT")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 75)
    lines.append(
        f"\n  {'Model':<30} {'Accuracy':>9} {'Precision':>10} "
        f"{'Recall':>8} {'F1':>8} {'AUC-PR★':>9}"
    )
    lines.append(f"  {'-'*75}")

    best_model = None
    best_score = -1

    for row in rows:
        star = ""
        if row['avg_precision'] and row['avg_precision'] > best_score:
            best_score = row['avg_precision']
            best_model = row['model_name']

        lines.append(
            f"  {row['model_name']:<30} "
            f"{row['accuracy']:>9.4f} "
            f"{(row['precision_score'] or 0):>10.4f} "
            f"{(row['recall_score'] or 0):>8.4f} "
            f"{(row['f1_score'] or 0):>8.4f} "
            f"{(row['avg_precision'] or 0):>9.4f}"
        )

    lines.append(f"\n  ★ Best Model (by AUC-PR): {best_model}  ({best_score:.4f})")
    lines.append("\n  Note: AUC-PR is the primary metric for imbalanced fraud detection.")
    lines.append("        Accuracy alone is misleading — a model predicting 'legit'")
    lines.append("        every time would score 99.8% accuracy but catch zero fraud.")

    report_text = "\n".join(lines)
    print(report_text)

    # Also export as CSV for further analysis
    ensure_reports_dir()
    df = pd.DataFrame([dict(row) for row in rows])
    csv_path = os.path.join(REPORTS_DIR, 'model_comparison.csv')
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(REPORTS_DIR, 'model_performance.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n[REPORT] Saved → {txt_path}")
    print(f"[REPORT] CSV   → {csv_path}")

    return report_text


def high_risk_alert_report(min_risk='HIGH', limit=50):
    """
    Lists all unreviewed HIGH and CRITICAL fraud alerts.
    In a real system, this would be sent to a fraud analyst's inbox.
    """
    conn = get_connection()

    risk_levels = ['CRITICAL', 'HIGH'] if min_risk == 'HIGH' else ['CRITICAL', 'HIGH', 'MEDIUM']
    placeholders = ','.join('?' * len(risk_levels))

    rows = conn.execute(f"""
        SELECT transaction_ref, amount, confidence, risk_level,
               model_used, flagged_at, reviewed
        FROM fraud_alerts
        WHERE risk_level IN ({placeholders}) AND reviewed = 0
        ORDER BY confidence DESC
        LIMIT ?
    """, (*risk_levels, limit)).fetchall()

    conn.close()

    lines = []
    lines.append("=" * 70)
    lines.append(f"  HIGH-RISK FRAUD ALERT REPORT  (Unreviewed Only)")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"  Minimum Risk Level : {min_risk}")
    lines.append("=" * 70)
    lines.append(f"\n  Total Unreviewed Alerts : {len(rows)}")
    lines.append(
        f"\n  {'#':<5} {'Ref':<15} {'Amount':>10} {'Confidence':>12} "
        f"{'Risk':<12} {'Model':<25} {'Flagged At'}"
    )
    lines.append(f"  {'-'*90}")

    for i, row in enumerate(rows, 1):
        lines.append(
            f"  {i:<5} {row['transaction_ref']:<15} €{row['amount']:>9,.2f} "
            f"{row['confidence']:>11.4f} {row['risk_level']:<12} "
            f"{row['model_used']:<25} {row['flagged_at']}"
        )

    if not rows:
        lines.append("  No unreviewed high-risk alerts found. You're all clear!")

    report_text = "\n".join(lines)
    print(report_text)

    ensure_reports_dir()
    path = os.path.join(REPORTS_DIR, f'high_risk_alerts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[REPORT] Saved → {path}")

    return report_text


def prediction_accuracy_report():
    """
    Shows how accurate our model predictions have been based on
    transactions where we know the true label.
    """
    conn = get_connection()

    stats = conn.execute("""
        SELECT
            model_name,
            COUNT(*) as total_predictions,
            SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_predictions,
            SUM(CASE WHEN prediction = 1 AND correct = 1 THEN 1 ELSE 0 END) as true_positives,
            SUM(CASE WHEN prediction = 1 AND correct = 0 THEN 1 ELSE 0 END) as false_positives,
            SUM(CASE WHEN prediction = 0 AND correct = 0 THEN 1 ELSE 0 END) as true_negatives,
            SUM(CASE WHEN prediction = 0 AND correct = 1 THEN 1 ELSE 0 END) as false_negatives,
            AVG(confidence) as avg_confidence
        FROM prediction_logs
        WHERE correct != -1
        GROUP BY model_name
        ORDER BY correct_predictions DESC
    """).fetchall()

    conn.close()

    lines = []
    lines.append("=" * 70)
    lines.append("  PREDICTION ACCURACY REPORT")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    for row in stats:
        total = row['total_predictions']
        acc = row['correct_predictions'] / total if total > 0 else 0
        tp = row['true_positives']
        fp = row['false_positives']
        tn = row['true_negatives']
        fn = row['false_negatives']

        lines.append(f"\n  Model : {row['model_name']}")
        lines.append(f"  {'-'*45}")
        lines.append(f"  Total Predictions  : {total:,}")
        lines.append(f"  Overall Accuracy   : {acc*100:.2f}%")
        lines.append(f"  True Positives     : {tp:,}   (caught fraud correctly)")
        lines.append(f"  False Positives    : {fp:,}   (flagged legit as fraud)")
        lines.append(f"  True Negatives     : {tn:,}   (correctly cleared legit)")
        lines.append(f"  False Negatives    : {fn:,}   (missed fraud — most costly!)")
        lines.append(f"  Avg Confidence     : {row['avg_confidence']:.4f}")

    report_text = "\n".join(lines)
    print(report_text)

    ensure_reports_dir()
    path = os.path.join(REPORTS_DIR, 'prediction_accuracy.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n[REPORT] Saved → {path}")

    return report_text


def export_fraud_alerts_csv():
    """Exports all fraud alerts to a CSV file for external analysis."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM fraud_alerts ORDER BY flagged_at DESC", conn)
    conn.close()

    ensure_reports_dir()
    path = os.path.join(REPORTS_DIR, f'fraud_alerts_export_{datetime.now().strftime("%Y%m%d")}.csv')
    df.to_csv(path, index=False)
    print(f"[EXPORT] {len(df):,} fraud alerts exported → {path}")
    return path


if __name__ == "__main__":
    print("\nRunning all reports...\n")
    fraud_summary_report(days_back=30)
    model_performance_report()
    high_risk_alert_report()
    prediction_accuracy_report()
    export_fraud_alerts_csv()
