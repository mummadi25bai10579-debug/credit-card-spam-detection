"""
main.py
-------
The single entry point for the entire Credit Card Fraud Detection system.
Run this file to go from raw dataset → trained models → predictions → reports.

Usage:
    python main.py                  # classical pipeline only
    python main.py --deep           # classical + all 5 deep models
    python main.py --train-only     # only train models
    python main.py --predict-only   # only run predictions (model must exist)
    python main.py --report-only    # only generate reports (data must exist)
    python main.py --demo           # quick demo with one transaction
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))

from modules.db_setup import create_tables


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║     CREDIT CARD FRAUD DETECTION SYSTEM                   ║
║     Classical ML  +  FT-Transformer  +  TabTransformer   ║
║     +  TabNet  +  ResNet-MLP  +  NODE                    ║
╚══════════════════════════════════════════════════════════╝
    """)


def run_training(include_deep: bool = False):
    print("\n[STEP 1/3] Training Models...")
    print("-" * 55)
    from modules.train_model import train_all
    best_name, best_score = train_all(include_deep=include_deep)
    return best_name, best_score


def run_predictions(sample_size=1000, threshold=0.4):
    print("\n[STEP 2/3] Running Predictions...")
    print("-" * 55)

    from modules.data_loader import load_raw_data
    from modules.predict     import predict_batch, load_model

    df    = load_raw_data().sample(sample_size, random_state=7).reset_index(drop=True)
    model = load_model('best_model')

    results = predict_batch(df, model=model, threshold=threshold, log_to_db=True)

    print(f"\n  Prediction breakdown:")
    print(results['risk_level'].value_counts().to_string())
    return results


def run_reports():
    print("\n[STEP 3/3] Generating Reports...")
    print("-" * 55)

    from modules.report import (
        fraud_summary_report,
        model_performance_report,
        high_risk_alert_report,
        prediction_accuracy_report,
        export_fraud_alerts_csv,
    )

    fraud_summary_report(days_back=7)
    print()
    model_performance_report()
    print()
    high_risk_alert_report()
    print()
    prediction_accuracy_report()
    export_fraud_alerts_csv()


def run_demo():
    print("\n[DEMO] Simulating a single real-time transaction check...")
    print("-" * 55)

    from modules.predict     import predict_single
    from modules.data_loader import load_raw_data

    df            = load_raw_data()
    fraud_samples = df[df['Class'] == 1]
    sample        = (
        fraud_samples.sample(n=1, random_state=42)
        if len(fraud_samples) > 0
        else df.sample(n=1, random_state=42)
    )
    transaction = sample.iloc[0].to_dict()
    result      = predict_single(transaction, threshold=0.4)

    print(f"\n  → {'🚨 BLOCK TRANSACTION' if result['is_fraud'] else '✅ APPROVE TRANSACTION'}")


def main():
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection System')
    parser.add_argument('--deep',         action='store_true', help='Also train deep learning models')
    parser.add_argument('--train-only',   action='store_true', help='Only run model training')
    parser.add_argument('--predict-only', action='store_true', help='Only run predictions')
    parser.add_argument('--report-only',  action='store_true', help='Only generate reports')
    parser.add_argument('--demo',         action='store_true', help='Run single transaction demo')
    parser.add_argument('--threshold',    type=float, default=0.4, help='Fraud detection threshold (default: 0.4)')
    parser.add_argument('--sample-size',  type=int,   default=1000, help='Transactions to predict on (default: 1000)')
    args = parser.parse_args()

    print_banner()
    create_tables()

    start = time.time()

    if args.demo:
        run_demo()

    elif args.train_only:
        run_training(include_deep=args.deep)

    elif args.predict_only:
        run_predictions(sample_size=args.sample_size, threshold=args.threshold)

    elif args.report_only:
        run_reports()

    else:
        run_training(include_deep=args.deep)
        run_predictions(sample_size=args.sample_size, threshold=args.threshold)
        run_reports()

    elapsed = time.time() - start
    print(f"\n{'='*55}")
    print(f"  ✓ Done in {elapsed:.1f} seconds")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
