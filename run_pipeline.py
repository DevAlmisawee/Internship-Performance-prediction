"""
run_pipeline.py
----------------
Convenience entry point that runs the full ML pipeline end to end:

    1. Verify the prepared dataset exists (data/internship_data.csv)
    2. Run EDA (prints summary stats, saves plots to logs/)
    3. Train & tune all models, save the best one to models/
    4. Run full evaluation (metrics, confusion matrices, ROC-AUC,
       feature importance) and save plots/summary to logs/
    5. Run a quick smoke-test prediction to confirm everything works

This project runs on real data — there is no synthetic fallback. If
data/internship_data.csv is missing, run data/prepare_dataset.py first
against your raw CSV to derive the categorical target and clean the data.

Usage:
    python run_pipeline.py
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(PROJECT_ROOT, "data", "internship_data.csv")


def main():
    parser = argparse.ArgumentParser(description="Run the full internship performance prediction pipeline.")
    args = parser.parse_args()

    # ---- Step 1: Verify the prepared dataset exists ----
    print("\n" + "#" * 70)
    print("STEP 1/5: Checking for prepared dataset")
    print("#" * 70)
    if not os.path.exists(DATA_CSV):
        print(f"ERROR: {DATA_CSV} not found.")
        print("This project runs on real data. Prepare it first with:")
        print("  python data/prepare_dataset.py --input /path/to/your_raw_data.csv --output data/internship_data.csv")
        sys.exit(1)
    print(f"Found dataset at {DATA_CSV}")

    # ---- Step 2: EDA ----
    print("\n" + "#" * 70)
    print("STEP 2/5: Exploratory Data Analysis")
    print("#" * 70)
    from src.eda import run_eda
    run_eda()

    # ---- Step 3: Train & tune models ----
    print("\n" + "#" * 70)
    print("STEP 3/5: Training & hyperparameter tuning (5 models)")
    print("#" * 70)
    from src.train import train_and_compare_models
    train_and_compare_models()

    # ---- Step 4: Evaluation ----
    print("\n" + "#" * 70)
    print("STEP 4/5: Full model evaluation")
    print("#" * 70)
    from src.evaluate import run_full_evaluation
    run_full_evaluation()

    # ---- Step 5: Smoke-test prediction ----
    print("\n" + "#" * 70)
    print("STEP 5/5: Smoke-test prediction")
    print("#" * 70)
    from src.predict import predict_performance
    result = predict_performance(
        gpa=3.6, course_scores=85, aptitude_score=78, attendance=92,
        supervisor_evaluation=8.5, report_quality=8.0, activity_log_frequency=15,
    )
    print(f"Sample prediction: {result}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Next steps:")
    print("  - Inspect plots/metrics in logs/")
    print("  - Start the REST API:  python api/app.py")
    print("  - Test it:             curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{...}'")


if __name__ == "__main__":
    main()
