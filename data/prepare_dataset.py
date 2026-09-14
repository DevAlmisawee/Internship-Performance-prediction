"""
prepare_dataset.py
--------------------
Prepares the real InternIQ dataset (internship_data.csv) for the modeling
pipeline, using the EXACT feature set and target thresholds the InternIQ
Express backend (`mlService.js` / `StudentPerformance.js`) was built
against. This makes the trained model a drop-in match for that backend's
contract — no field renaming or threshold mismatches.

Feature schema (9 features, all genuine measured data — no synthesis):

    GPA                      0.0 - 4.0
    Course_Scores            0 - 100
    Aptitude_Score           0 - 100
    Attendance                0 - 100 (%)
    Supervisor_Evaluation    0 - 10
    Report_Quality           0 - 10
    Activity_Log_Frequency   0 - 29 (count)
    Completion_Time          1 - 7  (lower = faster)
    Feedback_Rating          1 - 5

Target (5 classes, matches VALID_PREDICTIONS in mlService.js):

    >= 85   -> Excellent
    70 - 84 -> Good
    55 - 69 -> Average
    40 - 54 -> Fair
    <  40   -> Poor

What this script does
----------------------
1. Cleans the data (drops duplicate rows, median-imputes any missing values).
2. Derives the categorical `Performance` target from the continuous
   `Performance_Score` column using the thresholds above.
3. Keeps the extra real columns (Intern_ID, Performance_Score) in the
   output CSV for traceability — the modeling pipeline only reads the 9
   canonical FEATURE_COLUMNS plus Performance, so extra columns are simply
   ignored during training.

Run:
    python data/prepare_dataset.py \
        --input /mnt/user-data/uploads/internship_data.csv \
        --output data/internship_data.csv
"""

import argparse
import os
import pandas as pd

CANONICAL_FEATURES = [
    "GPA", "Course_Scores", "Aptitude_Score", "Attendance",
    "Supervisor_Evaluation", "Report_Quality", "Activity_Log_Frequency",
    "Completion_Time", "Feedback_Rating",
]


def derive_performance_label(score: float) -> str:
    """Map a 0-100 Performance_Score to a category using the backend's fixed thresholds."""
    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 55:
        return "Average"
    elif score >= 40:
        return "Fair"
    else:
        return "Poor"


def prepare_dataset(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    n_start = len(df)
    print(f"Loaded {n_start} records from {input_csv}")

    missing_cols = [c for c in CANONICAL_FEATURES + ["Performance_Score"] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input CSV is missing expected column(s): {missing_cols}")

    # ---- 1. Clean: drop duplicates, impute missing numeric values ----
    before = len(df)
    df = df.drop_duplicates()
    if len(df) != before:
        print(f"Dropped {before - len(df)} duplicate rows")

    numeric_cols = CANONICAL_FEATURES + ["Performance_Score"]
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {n_missing} missing '{col}' values with median={median_val:.2f}")

    # ---- 2. Derive categorical target from Performance_Score ----
    df["Performance"] = df["Performance_Score"].apply(derive_performance_label)
    print("\nDerived class distribution:")
    counts = df["Performance"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    print(pd.DataFrame({"count": counts, "pct": pct}))

    # ---- 3. Reorder columns: canonical features first, extras retained after ----
    extra_cols = [c for c in df.columns if c not in CANONICAL_FEATURES + ["Performance"]]
    df = df[CANONICAL_FEATURES + ["Performance"] + extra_cols]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nPrepared dataset saved to: {output_csv}")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the real InternIQ dataset for training.")
    parser.add_argument("--input", required=True, help="Path to the raw real CSV")
    parser.add_argument("--output", default=None, help="Path to save the prepared CSV")
    args = parser.parse_args()

    output = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), "internship_data.csv")
    prepare_dataset(args.input, output)
