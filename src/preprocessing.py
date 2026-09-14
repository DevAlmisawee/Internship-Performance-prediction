"""
preprocessing.py
-----------------
Data loading, cleaning, encoding, and scaling utilities for the
Internship Performance Prediction pipeline.

Design notes
------------
- Tree-based models (Random Forest, Decision Tree) do not require feature
  scaling, but distance/gradient-based models (Logistic Regression, SVM,
  Naive Bayes assumes gaussian likelihood too) benefit from it. We
  therefore ALWAYS produce a scaled version of the features and let each
  model's training routine decide whether to use raw or scaled data.
- Missing values are imputed with the median (robust to outliers) rather
  than the mean.
- Duplicate rows are dropped.
- The target label is encoded with a fixed, explicit ordinal mapping
  (Poor < Fair < Average < Good < Excellent) rather than sklearn's
  LabelEncoder default alphabetical ordering, since the classes are
  ordinal in nature and consistent ordering matters for interpretability
  of metrics/plots.
- This schema (9 features, 5 classes) matches the InternIQ Express
  backend's `mlService.js` / `StudentPerformance.js` contract exactly, so
  the trained model is a drop-in prediction service for that backend.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

FEATURE_COLUMNS = [
    "GPA",
    "Course_Scores",
    "Aptitude_Score",
    "Attendance",
    "Supervisor_Evaluation",
    "Report_Quality",
    "Activity_Log_Frequency",
    "Completion_Time",
    "Feedback_Rating",
]

TARGET_COLUMN = "Performance"

# Explicit ordinal class ordering (index = encoded label)
CLASS_ORDER = ["Poor", "Fair", "Average", "Good", "Excellent"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV dataset from disk."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            f"Run data/generate_dataset.py first, or supply your own CSV "
            f"with columns: {FEATURE_COLUMNS + [TARGET_COLUMN]}"
        )
    return pd.read_csv(csv_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and duplicate records.

    - Drops exact duplicate rows.
    - Imputes missing numeric feature values with the column median.
    - Drops any row missing the target label (can't train/evaluate on it).
    """
    df = df.copy()

    # Drop rows with missing target - cannot be used for supervised learning
    df = df.dropna(subset=[TARGET_COLUMN])

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"[clean_data] Dropped {before - after} duplicate rows.")

    # Impute missing numeric features with median
    for col in FEATURE_COLUMNS:
        if df[col].isna().sum() > 0:
            median_val = df[col].median()
            n_missing = df[col].isna().sum()
            df[col] = df[col].fillna(median_val)
            print(f"[clean_data] Imputed {n_missing} missing values in '{col}' with median={median_val:.2f}")

    return df.reset_index(drop=True)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the categorical target using the fixed ordinal mapping."""
    df = df.copy()
    unknown = set(df[TARGET_COLUMN].unique()) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"Unexpected target classes found: {unknown}. Expected one of {CLASS_ORDER}")
    df["Performance_Encoded"] = df[TARGET_COLUMN].map(CLASS_TO_IDX)
    return df


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_path: str = None,
):
    """
    Split into train/test sets (stratified on target) and fit a
    StandardScaler on the training set only (to avoid data leakage).

    Returns
    -------
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler
    """
    X = df[FEATURE_COLUMNS]
    y = df["Performance_Encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=FEATURE_COLUMNS, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=FEATURE_COLUMNS, index=X_test.index
    )

    if scaler_path:
        joblib.dump(scaler, scaler_path)
        print(f"[split_and_scale] Scaler saved to {scaler_path}")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def full_preprocessing_pipeline(csv_path: str, scaler_path: str = None):
    """
    Convenience function that runs the full preprocessing pipeline end to end:
    load -> clean -> encode -> split -> scale.
    """
    df = load_data(csv_path)
    df = clean_data(df)
    df = encode_target(df)
    return split_and_scale(df, scaler_path=scaler_path), df
