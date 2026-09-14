"""
train.py
--------
Trains and compares multiple classification models for internship
performance prediction:

    - Random Forest (primary model, hyperparameter-tuned via GridSearchCV)
    - Decision Tree
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Naive Bayes (GaussianNB)

The best-performing model (by weighted F1-score on the held-out test set)
is saved to disk with joblib, along with the fitted StandardScaler and
metadata describing feature order / class labels, so the API layer can
load everything needed for inference without re-running training code.

Run directly:
    python src/train.py
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (
    full_preprocessing_pipeline,
    FEATURE_COLUMNS,
    CLASS_ORDER,
    IDX_TO_CLASS,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "internship_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

RANDOM_STATE = 42


def get_model_zoo():
    """
    Returns a dict of {model_name: (estimator, param_grid, uses_scaled_data)}.
    uses_scaled_data indicates whether that model should be trained on the
    standardized features (True) or the raw features (False, tree models
    are scale-invariant so raw is fine and keeps things interpretable).
    """
    zoo = {
        "RandomForest": (
            RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 8, 12, 16],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
            False,
        ),
        "DecisionTree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "max_depth": [None, 5, 8, 12],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"],
            },
            False,
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["lbfgs"],
            },
            True,
        ),
        "SVM": (
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {
                "C": [0.1, 1, 10],
                "kernel": ["rbf"],
                "gamma": ["scale", "auto"],
            },
            True,
        ),
        "NaiveBayes": (
            GaussianNB(),
            {
                "var_smoothing": [1e-9, 1e-8, 1e-7],
            },
            True,
        ),
    }
    return zoo


def train_and_compare_models(csv_path: str = DATA_PATH, models_dir: str = MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)

    (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler), df_clean = \
        full_preprocessing_pipeline(csv_path, scaler_path=os.path.join(models_dir, "scaler.joblib"))

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Class balance (train):\n{y_train.value_counts().sort_index()}")

    zoo = get_model_zoo()
    results = {}
    fitted_models = {}

    for name, (estimator, param_grid, use_scaled) in zoo.items():
        print("\n" + "=" * 60)
        print(f"Training & tuning: {name}")
        print("=" * 60)

        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test

        # Estimate grid size to decide between exhaustive GridSearchCV
        # (small grids) and RandomizedSearchCV (larger grids, capped
        # number of iterations) to keep training time reasonable.
        grid_size = 1
        for v in param_grid.values():
            grid_size *= len(v)

        start = time.time()
        if grid_size > 24:
            grid = RandomizedSearchCV(
                estimator,
                param_grid,
                n_iter=20,
                cv=3,
                scoring="f1_weighted",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=0,
            )
        else:
            grid = GridSearchCV(
                estimator,
                param_grid,
                cv=3,
                scoring="f1_weighted",
                n_jobs=-1,
                verbose=0,
            )
        grid.fit(Xtr, y_train)
        elapsed = time.time() - start

        best_model = grid.best_estimator_
        y_pred = best_model.predict(Xte)
        test_f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"Best params: {grid.best_params_}")
        print(f"CV best F1 (weighted): {grid.best_score_:.4f}")
        print(f"Test F1 (weighted): {test_f1:.4f}")
        print(f"Time: {elapsed:.1f}s")

        results[name] = {
            "best_params": grid.best_params_,
            "cv_f1_weighted": grid.best_score_,
            "test_f1_weighted": test_f1,
            "uses_scaled_data": use_scaled,
            "train_time_sec": elapsed,
        }
        fitted_models[name] = best_model

    # ---- Identify best model by test F1 ----
    best_name = max(results, key=lambda k: results[k]["test_f1_weighted"])
    best_model = fitted_models[best_name]
    best_uses_scaled = results[best_name]["uses_scaled_data"]

    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY (sorted by test F1-weighted)")
    print("=" * 60)
    summary_df = pd.DataFrame(results).T.sort_values("test_f1_weighted", ascending=False)
    print(summary_df[["cv_f1_weighted", "test_f1_weighted", "train_time_sec"]])
    print(f"\n>>> BEST MODEL: {best_name} (test F1-weighted = {results[best_name]['test_f1_weighted']:.4f})")

    # ---- Save best model + metadata ----
    model_path = os.path.join(models_dir, "best_model.joblib")
    joblib.dump(best_model, model_path)

    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "class_order": CLASS_ORDER,
        "idx_to_class": IDX_TO_CLASS,
        "uses_scaled_data": best_uses_scaled,
        "results_summary": results,
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Also persist ALL tuned models (useful for comparison / evaluate.py)
    for name, model in fitted_models.items():
        joblib.dump(model, os.path.join(models_dir, f"model_{name}.joblib"))

    print(f"\nBest model ('{best_name}') saved to: {model_path}")
    print(f"Metadata saved to: {os.path.join(models_dir, 'metadata.json')}")

    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "results": results,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


if __name__ == "__main__":
    train_and_compare_models()
