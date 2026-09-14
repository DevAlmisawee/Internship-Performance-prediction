"""
evaluate.py
-----------
Comprehensive evaluation of the trained models:
    - Accuracy, Precision, Recall, F1-score (per-class + weighted/macro)
    - Confusion Matrix (saved as PNG + printed)
    - Full classification report
    - ROC-AUC (One-vs-Rest, multiclass)
    - Feature importance ranking (from the Random Forest model specifically,
      since it natively exposes impurity-based importances; for the overall
      best model we also report permutation importance if it lacks a
      native attribute)

Run directly (assumes train.py has already been run):
    python src/evaluate.py
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import full_preprocessing_pipeline, FEATURE_COLUMNS, CLASS_ORDER

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "internship_data.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")


def load_test_split():
    """Reproduce the exact same train/test split used during training."""
    (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler), df_clean = \
        full_preprocessing_pipeline(DATA_PATH)
    return X_test, y_test, X_test_scaled


def evaluate_model(model, X_test, y_test, model_name: str, output_dir: str = LOGS_DIR):
    """Compute and print/save a full evaluation report for a single model."""
    os.makedirs(output_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    print("\n" + "=" * 60)
    print(f"EVALUATION REPORT: {model_name}")
    print("=" * 60)

    acc = accuracy_score(y_test, y_pred)
    precision_w = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_w = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"Accuracy:            {acc:.4f}")
    print(f"Precision (weighted): {precision_w:.4f}")
    print(f"Recall (weighted):    {recall_w:.4f}")
    print(f"F1-score (weighted):  {f1_w:.4f}")

    print("\nFull Classification Report:")
    report = classification_report(y_test, y_pred, target_names=CLASS_ORDER, zero_division=0)
    print(report)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_yticklabels(CLASS_ORDER)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix: {model_name}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"Confusion matrix plot saved to: {cm_path}")

    # ---- ROC-AUC (multiclass, One-vs-Rest) ----
    roc_auc = None
    if y_proba is not None:
        y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_ORDER))))
        try:
            roc_auc = roc_auc_score(y_test_bin, y_proba, average="weighted", multi_class="ovr")
            print(f"\nROC-AUC (weighted, One-vs-Rest): {roc_auc:.4f}")

            fig, ax = plt.subplots(figsize=(7, 6))
            for i, cls in enumerate(CLASS_ORDER):
                RocCurveDisplay.from_predictions(
                    y_test_bin[:, i], y_proba[:, i], name=cls, ax=ax
                )
            ax.set_title(f"ROC Curves (One-vs-Rest): {model_name}")
            plt.tight_layout()
            roc_path = os.path.join(output_dir, f"roc_curves_{model_name}.png")
            plt.savefig(roc_path, dpi=120)
            plt.close(fig)
            print(f"ROC curve plot saved to: {roc_path}")
        except Exception as e:
            print(f"Could not compute ROC-AUC: {e}")

    return {
        "model_name": model_name,
        "accuracy": acc,
        "precision_weighted": precision_w,
        "recall_weighted": recall_w,
        "f1_weighted": f1_w,
        "roc_auc_weighted": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def plot_feature_importance(model, feature_names, model_name: str, X_test=None, y_test=None,
                             output_dir: str = LOGS_DIR):
    """
    Plot feature importance. Uses native `feature_importances_` if available
    (tree-based models). Otherwise falls back to permutation importance,
    which works for any fitted estimator (e.g. Logistic Regression, SVM).
    """
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        method = "Impurity-based (native)"
    elif X_test is not None and y_test is not None:
        result = permutation_importance(model, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1)
        importances = result.importances_mean
        method = "Permutation importance"
    else:
        print(f"[plot_feature_importance] Cannot compute importance for {model_name} (no native attribute, no test data provided).")
        return None

    order = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in order]
    sorted_importances = importances[order]

    print(f"\nFeature Importance ({method}) - {model_name}:")
    for f, imp in zip(sorted_features, sorted_importances):
        print(f"  {f:30s} {imp:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_features[::-1], sorted_importances[::-1], color="#4c72b0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance ({method}): {model_name}")
    plt.tight_layout()
    path = os.path.join(output_dir, f"feature_importance_{model_name}.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Feature importance plot saved to: {path}")

    return dict(zip(sorted_features, sorted_importances.tolist()))


def run_full_evaluation():
    X_test, y_test, X_test_scaled = load_test_split()

    with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    all_reports = {}

    # Evaluate every trained model that was persisted by train.py
    for fname in os.listdir(MODELS_DIR):
        if fname.startswith("model_") and fname.endswith(".joblib"):
            model_name = fname.replace("model_", "").replace(".joblib", "")
            model = joblib.load(os.path.join(MODELS_DIR, fname))

            # Determine whether this model expects scaled or raw features
            uses_scaled = metadata["results_summary"].get(model_name, {}).get("uses_scaled_data", False)
            X_eval = X_test_scaled if uses_scaled else X_test

            report = evaluate_model(model, X_eval, y_test, model_name)
            all_reports[model_name] = report

    # Feature importance specifically for Random Forest (native, most interpretable)
    rf_path = os.path.join(MODELS_DIR, "model_RandomForest.joblib")
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        plot_feature_importance(rf_model, FEATURE_COLUMNS, "RandomForest")

    # Feature importance for the actual BEST model (may need permutation importance)
    best_name = metadata["best_model_name"]
    best_model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    uses_scaled = metadata["uses_scaled_data"]
    X_eval = X_test_scaled if uses_scaled else X_test
    plot_feature_importance(best_model, FEATURE_COLUMNS, f"BEST_{best_name}", X_eval, y_test)

    # Save combined evaluation summary (all models, for offline analysis)
    with open(os.path.join(LOGS_DIR, "evaluation_summary.json"), "w") as f:
        json.dump(all_reports, f, indent=2, default=str)

    # Also persist just the winning model's held-out test report alongside the
    # model artifacts in models/, so the Flask API can serve it directly
    # (api/app.py has no access to logs/, only models/). This is what fixes
    # the admin dashboard showing in-sample "training-evaluation" accuracy
    # (~90%, computed against all imported rows including training data) as
    # if it were the model's real accuracy -- the frontend can now also pull
    # this file's genuine held-out test figures (~63%) via GET /test-evaluation.
    if best_name in all_reports:
        with open(os.path.join(MODELS_DIR, "test_evaluation.json"), "w") as f:
            json.dump(all_reports[best_name], f, indent=2, default=str)
        print(f"Held-out test evaluation for best model ('{best_name}') saved to: "
              f"{os.path.join(MODELS_DIR, 'test_evaluation.json')}")

    print("\n" + "=" * 60)
    print("Evaluation complete. Summary saved to logs/evaluation_summary.json")
    print("=" * 60)

    return all_reports


if __name__ == "__main__":
    run_full_evaluation()
