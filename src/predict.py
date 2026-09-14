"""
predict.py
----------
Loads the persisted best model (+ scaler + metadata) and exposes a single
clean prediction function that the API layer (and any other Python code)
can call directly.

Argument names match the InternIQ backend's ML payload EXACTLY (see
buildMLPayload() in mlService.js) — i.e. the dataset's own column names
(GPA, Course_Scores, ...) rather than camelCase — so the Flask API layer
can pass through `request.get_json()` with zero field renaming.

Usage:
    from src.predict import predict_performance

    result = predict_performance(
        GPA=3.5,
        Course_Scores=82,
        Aptitude_Score=75,
        Attendance=91,
        Supervisor_Evaluation=8.2,
        Report_Quality=7.9,
        Activity_Log_Frequency=14,
        Completion_Time=3,
        Feedback_Rating=4.1,
    )
    print(result)
    # {
    #   "predicted_class": "Good",
    #   "confidence": 0.71,
    #   "probabilities": {"Poor": 0.0, "Fair": 0.05, "Average": 0.10, "Good": 0.71, "Excellent": 0.14},
    #   "model_used": "RandomForest"
    # }
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

_model = None
_scaler = None
_metadata = None

# Valid ranges for each feature, matching the InternIQ backend's Mongoose
# schema / express-validator rules exactly (see StudentPerformance.js and
# performanceRoutes.js in the backend repo).
FEATURE_BOUNDS = {
    "GPA": (0.0, 4.0),
    "Course_Scores": (0.0, 100.0),
    "Aptitude_Score": (0.0, 100.0),
    "Attendance": (0.0, 100.0),
    "Supervisor_Evaluation": (0.0, 10.0),
    "Report_Quality": (0.0, 10.0),
    "Activity_Log_Frequency": (0, 29),
    "Completion_Time": (1, 7),
    "Feedback_Rating": (1, 5),
}


class ModelNotLoadedError(RuntimeError):
    pass


def _load_artifacts():
    """Lazily load model, scaler, and metadata into module-level caches."""
    global _model, _scaler, _metadata

    if _model is not None:
        return

    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")

    for p in (model_path, scaler_path, metadata_path):
        if not os.path.exists(p):
            raise ModelNotLoadedError(
                f"Required artifact missing: {p}. Run `python src/train.py` first."
            )

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    with open(metadata_path) as f:
        _metadata = json.load(f)


def _validate_inputs(values: dict):
    """Basic sanity-range validation for each feature. Raises ValueError on failure."""
    for key, (lo, hi) in FEATURE_BOUNDS.items():
        val = values.get(key)
        if val is None:
            raise ValueError(f"Missing required field: '{key}'")
        try:
            val = float(val)
        except (TypeError, ValueError):
            raise ValueError(f"Field '{key}' must be numeric, got: {val!r}")
        if not (lo <= val <= hi):
            raise ValueError(f"Field '{key}'={val} is outside the expected range [{lo}, {hi}]")


def predict_performance(
    GPA: float,
    Course_Scores: float,
    Aptitude_Score: float,
    Attendance: float,
    Supervisor_Evaluation: float,
    Report_Quality: float,
    Activity_Log_Frequency: int,
    Completion_Time: float,
    Feedback_Rating: float,
) -> dict:
    """
    Predict a student's internship performance category.

    Parameters
    ----------
    GPA : float                     0.0 - 4.0
    Course_Scores : float           0 - 100
    Aptitude_Score : float          0 - 100
    Attendance : float              0 - 100 (percentage)
    Supervisor_Evaluation : float   0 - 10
    Report_Quality : float          0 - 10
    Activity_Log_Frequency : int    0 - 29 (count)
    Completion_Time : float         1 - 7 (lower = faster completion)
    Feedback_Rating : float         1 - 5

    Returns
    -------
    dict with keys:
        predicted_class : str            one of Poor/Fair/Average/Good/Excellent
        confidence : float                probability of the predicted class
        probabilities : dict[str, float]  full class -> probability mapping
        model_used : str                  name of the underlying sklearn model
    """
    _load_artifacts()

    raw_values = {
        "GPA": GPA,
        "Course_Scores": Course_Scores,
        "Aptitude_Score": Aptitude_Score,
        "Attendance": Attendance,
        "Supervisor_Evaluation": Supervisor_Evaluation,
        "Report_Quality": Report_Quality,
        "Activity_Log_Frequency": Activity_Log_Frequency,
        "Completion_Time": Completion_Time,
        "Feedback_Rating": Feedback_Rating,
    }
    _validate_inputs(raw_values)

    feature_columns = _metadata["feature_columns"]
    input_row = pd.DataFrame([raw_values])[feature_columns]  # enforce correct column order

    if _metadata["uses_scaled_data"]:
        input_processed = pd.DataFrame(
            _scaler.transform(input_row), columns=feature_columns
        )
    else:
        input_processed = input_row

    pred_idx = int(_model.predict(input_processed)[0])
    idx_to_class = {int(k): v for k, v in _metadata["idx_to_class"].items()}
    predicted_class = idx_to_class[pred_idx]

    probabilities = {}
    confidence = None
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(input_processed)[0]
        probabilities = {idx_to_class[i]: round(float(p), 4) for i, p in enumerate(proba)}
        confidence = probabilities[predicted_class]

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "model_used": _metadata["best_model_name"],
    }


def predict_batch_vectorized(records: list) -> list:
    """
    Vectorized prediction for a large batch of records -- e.g. bulk-scoring
    thousands of imported training rows to pre-fill real student accounts.

    Unlike predict_performance() (one row at a time), this runs ONE
    model.predict() / model.predict_proba() call across the entire valid
    subset, so it stays fast for thousands of rows (~0.5s for 10,000 rows,
    matching evaluate_dataset()'s measured performance).

    Parameters
    ----------
    records : list of dicts, each containing the 9 feature keys (GPA,
        Course_Scores, ...). No ground-truth label needed (unlike
        evaluate_dataset).

    Returns
    -------
    list of dicts, ONE PER INPUT RECORD, in the same order as the input:
        on success: {"success": True, "predicted_class": ..., "confidence": ...,
                     "probabilities": {...}}
        on failure (invalid row): {"success": False, "error": "..."}
    """
    _load_artifacts()

    if not records:
        raise ValueError("No records provided")

    feature_columns = _metadata["feature_columns"]
    class_order = _metadata["class_order"]
    idx_to_class = {i: c for i, c in enumerate(class_order)}

    valid_indices = []
    valid_rows = []
    results = [None] * len(records)

    for i, rec in enumerate(records):
        try:
            _validate_inputs(rec)
            valid_rows.append({k: rec[k] for k in feature_columns})
            valid_indices.append(i)
        except ValueError as e:
            results[i] = {"success": False, "error": str(e)}

    if valid_rows:
        X = pd.DataFrame(valid_rows)[feature_columns]
        X_processed = (
            pd.DataFrame(_scaler.transform(X), columns=feature_columns)
            if _metadata["uses_scaled_data"] else X
        )

        preds = _model.predict(X_processed)
        probas = _model.predict_proba(X_processed) if hasattr(_model, "predict_proba") else None

        for row_pos, original_idx in enumerate(valid_indices):
            pred_idx = int(preds[row_pos])
            predicted_class = idx_to_class[pred_idx]
            entry = {"success": True, "predicted_class": predicted_class}
            if probas is not None:
                proba_row = probas[row_pos]
                probabilities = {idx_to_class[j]: round(float(p), 4) for j, p in enumerate(proba_row)}
                entry["confidence"] = probabilities[predicted_class]
                entry["probabilities"] = probabilities
            results[original_idx] = entry

    return results


def evaluate_dataset(records: list) -> dict:
    """
    Vectorized evaluation of a large batch of records against their known
    ground-truth labels — e.g. scoring the full training dataset to show
    live model accuracy/confusion-matrix stats in an admin dashboard.

    Unlike predict_performance() (one row at a time, used for single live
    predictions), this runs ONE model.predict() call across the entire
    input matrix, so it stays fast even for thousands of rows.

    Parameters
    ----------
    records : list of dicts, each containing the 9 feature keys (GPA,
        Course_Scores, ...) PLUS an "actual" key with the true label
        (one of Poor/Fair/Average/Good/Excellent).

    Returns
    -------
    dict with keys:
        total : int                        number of records evaluated
        accuracy : float
        precision_weighted / recall_weighted / f1_weighted : float
        confusion_matrix : list of lists   rows=actual, cols=predicted, in CLASS_ORDER
        class_order : list of str
        per_class : dict                   {class_name: {precision, recall, f1, support}}
        model_used : str
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, precision_recall_fscore_support,
    )

    _load_artifacts()

    if not records:
        raise ValueError("No records provided to evaluate")

    feature_columns = _metadata["feature_columns"]
    class_order = _metadata["class_order"]
    class_to_idx = {c: i for i, c in enumerate(class_order)}

    skipped = 0
    valid_records = []
    y_true = []

    for i, rec in enumerate(records):
        actual = rec.get("actual") or rec.get("Performance") or rec.get("performance")
        if actual not in class_to_idx:
            skipped += 1
            continue
        try:
            _validate_inputs(rec)
        except ValueError:
            skipped += 1
            continue
        valid_records.append(rec)
        y_true.append(class_to_idx[actual])

    if not valid_records:
        raise ValueError("No valid records to evaluate after filtering")

    X = pd.DataFrame(valid_records)[feature_columns]

    if _metadata["uses_scaled_data"]:
        X_processed = pd.DataFrame(_scaler.transform(X), columns=feature_columns)
    else:
        X_processed = X

    y_pred = _model.predict(X_processed)
    y_true = np.array(y_true)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_order))))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_order))), zero_division=0
    )

    per_class = {
        class_order[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(class_order))
    }

    return {
        "total": len(valid_records),
        "skipped": skipped,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_weighted": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": cm.tolist(),
        "class_order": class_order,
        "per_class": per_class,
        "model_used": _metadata["best_model_name"],
    }


if __name__ == "__main__":
    # Quick smoke test
    example = predict_performance(
        GPA=3.6,
        Course_Scores=85,
        Aptitude_Score=78,
        Attendance=92,
        Supervisor_Evaluation=8.5,
        Report_Quality=8.0,
        Activity_Log_Frequency=15,
        Completion_Time=3,
        Feedback_Rating=4.5,
    )
    print(json.dumps(example, indent=2))
