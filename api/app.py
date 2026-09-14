"""
app.py
------
Flask REST API exposing the trained internship performance prediction
model. Built to be a DROP-IN match for the InternIQ Express backend's
`mlService.js` — same request field names, same response shape — so no
changes are needed on the Node side beyond setting ML_API_URL.

Contract (see mlService.js `buildMLPayload` / `predictPerformance`)
---------------------------------------------------------------------
Request body (POST /predict), exact dataset column names:
    {
        "GPA": 2.5,
        "Course_Scores": 75.3,
        "Aptitude_Score": 60.0,
        "Attendance": 85.5,
        "Supervisor_Evaluation": 7.2,
        "Report_Quality": 6.8,
        "Activity_Log_Frequency": 12,
        "Completion_Time": 3,
        "Feedback_Rating": 4.1
    }

Response body (flat JSON, matches what mlService.js reads):
    {
        "prediction": "Good",       <- one of Excellent/Good/Average/Fair/Poor
        "confidence": 0.87,          <- probability of the predicted class
        "model_version": "1.0.0",
        "probabilities": {...}       <- extra field, ignored by mlService.js but useful for debugging/logging
    }

Endpoints
---------
GET  /health          -> liveness/readiness check
GET  /model-info       -> metadata about the currently loaded model
POST /predict          -> single prediction (matches mlService.js contract)
POST /predict/batch    -> batch predictions (array of student records)

Run:
    python api/app.py
    (defaults to http://0.0.0.0:5001 -- the Express backend runs on 5000,
    so this deliberately defaults to a different port. Set ML_API_URL in
    the Express backend's .env to http://localhost:5001/predict)
"""

import os
import sys
import json
import logging
from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import predict as predict_module
from src.predict import predict_performance, evaluate_dataset, predict_batch_vectorized, ModelNotLoadedError, _load_artifacts, FEATURE_BOUNDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("internship_api")

app = Flask(__name__)

# Exact dataset column names -- matches buildMLPayload() in mlService.js
REQUIRED_FIELDS = list(FEATURE_BOUNDS.keys())

MODEL_VERSION = "1.0.0"


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception("Unhandled server error")
    return jsonify({"error": "Internal server error"}), 500


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness/readiness probe."""
    try:
        _load_artifacts()
        model_loaded = True
    except ModelNotLoadedError:
        model_loaded = False
    return jsonify({"status": "ok", "model_loaded": model_loaded}), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    """Return metadata about the currently loaded best model."""
    try:
        _load_artifacts()
    except ModelNotLoadedError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify({
        "best_model_name": predict_module._metadata["best_model_name"],
        "model_version": MODEL_VERSION,
        "feature_columns": predict_module._metadata["feature_columns"],
        "class_order": predict_module._metadata["class_order"],
        "performance_summary": {
            name: {
                "test_f1_weighted": res["test_f1_weighted"],
                "cv_f1_weighted": res["cv_f1_weighted"],
            }
            for name, res in predict_module._metadata["results_summary"].items()
        },
    }), 200


def _extract_and_validate(payload: dict):
    """Extract required fields from a JSON payload; raise ValueError if missing."""
    missing = [f for f in REQUIRED_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")
    return {f: payload[f] for f in REQUIRED_FIELDS}


@app.route("/test-evaluation", methods=["GET"])
def test_evaluation():
    """
    Return the best model's genuine held-out test evaluation (accuracy,
    precision/recall/F1, ROC-AUC, confusion matrix), as saved by
    src/evaluate.py to models/test_evaluation.json.

    This is distinct from POST /evaluate, which re-scores whatever dataset
    the caller sends it (used by the Express backend's "training-evaluation"
    endpoint for a live in-sample sanity check against imported data,
    including rows the model was trained on -- that number reads higher
    than this one and should not be presented as the model's real accuracy).
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "test_evaluation.json")
    if not os.path.exists(path):
        return jsonify({
            "error": "No held-out test evaluation available yet. Run `python src/evaluate.py` after training."
        }), 404

    with open(path) as f:
        data = json.load(f)

    return jsonify(data), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Single-record prediction endpoint. Matches the InternIQ backend's
    mlService.js contract exactly -- request field names are the dataset's
    own column names, response is a flat object with `prediction` at the
    top level.
    """
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    try:
        fields = _extract_and_validate(payload)
        result = predict_performance(**fields)
        return jsonify({
            "success": True,
            "prediction": result["predicted_class"],
            "confidence": result["confidence"],
            "model_version": MODEL_VERSION,
            "probabilities": result["probabilities"],
            "model_used": result["model_used"],
        }), 200
    except ModelNotLoadedError as e:
        logger.error(f"Model not loaded: {e}")
        return jsonify({"success": False, "error": str(e)}), 503
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"success": False, "error": "Prediction failed", "details": str(e)}), 500


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Batch evaluation endpoint for scoring a whole dataset (e.g. the imported
    training set) against its known ground-truth labels, and returning
    accuracy/confusion-matrix/per-class metrics. Vectorized -- one
    model.predict() call across the whole batch, not a loop, so this stays
    fast even for thousands of rows.

    Expected JSON body:
    {
        "records": [
            { "GPA": 2.5, "Course_Scores": 75.3, ..., "actual": "Good" },
            ...
        ]
    }
    "actual" can also be spelled "Performance" or "performance".
    """
    payload = request.get_json(silent=True)
    if payload is None or "records" not in payload:
        return jsonify({"error": "Request body must be JSON with a 'records' array"}), 400

    records = payload["records"]
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "'records' must be a non-empty array"}), 400

    try:
        summary = evaluate_dataset(records)
        return jsonify({"success": True, **summary}), 200
    except ModelNotLoadedError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Evaluation failed")
        return jsonify({"success": False, "error": "Evaluation failed", "details": str(e)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Batch prediction endpoint.

    Expected JSON body:
    {
        "records": [
            { "GPA": 3.6, "Course_Scores": 85, ... },
            { "GPA": 2.1, "Course_Scores": 55, ... }
        ]
    }
    """
    payload = request.get_json(silent=True)
    if payload is None or "records" not in payload:
        return jsonify({"error": "Request body must be JSON with a 'records' array"}), 400

    records = payload["records"]
    if not isinstance(records, list) or len(records) == 0:
        return jsonify({"error": "'records' must be a non-empty array"}), 400

    try:
        raw_results = predict_batch_vectorized(records)
    except ModelNotLoadedError as e:
        return jsonify({"success": False, "error": str(e)}), 503
    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({"success": False, "error": "Batch prediction failed", "details": str(e)}), 500

    results = []
    for i, r in enumerate(raw_results):
        if r["success"]:
            results.append({
                "index": i,
                "success": True,
                "prediction": r["predicted_class"],
                "confidence": r.get("confidence"),
                "model_version": MODEL_VERSION,
                "probabilities": r.get("probabilities", {}),
            })
        else:
            results.append({"index": i, "success": False, "error": r["error"]})

    return jsonify({"success": True, "results": results}), 200


if __name__ == "__main__":
    # Warm up the model at startup so the first request isn't slow
    try:
        _load_artifacts()
        logger.info(f"Model loaded successfully: {predict_module._metadata['best_model_name']}")
    except ModelNotLoadedError as e:
        logger.warning(f"Model not yet available at startup: {e}")

    # Defaults to 5001 -- the Express backend already occupies 5000.
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
