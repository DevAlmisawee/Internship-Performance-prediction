# Internship Performance Prediction — ML Service for InternIQ

A production-ready supervised machine learning system that predicts a
student's internship performance (`Poor`, `Fair`, `Average`, `Good`,
`Excellent`) from academic and internship activity data. This is the ML
service layer for the **InternIQ** Express/MongoDB backend — the Flask API
here is a drop-in match for that backend's `mlService.js` contract.

## Project Structure

```
internship_performance_prediction/
├── data/
│   ├── prepare_dataset.py      # Cleans raw data + derives the categorical target
│   └── internship_data.csv     # Prepared dataset used for training
├── src/
│   ├── preprocessing.py        # Cleaning, encoding, scaling, train/test split
│   ├── eda.py                  # Exploratory Data Analysis + plots
│   ├── train.py                # Trains & tunes 5 models, saves the best one
│   ├── evaluate.py             # Metrics, confusion matrix, ROC-AUC, feature importance
│   ├── predict.py              # Loads saved model, exposes predict_performance()
│   └── utils.py                # Shared logging / IO helpers
├── models/                     # Saved artifacts (created after training)
│   ├── best_model.joblib
│   ├── model_<Name>.joblib     # Every tuned model (for comparison/evaluation)
│   ├── scaler.joblib
│   └── metadata.json
├── api/
│   └── app.py                  # Flask REST API (/predict, /predict/batch, /health)
├── logs/                       # EDA plots, confusion matrices, ROC curves, reports
├── run_pipeline.py             # Runs EDA -> train -> evaluate -> smoke test
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare the dataset (derives the categorical target from Performance_Score)
python data/prepare_dataset.py --input /path/to/your_raw_data.csv --output data/internship_data.csv

# 3. Run the rest of the pipeline (EDA -> train -> evaluate -> smoke test)
python run_pipeline.py

# 4. Start the REST API (runs on port 5001 by default -- Express occupies 5000)
python api/app.py
```

Or run each stage independently:

```bash
python data/prepare_dataset.py --input <raw.csv> --output data/internship_data.csv
python src/eda.py                 # prints stats, saves plots to logs/
python src/train.py               # trains & tunes 5 models, saves best to models/
python src/evaluate.py            # full evaluation report + plots
python src/predict.py             # smoke-test a single prediction
python api/app.py                 # start the REST API
```

## Dataset

The model trains on **real data only** — no synthetic features anywhere in
this version. All 9 features are genuine measured values from
`internship_data.csv` (10,000 records):

| Feature | Type | Range |
|---|---|---|
| GPA | float | 0.0 - 4.0 |
| Course_Scores | float | 0 - 100 |
| Aptitude_Score | float | 0 - 100 |
| Attendance | float | 0 - 100 (%) |
| Supervisor_Evaluation | float | 0 - 10 |
| Report_Quality | float | 0 - 10 |
| Activity_Log_Frequency | int | 0 - 29 |
| Completion_Time | float | 1 - 7 (lower = faster) |
| Feedback_Rating | float | 1 - 5 |
| **Performance** (target) | categorical | Poor / Fair / Average / Good / Excellent |

`data/prepare_dataset.py` derives the categorical target from the dataset's
continuous `Performance_Score` column using the InternIQ backend's fixed
thresholds (matches `scoreToLabel()` in `mlService.js` exactly):

| Overall Score (%) | Performance Level | Share of data |
|---|---|---|
| >= 85   | Excellent | 14.2% |
| 70 - 84 | Good      | 30.6% |
| 55 - 69 | Average   | 34.0% |
| 40 - 54 | Fair      | 18.0% |
| < 40    | Poor      | 3.1%  |

The extra real columns (`Intern_ID`, `Performance_Score`) are retained in
the output CSV for traceability; the modeling pipeline only reads the 9
canonical feature columns plus `Performance`.

## Methodology

1. **Cleaning** — duplicate rows dropped; missing numeric values imputed
   with the column median (robust to outliers).
2. **Encoding** — the target is mapped with a fixed **ordinal** encoding
   (`Poor=0 < Fair=1 < Average=2 < Good=3 < Excellent=4`) rather than
   sklearn's default alphabetical `LabelEncoder`, since the classes have a
   natural order.
3. **Scaling** — a `StandardScaler` is fit on the training split only (no
   leakage) and applied to distance/gradient-based models (Logistic
   Regression, SVM, Naive Bayes). Tree-based models (Random Forest, Decision
   Tree) use raw features since they are scale-invariant.
4. **Split** — stratified 80/20 train/test split to preserve class balance.
5. **Models compared**: Random Forest (primary), Decision Tree, Logistic
   Regression, SVM (RBF kernel), Gaussian Naive Bayes — each tuned via
   `GridSearchCV` or `RandomizedSearchCV` (the latter kicks in automatically
   for parameter grids larger than 24 combinations, to keep runtime
   reasonable — see `get_model_zoo()` in `train.py`).
6. **Model selection** — the model with the highest **weighted F1-score**
   on the held-out test set is saved as `models/best_model.joblib`. All
   five tuned models are also persisted individually for comparison.
7. **Evaluation** — accuracy, precision/recall/F1 (per-class + weighted),
   confusion matrix, full classification report, and multiclass ROC-AUC
   (One-vs-Rest) are computed for every model. Feature importance is
   reported both natively (Random Forest's impurity-based importances) and
   via permutation importance for the overall best model.

**Current results** (trained on the real 10,000-row dataset):
**Random Forest** is the best-performing model — accuracy 62.8%, weighted
F1 = 0.626, ROC-AUC ~0.85 — ahead of Naive Bayes (0.596), SVM (0.595),
Logistic Regression (0.576), and Decision Tree (0.530). Feature importance
is led by `GPA` (0.20), `Completion_Time` (0.16), `Report_Quality` (0.14),
and `Course_Scores` (0.14); `Activity_Log_Frequency` contributes least
(0.04). Most misclassifications fall between *adjacent* classes (e.g. Good
vs. Excellent) rather than distant ones — expected for an ordinal target,
and a sign the model has learned a sensible ordering rather than noise.
The `Poor` class (3.1% of data) has the lowest recall, which is typical for
a minority class this small; `class_weight="balanced"` is used throughout
to partially offset this.

## REST API — InternIQ Integration Contract

This API is built to be a **drop-in match** for the InternIQ Express
backend's `mlService.js` — same request field names (the dataset's own
PascalCase/underscore column names, not camelCase), same response shape.
No changes are needed on the Node side beyond setting `ML_API_URL`.

### `GET /health`
```json
{"status": "ok", "model_loaded": true}
```

### `GET /model-info`
Returns metadata about the currently deployed model, including comparison
metrics for all five candidate models.

### `POST /predict`
Request body — exact dataset column names, matching `buildMLPayload()` in
`mlService.js`:
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
        "GPA": 2.5,
        "Course_Scores": 75.3,
        "Aptitude_Score": 60.0,
        "Attendance": 85.5,
        "Supervisor_Evaluation": 7.2,
        "Report_Quality": 6.8,
        "Activity_Log_Frequency": 12,
        "Completion_Time": 3,
        "Feedback_Rating": 4.1
      }'
```
Response — flat JSON, matches what `mlService.js` reads directly:
```json
{
  "success": true,
  "prediction": "Good",
  "confidence": 0.5495,
  "model_version": "1.0.0",
  "probabilities": {"Poor": 0.0, "Fair": 0.0002, "Average": 0.0697, "Good": 0.5495, "Excellent": 0.3806},
  "model_used": "RandomForest"
}
```
On invalid input, the API returns `400` with a descriptive error, which
`mlService.js` catches and gracefully falls back to its rule-based scorer.

### `POST /predict/batch`
Accepts `{"records": [ {...}, {...} ]}` and returns a per-record result
array (errors for individual malformed records don't fail the whole batch).

## Wiring Up the InternIQ Backend

No Node code changes needed — just set the ML API URL in the backend's `.env`:
```bash
ML_API_URL=http://localhost:5001/predict
```
and start both services:
```bash
# Terminal 1 -- ML API
python api/app.py

# Terminal 2 -- Express backend
npm run dev
```
This was verified end-to-end by calling the backend's actual `mlService.js`
against this live Flask API — the full round trip (payload build -> HTTP
POST -> response parsing -> label validation) works with zero adapter code.

If `ML_API_URL` is unset or the Flask API is unreachable, `mlService.js`
automatically falls back to its own rule-based weighted-average scorer, so
the backend degrades gracefully rather than failing hard.

## Retraining on Updated Data

1. Replace the raw CSV with your updated data (same 9 feature columns +
   a `Performance_Score` column).
2. Run:
   ```bash
   python data/prepare_dataset.py --input /path/to/new_data.csv --output data/internship_data.csv
   python src/train.py
   python src/evaluate.py
   ```
3. Restart the API (`python api/app.py`) to pick up the newly saved
   `models/best_model.joblib`.

## Design Notes & Best Practices Applied

- No data leakage: the scaler is fit only on the training split.
- Reproducibility: a fixed `RANDOM_STATE=42` is used throughout (train/test
  split, model initialization, CV folds).
- Stratified splitting preserves class balance in train/test.
- `class_weight="balanced"` is used for applicable models to reduce bias
  toward the majority class (`Average`, 34%) and support the minority
  class (`Poor`, 3.1%).
- Input validation on the API layer rejects out-of-range or missing fields
  with clear error messages before they ever reach the model — bounds
  match the InternIQ backend's own Mongoose schema / express-validator
  rules exactly.
- Model, scaler, and metadata are versioned together as a single artifact
  bundle (`models/`) so the API never has to guess which preprocessing
  matches which model.
