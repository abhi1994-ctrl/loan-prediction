"""LoanIQ academic loan outcome decision-support dashboard."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("LOANIQ_MODEL_PATH", ROOT / "models" / "loan_model.joblib"))
METRICS_PATH = Path(os.environ.get("LOANIQ_METRICS_PATH", ROOT / "outputs" / "model_metrics.json"))
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


def load_artifacts():
    if not MODEL_PATH.is_file() or not METRICS_PATH.is_file():
        return None, None
    try:
        return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, EOFError, TypeError):
        return None, None


def create_app():
    return app


@app.get("/")
def index():
    return render_template("overview.html", active_page="overview", page_title="Overview", page_intro="A clear snapshot of historical outcomes and the model behind LoanIQ.")


@app.get("/<page_name>")
def page(page_name):
    pages = {
        "application-analysis": ("Application Analysis", "Explore a sample application with the trained model."),
        "model-performance": ("Model Performance", "Compare held-out results and inspect the selected model's confusion matrix."),
        "influencing-factors": ("Influencing Factors", "See which input fields most affected model scores on held-out records."),
        "dataset-insights": ("Dataset Insights", "Explore the actual training data, distributions and summary statistics."),
        "about": ("About LoanIQ", "Project scope, machine-learning workflow and responsible-use notes."),
    }
    if page_name not in pages:
        return "Page not found", 404
    title, intro = pages[page_name]
    return render_template(f"{page_name}.html", active_page=page_name, page_title=title, page_intro=intro)


@app.get("/api/dashboard")
def dashboard():
    model, metrics = load_artifacts()
    if model is None:
        return jsonify({"ready": False, "message": "Model artifacts are missing. Add the training CSV at data/loan_dataset.csv, install requirements, then run: python src/train_model.py"}), 503
    return jsonify({"ready": True, **metrics})


@app.post("/api/predict")
def predict():
    model, metrics = load_artifacts()
    if model is None:
        return jsonify({"error": "The model is not trained yet. Run python src/train_model.py first."}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send application fields as a JSON object."}), 400
    features = metrics.get("feature_schema", [])
    row = {}
    for field in features:
        name = field["name"]
        value = payload.get(name)
        if value is None or value == "":
            return jsonify({"error": f"{field['label']} is required."}), 400
        if field["type"] == "number":
            try:
                value = float(value)
            except (TypeError, ValueError):
                return jsonify({"error": f"{field['label']} must be numeric."}), 400
            if not pd.notna(value) or value < field["min"] or value > field["max"]:
                return jsonify({"error": f"{field['label']} must be between {field['min']:g} and {field['max']:g}."}), 400
        elif value not in field["options"]:
            return jsonify({"error": f"Choose a valid value for {field['label']}."}), 400
        row[name] = value
    frame = pd.DataFrame([row], columns=[f["name"] for f in features])
    try:
        probabilities = model.predict_proba(frame)[0]
        classes = list(model.classes_)
        approval_idx = classes.index(metrics["positive_label"])
        probability = float(probabilities[approval_idx])
        prediction = str(model.predict(frame)[0])
    except (ValueError, KeyError, IndexError) as exc:
        return jsonify({"error": f"Could not analyze this application: {exc}"}), 400
    return jsonify({"prediction": prediction, "approved": prediction == metrics["positive_label"],
                    "probability": probability, "model": metrics["selected_model"],
                    "inputs": row, "note": "Model confidence is not a guarantee or a lending decision."})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "8000")), debug=False)
