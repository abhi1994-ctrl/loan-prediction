"""Train and evaluate LoanIQ classification pipelines from the Kaggle CSV."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, make_scorer)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path(__import__("os").environ.get("LOANIQ_DATA_PATH", ROOT / "data" / "loan_dataset.csv"))
MODEL_PATH = ROOT / "models" / "loan_model.joblib"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"

ALIASES = {
    "loanid": "Loan_ID", "gender": "Gender", "married": "Married", "dependents": "Dependents",
    "education": "Education", "selfemployed": "Self_Employed", "applicantincome": "ApplicantIncome",
    "coapplicantincome": "CoapplicantIncome", "loanamount": "LoanAmount",
    "loanamountterm": "Loan_Amount_Term", "credithistory": "Credit_History",
    "propertyarea": "Property_Area", "loanstatus": "Loan_Status", "status": "Loan_Status",
}


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    names = {}
    for column in normalized.columns:
        key = "".join(ch.lower() for ch in str(column) if ch.isalnum())
        names[column] = ALIASES.get(key, str(column).strip())
    normalized = normalized.rename(columns=names)
    if len(set(normalized.columns)) != len(normalized.columns):
        raise ValueError("Column normalization produced duplicate names; please review the CSV header.")
    return normalized


def load_dataset(path: Path = DATA_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"Training dataset not found: {path}. Place the labeled Kaggle CSV at data/loan_dataset.csv or pass its path to this script.")
    frame = normalize_columns(pd.read_csv(path))
    if "Loan_Status" not in frame:
        raise ValueError("The training CSV must include a Loan_Status target column (Y/N). The Kaggle test file has no labels and cannot train a model.")
    frame = frame.drop(columns=["Loan_ID"], errors="ignore")
    frame["Loan_Status"] = frame["Loan_Status"].astype("string").str.strip().str.upper().replace({"APPROVED": "Y", "REJECTED": "N"})
    frame = frame.dropna(subset=["Loan_Status"])
    labels = sorted(frame.Loan_Status.unique().tolist())
    if len(labels) != 2 or not {"Y", "N"}.issubset(set(labels)):
        raise ValueError(f"Loan_Status must contain binary Y/N values; found {labels}.")
    X = frame.drop(columns="Loan_Status")
    y = frame["Loan_Status"].astype(str)
    for col in X:
        if col in {"ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"}:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X, y


def make_pipeline(model, numeric, categorical, scale=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = [("numeric", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", prep), ("model", model)])


def feature_schema(X):
    fields = []
    for name in X.columns:
        series = X[name]
        if pd.api.types.is_numeric_dtype(series):
            vals = pd.to_numeric(series, errors="coerce")
            low, high = float(vals.min()), float(vals.max())
            default = float(vals.median())
            if name in ("Credit_History", "Loan_Amount_Term"):
                default = float(vals.mode().iloc[0])
            fields.append({"name": name, "label": name.replace("_", " ").replace("Income", " Income"), "type": "number", "min": low, "max": high, "default": default, "step": 1 if name in ("Loan_Amount_Term", "Credit_History") else "any"})
        else:
            options = sorted(str(v) for v in series.dropna().unique())
            default = str(series.mode().iloc[0]) if series.notna().any() else options[0]
            fields.append({"name": name, "label": name.replace("_", " "), "type": "category", "options": options, "default": default})
    return fields


def train(data_path: Path = DATA_PATH):
    X, y = load_dataset(data_path)
    if y.value_counts().min() < 2:
        raise ValueError("Need at least two examples from each loan outcome to make a stratified evaluation split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    categorical = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric = [col for col in X.columns if col not in categorical]
    candidates = [
        ("Logistic Regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42), True),
        ("Decision Tree", DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=3, random_state=42), False),
        ("Random Forest", RandomForestClassifier(n_estimators=350, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1), False),
    ]
    reports, fitted = [], {}
    for name, estimator, scale in candidates:
        pipe = make_pipeline(estimator, numeric, categorical, scale)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        probabilities = pipe.predict_proba(X_test)[:, list(pipe.classes_).index("Y")]
        reports.append({"name": name, "accuracy": float(accuracy_score(y_test, pred)),
                        "precision": float(precision_score(y_test, pred, pos_label="Y", zero_division=0)),
                        "recall": float(recall_score(y_test, pred, pos_label="Y", zero_division=0)),
                        "f1": float(f1_score(y_test, pred, pos_label="Y", zero_division=0)),
                        "roc_auc": float(roc_auc_score((y_test == "Y").astype(int), probabilities)),
                        "confusion_matrix": confusion_matrix(y_test, pred, labels=["N", "Y"]).tolist()})
        fitted[name] = pipe
    best = max(reports, key=lambda m: (m["f1"], m["roc_auc"]))
    selected = make_pipeline(dict((n, (e, s)) for n, e, s in candidates)[best["name"]][0], numeric, categorical, dict((n, s) for n, e, s in candidates)[best["name"]])
    selected.fit(X, y)
    importance_result = permutation_importance(fitted[best["name"]], X_test, y_test, scoring=make_scorer(f1_score, pos_label="Y"), n_repeats=8, random_state=42, n_jobs=-1)
    importance = [{"feature": str(name).replace("_", " "), "importance": max(0.0, float(score)), "raw_importance": float(score)}
                  for name, score in zip(X.columns, importance_result.importances_mean)]
    importance.sort(key=lambda item: item["importance"], reverse=True)
    importance = importance[:10]
    counts = y.value_counts()
    numeric_statistics = {}
    distributions = {}
    for col in numeric:
        values = pd.to_numeric(X[col], errors="coerce").dropna()
        if values.empty:
            continue
        stats = values.describe()
        numeric_statistics[col] = {key: float(stats[key]) for key in ("mean", "50%", "min", "max")}
        if col in ("ApplicantIncome", "LoanAmount"):
            hist_counts, edges = np.histogram(values, bins=7)
            distributions[col] = [{"label": f"{edges[i]:,.0f}–{edges[i + 1]:,.0f}", "count": int(hist_counts[i])} for i in range(len(hist_counts))]
    metrics = {
        "selected_model": best["name"], "selection_reason": "Highest held-out F1-score; ROC-AUC breaks ties.",
        "positive_label": "Y", "records": int(len(X)), "feature_count": int(X.shape[1]),
        "missing_before": int(X.isna().sum().sum()), "missing_after": 0,
        "class_counts": {"Approved": int(counts.get("Y", 0)), "Rejected": int(counts.get("N", 0))},
        "target_distribution": [{"label": "Approved", "count": int(counts.get("Y", 0))}, {"label": "Rejected", "count": int(counts.get("N", 0))}],
        "models": reports, "selected_metrics": best,
        "confusion_labels": ["Rejected", "Approved"], "feature_importance": importance,
        "feature_schema": feature_schema(X),
        "numeric_statistics": numeric_statistics, "distributions": distributions,
        "approval_by": {
            col: [{"label": str(group), "rate": float((part.__target == "Y").mean()), "count": int(len(part))}
                  for group, part in X.assign(__target=y).groupby(col, dropna=False)]
            for col in [c for c in ["Credit_History", "Education", "Property_Area"] if c in X.columns]
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Selected {best['name']} (F1={best['f1']:.3f}, ROC-AUC={best['roc_auc']:.3f})")
    print(f"Saved model: {MODEL_PATH}\nSaved metrics: {METRICS_PATH}")
    return selected, metrics


if __name__ == "__main__":
    train()
