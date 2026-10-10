# LoanIQ — AI-Based Loan Approval System

LoanIQ is a local, academic decision-support web app for exploring historical loan outcomes and trying a sample application against a trained classification model. Its overview, application analysis, model performance, influencing factors, dataset insights, and project information are presented on separate pages. It does **not** approve loans or represent an actual lender.

## Problem and objective

Historical application data can be used to study patterns associated with loan outcomes. This project demonstrates a repeatable preprocessing and model-evaluation workflow, and presents predictions as estimates for human review.

## Dataset

The labeled Kaggle Loan Prediction training CSV is included at `data/loan_dataset.csv`. The companion Kaggle test CSV is at `data/loan_test.csv`; it has no `Loan_Status` target, so it is retained as reference data and is not used for supervised training. Dataset source: [Loan Prediction Problem Dataset on Kaggle](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset).

To use a different copy of the dataset, put the labeled CSV at `data/loan_dataset.csv`, or pass a path to `python src/train_model.py path/to/LoanApprovalPrediction.csv`. The loader normalizes common header variations and removes `Loan_ID`. It expects a binary `Loan_Status` column containing `Y`/`N` (or `Approved`/`Rejected`). If the dataset is unavailable, the web app displays setup instructions instead of inventing results.

## Features

- Separate responsive pages for Overview, Application Analysis, Model Performance, Influencing Factors, Dataset Insights and About.
- Overview with training-set approval distribution, selected model, and record/feature counts.
- Server-validated application form with dataset-derived fields, ranges, categories and defaults.
- Held-out model comparison, confusion matrix, and performance metrics.
- Permutation importance measured on held-out records, mapped to original input features.
- Observed outcome rates for credit history, education, and property area where available.
- Local inference; submitted application values are not persisted.

## Machine-learning workflow

The trainer cleans the target and drops the identifier, uses a stratified 80/20 split (random state 42), and fits preprocessing only on training folds. Numeric values receive median imputation; categorical values receive most-frequent imputation and one-hot encoding. Logistic Regression uses standardized numeric fields. Decision Tree and Random Forest use the same imputation/encoding pipeline without scaling.

Each model is evaluated on the same holdout with accuracy, approval precision, approval recall, F1, ROC-AUC and confusion matrix. The selected model maximizes held-out F1; ROC-AUC breaks ties. That pipeline is then fitted to all labeled records for the interactive form. Evaluation numbers remain the held-out scores. Feature analysis uses permutation importance on the held-out set; values are model associations, not causes.

### Algorithms

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Project structure

```text
app.py                         Flask routes and validated prediction API
src/train_model.py             Data prep, training, evaluation and artifacts
templates/base.html             Shared navigation and page layout
templates/overview.html         Overview page
templates/application-analysis.html
templates/model-performance.html
templates/influencing-factors.html
templates/dataset-insights.html
templates/about.html
static/css/style.css           Responsive visual system
static/js/app.js                Dashboard and prediction interactions
data/loan_dataset.csv           Labeled training dataset
data/loan_test.csv              Unlabeled Kaggle test dataset
models/loan_model.joblib        Trained sklearn pipeline (generated)
outputs/model_metrics.json      Holdout metrics and dashboard data (generated)
```

## Install and run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python src/train_model.py
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The Flask app loads the saved pipeline; training is a separate step. To train against another labeled CSV:

```bash
python src/train_model.py path/to/loan_dataset.csv
```

## Example prediction

The form uses valid defaults derived from the included training dataset. Adjust the fields and choose **Analyze application**. The result includes the predicted class, the model's estimated approval probability, and the model name. This probability is not a guarantee and is not a lending decision.

## Evaluation metrics

The dashboard reports accuracy, precision, recall, F1 and ROC-AUC for each classifier, along with the selected classifier's confusion matrix. Metrics are calculated from a stratified test split; they should not be interpreted as evidence of suitability for real credit decisions.

## Screenshots

Add screenshots of the dashboard, model comparison and prediction result here.

## Limitations and responsible use

- The dataset is small and historical; it may be imbalanced, outdated or biased. The train/test split is a demonstration, not independent external validation.
- Feature importance and group rates are descriptive associations and do not establish causation or fairness.
- The displayed class probability is a model score and has not been calibrated for lending use.
- No real financial decision, underwriting recommendation or eligibility guarantee should be based on this application. A qualified human decision-maker would need to review any real application using appropriate, lawful, validated processes.
- Inputs are processed in memory for the request and are not saved by this application. Do not enter real applicant personal information.

## Future improvements

Add cross-validation and calibration, robustness and subgroup fairness audits, model/data versioning, and accessibility testing before any research deployment beyond this classroom demonstration.

## Technologies

Python, Flask, Pandas, NumPy, Scikit-learn, Joblib, HTML, CSS and JavaScript.
