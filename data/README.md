# Dataset files

`loan_dataset.csv` is the labeled Kaggle training CSV used by LoanIQ. It must include `Loan_Status` with binary `Y`/`N` labels. `loan_test.csv` is the provided unlabeled Kaggle companion file and cannot be used for supervised training.

To train on another copy, replace `loan_dataset.csv` or pass the path to `python src/train_model.py <csv-path>`. The loader accepts common variations in the field headers and omits `Loan_ID` from model features.

Dataset source: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
