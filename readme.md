# MLE Case Study 2 — Credit Card Fraud Detection

A machine learning case study that flags fraudulent credit card transactions from transaction-level behavioral data, handling severe class imbalance with SMOTE and an XGBoost classifier.

## Problem Statement

Fraudulent transactions are rare compared to normal ones, making this a heavily imbalanced classification problem. This project builds a model to predict `IsFraud` (0 = normal, 1 = fraud) for a transaction based on its amount, timing, and account/customer behavior signals, and tunes the decision threshold to balance precision and recall for the fraud class.

## Dataset

`credit_card_transactions.csv` — 1,000 rows, 8 columns:

| Column | Description |
|---|---|
| `TransactionID` | Unique transaction identifier |
| `TransactionAmount` | Transaction amount |
| `TransactionHour` | Hour of day the transaction occurred |
| `CustomerAge` | Age of the customer |
| `AccountAgeDays` | Age of the account, in days |
| `NumPrevTransactions` | Number of prior transactions on the account |
| `DistanceFromHomeKM` | Distance between transaction location and customer's home |
| `IsFraud` | **Target** — whether the transaction is fraudulent |

Class distribution is imbalanced: **94% normal (940) vs. 6% fraud (60)**.

## Approach

1. **EDA** — class balance, summary statistics, and normal-vs-fraud feature means (`case study 2.ipynb`).
2. **Preprocessing**
   - Selected behavioral features: `TransactionAmount`, `TransactionHour`, `CustomerAge`, `AccountAgeDays`, `NumPrevTransactions`, `DistanceFromHomeKM`.
   - Split into train/test sets (70/30, stratified on `IsFraud`).
   - Applied **SMOTE** to the training set to oversample the minority (fraud) class.
3. **Modeling** — trained an `XGBClassifier` (100 estimators, max depth 4, learning rate 0.1) on the SMOTE-balanced training data.
4. **Threshold tuning** — swept decision thresholds (0.10–0.90) and selected the one maximizing F1-score for the fraud class.
5. **Evaluation** — confusion matrix, classification report, ROC-AUC, and feature importance.
6. **Submission** — exported per-transaction fraud probabilities and predictions to `submission.csv`.

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.995 |
| Best threshold | 0.25 |
| Fraud precision | 0.86 |
| Fraud recall | 1.00 |
| Fraud F1-score | 0.92 |
| Overall accuracy | 0.99 |

At the tuned threshold, the model correctly catches **all 18 fraud cases** in the test set, with 3 false positives out of 282 normal transactions.

## Project Structure

```
MLE_CASE_STUDY_2/
├── Readme.md
├── case study 2.ipynb              # Main analysis & model notebook
├── credit_card_transactions.csv    # Dataset
└── submission (1).csv              # Model output: fraud probabilities & predictions
```

## Getting Started

```bash
pip install numpy pandas matplotlib scikit-learn imbalanced-learn xgboost
jupyter notebook "case study 2.ipynb"
```

The notebook expects `credit_card_transactions.csv` to be uploaded via `google.colab.files.upload()`. To run locally instead, replace that cell with:

```python
df = pd.read_csv("credit_card_transactions.csv")
```