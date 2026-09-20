# Credit Risk Prediction

Predicting the probability that a borrower will experience serious delinquency (90+ days past due) within the next two years, using the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset.

## Problem Statement

Given a borrower's credit history, income, and debt profile, predict the probability of serious delinquency — a binary classification problem (`SeriousDlqin2yrs`). The target is heavily imbalanced (6.7% positive class), which shapes every major decision in this project, from evaluation metrics to classification threshold.

## Dataset

- **Source:** [Give Me Some Credit](https://www.kaggle.com/datasets/brycecf/give-me-some-credit-dataset/code) — a Kaggle competition (2011), evaluated on ROC-AUC. Direct competition entry was unavailable, so the data was obtained from a public dataset mirror of the same files.
- **Size:** ~150,000 labeled rows (training), 101,503 unlabeled rows (Kaggle test set)
- **Features:** 10 numeric features — revolving credit utilization, age, past-due payment counts, debt ratio, monthly income, credit lines, real estate loans, dependents

## Repository Structure

```
credit-risk-prediction/
├── data/
│   ├── cs-training.csv
│   ├── cs-test.csv
│   └── processed/              # output of 02_preprocessing.ipynb
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb       # includes model comparison and evaluation
├── models/                     # saved models, scaler, results (generated)
├── requirements.txt
└── README.md
```

## Approach

**1. EDA** — data quality inspection: missing values, duplicates, anomalous values, target imbalance, feature distributions and correlations.

**2. Preprocessing** — see [Key Decisions](#key-decisions) below; produces two parallel, leakage-free feature sets (tree-based and linear/NN).

**3. Modeling** — 7 models trained and compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, and 3 neural network architectures. Best model selected by CV ROC-AUC, classification threshold tuned for the business context, final evaluation on a held-out test set, submission generated for the external Kaggle test set.

## Key Decisions

**Data leakage prevention.** The train/CV/test split happens *before* any statistic (imputation medians, clipping thresholds, scaler parameters) is computed. All such statistics are derived from the training set only and applied unchanged elsewhere — including to the external Kaggle test set. This was the single most consequential methodological decision in the project; catching and fixing an initial leak (statistics computed on the full dataset before splitting) was a major part of the preprocessing work.

**Two parallel feature sets.** Tree-based models (Decision Tree, Random Forest, XGBoost) use unscaled features with ordinal age encoding. Logistic Regression and the neural networks use scaled features with one-hot age encoding. This reflects a deliberate choice: linear models treat inputs as weighted, ordered quantities and need scaling plus non-ordinal categorical encoding, while tree-based models split on thresholds and are invariant to both.

**A shared data artifact.** The three "days past due" columns contained 225 rows with identical, implausibly large values (≥96) — not genuine late-payment counts, but a placeholder/sentinel code. These were flagged (`has_special_value`) and replaced with each column's own median, computed on the clean subset of the training data.

**Metric choice.** Given the 93.3% / 6.7% class imbalance, accuracy is misleading (a model predicting "no default" for everyone would score ~93%). ROC-AUC (the competition's own metric) is used for model comparison; F1 and recall/precision are used to evaluate behavior at a specific threshold.

**Threshold selection on CV, not test.** The classification threshold was tuned by evaluating recall/precision across a range of values on `x_cv` — treated as a model-selection decision, the same way hyperparameters are. `x_test` was touched exactly once, for the final, unbiased performance estimate.

## Results

| Model | CV ROC-AUC | CV F1 |
|---|---|---|
| **XGBoost** | **0.865** | 0.296 |
| Random Forest | 0.864 | 0.259 |
| Neural Network (model_2) | 0.861 | 0.345 |
| Neural Network (model_1) | 0.858 | 0.308 |
| Neural Network (model_3, BatchNorm) | 0.857 | 0.398 |
| Decision Tree | 0.847 | 0.264 |
| Logistic Regression | 0.839 | 0.220 |

**Final model: XGBoost**, threshold = 0.2 (tuned on CV; default 0.5 would miss 79% of actual defaults)

| | Threshold 0.5 (default) | Threshold 0.2 (selected) |
|---|---|---|
| Recall (defaults) | 0.21 | 0.51 |
| Precision (defaults) | 0.59 | 0.38 |

| Metric | Value |
|---|---|
| Test ROC-AUC | 0.862 |
| Test F1 | 0.434 |
| **Kaggle Private Score** | **0.867** |
| Kaggle Public Score | 0.861 |

The Kaggle score — computed on data never seen during development — closely matches the internal CV and test results, supporting that the pipeline is free of data leakage.

**Top features by importance (XGBoost):** the three "days past due" columns and `RevolvingUtilizationOfUnsecuredLines` — consistent with what both mutual information and correlation analysis identified during preprocessing.

## How to Run

```bash
pip install -r requirements.txt
```

Run the notebooks in order: `01_eda.ipynb` → `02_preprocessing.ipynb` → `03_modeling.ipynb`. All random seeds are fixed, so results are reproducible end to end.

## Limitations & Future Work

- The current pipeline uses a manual sequence of transformations with saved fit statistics, rather than an `sklearn.Pipeline`. For production use, wrapping the logic in a `Pipeline` with custom `Transformer` classes would reduce the risk of train/inference inconsistency (a bug of exactly this kind — a stale variable reference causing a mismatched value to be applied — was caught and fixed during development).
- Hyperparameter search used a modest grid/iteration budget; a wider search (or Bayesian optimization) could yield a further, likely small, improvement.
- Predicted probabilities are not calibrated; for a real lending decision, calibration (e.g., Platt scaling) would make the probabilities more directly interpretable as default likelihoods.
- The classification threshold was chosen via a qualitative recall/precision trade-off; a formal cost-based analysis (explicit dollar cost per false negative vs. false positive) would give a more rigorous justification.
- No systematic error analysis was performed on misclassified cases. Inspecting false negatives (missed defaulters) and false positives individually could reveal shared patterns — e.g., a borrower segment or feature combination the model consistently struggles with — that would inform further feature engineering or highlight a subgroup where the model's predictions are less reliable.

