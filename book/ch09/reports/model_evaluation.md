# TalentLens Role Classifier — Model Evaluation Report

## Model Selection (5-fold cross-validation)

| Model | F1 (macro) | Std |
|-------|-----------|-----|
| Random Forest | 0.869 | ±0.058 |
| Logistic Regression ← selected | 0.867 | ±0.050 |
| SGD Classifier | 0.843 | ±0.061 |

**Selected model:** Logistic Regression (simplest model within one standard deviation of the top CV score)

## Holdout Evaluation

**Overall F1 (macro):** 0.851

| Role | Precision | Recall | F1 | Support |
|------|-----------|--------|-----|---------|
| AI Engineer | 0.818 | 0.947 | 0.878 | 19 |
| ML Engineer | 0.793 | 0.852 | 0.821 | 27 |
| Data Scientist | 0.923 | 0.889 | 0.906 | 27 |
| Data Engineer | 0.895 | 0.944 | 0.919 | 18 |
| Data Analyst | 0.846 | 0.846 | 0.846 | 13 |

## Interpretation

- Overall F1 of 0.851 — good — usable with confidence thresholding
- See confusion matrix for class-level error patterns.
- See feature importance chart for which words drive each prediction.

## Model location
Saved to: `book/ch09/models/role_classifier.joblib`

## Usage
```python
import joblib
pipeline = joblib.load('book/ch09/models/role_classifier.joblib')
role, confidence = predict_role({'title': 'ML Engineer', 'description': '...'}, pipeline)
```