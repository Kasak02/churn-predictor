
# Week 2 Day 4 — SMOTE + Final Model Selection

## What is SMOTE?
SMOTE = Synthetic Minority Over-sampling Technique
Instead of duplicating existing minority samples,
SMOTE creates NEW synthetic churner samples by
interpolating between existing churner data points.

## SMOTE Configuration
- k_neighbors  : 5 (uses 5 nearest neighbours)
- random_state : 42
- Applied on   : X_train_proc only (never on test data)

## Class Distribution Change
| | Before SMOTE | After SMOTE |
|--|--|--|
| Retained (0) | 4139 | 4139 |
| Churned  (1) | 1495 | 4139 |
| Total        | 5634 | 8278 |
| Ratio        | 3:1 imbalanced | 1:1 balanced |

## Full Model Comparison
| Metric      | LR     | XGBoost | XGB+SMOTE |
|-------------|--------|---------|-----------|
| Accuracy %  | 73.95  | 73.67   | 75.51      |
| F1 churn    | 0.6157 | 0.6195  | 0.6213    |
| F1 weighted | 0.7533 | 0.7511  | 0.7666    |
| ROC-AUC     | 0.8458 | 0.8415  | 0.8374    |
| TP caught   | 294    | 302     | 283       |
| FN missed   | 80     | 72      | 91        |
| FP alarms   | 287    | 299     | 254       |

## Final Model Selected
Model  : XGBoost + SMOTE
Reason : Highest F1 churn score (0.6213)
Saved  : models/final_model.pkl

## Files Saved
- models/final_model.pkl
- models/X_train_smote.npy
- models/y_train_smote.npy
- reports/smote_comparison.png
- reports/full_model_comparison.png
- reports/confusion_matrix_comparison.png

## Tomorrow — Week 2 Day 5
- Install and set up SHAP
- Run TreeExplainer on final model
- Generate global feature importance summary plot
- Identify top features driving churn predictions
