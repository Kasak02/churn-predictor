
# Week 2 Day 1 — Train/Test Split + Preprocessing Pipeline

## What was done today

### Train/Test Split
- Split ratio : 80% train / 20% test
- Train size  : 5,634 customers
- Test size   : 1,409 customers
- stratify=y  : ensures equal churn rate in both splits (26.5%)
- random_state: 42 (reproducible results every run)

### Why stratify?
Without stratify, random chance could put more churners in train
than test, making evaluation unreliable. Stratification guarantees
the 26.5% churn rate is preserved in both splits.

### Preprocessing Pipeline (ColumnTransformer)
| Transformer    | Columns              | Why |
|----------------|----------------------|-----|
| StandardScaler | numerical (6 cols)   | XGBoost works better with normalised features |
| OneHotEncoder  | categorical (4 cols) | ML models need numbers not text |
| passthrough    | binary (13 cols)     | Already 0/1, no change needed |

### Data Leakage Prevention
- preprocessor.fit_transform() called ONLY on X_train
- preprocessor.transform() called on X_test (no fitting)
- This prevents test data statistics leaking into training

### Shape after preprocessing
- Before : 23 columns
- After  : 33 columns (OneHotEncoder added 10 binary columns)

## Files Saved
- models/preprocessor.pkl     ← reusable preprocessor
- models/X_train_proc.npy     ← processed training features
- models/X_test_proc.npy      ← processed test features
- models/y_train.npy          ← training labels
- models/y_test.npy           ← test labels
- models/feature_names.json   ← feature names for SHAP

## Tomorrow — Week 2 Day 2
- Train Logistic Regression as baseline model
- Evaluate: accuracy, F1, ROC-AUC, confusion matrix
- This gives us a benchmark to beat with XGBoost

# Week 2 Day 2 — Logistic Regression Baseline

## Model Configuration
- Model        : LogisticRegression
- max_iter     : 1000 (ensures convergence)
- class_weight : balanced (handles 26.5% churn imbalance)
- random_state : 42

## Actual Results
| Metric         | Score  |
|----------------|--------|
| Accuracy       | 73.95% |
| F1 (churn)     | 0.6157 |
| F1 weighted    | 0.7533 |
| ROC-AUC        | 0.8458 |

## Confusion Matrix
|                | Predicted Retained | Predicted Churned |
|----------------|--------------------|-------------------|
| Actual Retained| 748 (TN)           | 287 (FP)          |
| Actual Churned | 80  (FN)           | 294 (TP)          |

## Classification Report
              precision  recall  f1-score  support
Retained         0.90    0.72      0.80     1035
Churned          0.51    0.79      0.62      374
accuracy                           0.74     1409
macro avg        0.70    0.75      0.71     1409
weighted avg     0.80    0.74      0.75     1409

## Key Observations
- Accuracy (73.95%) barely beats the 73.5% do-nothing baseline
- class_weight=balanced boosted recall to 0.79 (catching churners)
  but hurt precision badly (0.51 — too many false alarms)
- 287 false positives = wasted retention emails sent to happy customers
- 80 false negatives = churners we completely missed (costly!)
- ROC-AUC of 0.8458 is decent — model ranks churners well

## Primary Metric Decision
F1 Score on churn class (0.6157) is our target to beat because:
1. Accuracy is misleading due to class imbalance
2. We care most about catching actual churners
3. F1 balances precision and recall equally

## Targets for XGBoost (Week 2 Day 3)
| Metric     | LR Baseline | XGBoost Target |
|------------|-------------|----------------|
| Accuracy   | 73.95%      | > 78%          |
| F1 churn   | 0.6157      | > 0.64         |
| ROC-AUC    | 0.8458      | > 0.86         |
| Precision  | 0.51        | > 0.60         |
| FN (missed)| 80          | < 70           |

## Tomorrow — Week 2 Day 3
- Train XGBoost classifier on same train/test split
- Compare all metrics vs this baseline
- XGBoost handles class imbalance better via scale_pos_weight
- Should significantly improve precision without losing recall

# Week 2 Day 3 — XGBoost Training + Tuning

## Best Parameters Found
- n_estimators     : 50
- max_depth        : 2
- learning_rate    : 0.1
- scale_pos_weight : 2.77

## Tuning Method
- Cross validation (5-fold) on training data only
- Metric used for tuning: F1 score on churn class
- Tuned n_estimators first, then max_depth

## Actual Results vs Baseline
| Metric      | LR Baseline | XGBoost Tuned | Change      |
|-------------|-------------|---------------|-------------|
| Accuracy %  | 73.95       | 73.67         | ↓ 0.28      |
| F1 churn    | 0.6157      | 0.6195        | ↑ 0.0038    |
| F1 weighted | 0.7533      | 0.7511        | ↓ 0.0022    |
| ROC-AUC     | 0.8458      | 0.8415        | ↓ 0.0043    |
| TP caught   | 294         | 302            | ↑ 8           |
| FN missed   | 80          | 72             | ↓ 8            |
| FP alarms   | 287         | 299            |             |

## Classification Report
              precision  recall  f1-score  support
Retained         0.91    0.71      0.80     1035
Churned          0.50    0.81      0.62      374
accuracy                           0.74     1409
macro avg        0.71    0.76      0.71     1409
weighted avg     0.80    0.74      0.75     1409

## Honest Analysis of Results
XGBoost with default tuning did NOT significantly beat LR:
- F1 churn improved only slightly: 0.6157 → 0.6195 (+0.004)
- ROC-AUC actually dropped slightly: 0.8458 → 0.8415
- Accuracy almost identical: 73.95% → 73.67%
- Precision on churn still low at 0.50 (too many false alarms)

## Why This Happened
- n_estimators=50 and max_depth=2 is a very shallow model
- CV tuning selected simplest model to avoid overfitting
- scale_pos_weight alone is not enough for this imbalance
- This confirms we need SMOTE (Day 4) for real improvement

## Key Insight
Both LR and XGBoost have similar precision (0.51 vs 0.50)
and recall (0.79 vs 0.81) — the class imbalance is the
real bottleneck, not the model choice.
SMOTE should break this pattern tomorrow.

## Files Saved
- models/xgb_tuned.pkl
- models/xgb_best_params.json
- reports/model_comparison.png
- reports/roc_comparison.png
- reports/xgb_n_estimators_tuning.png
- reports/xgb_max_depth_tuning.png

## Tomorrow — Week 2 Day 4
- Apply SMOTE to training data to fix class imbalance
- Retrain XGBoost on balanced data
- Target: precision > 0.60 while keeping recall > 0.75
- Select and save the single best final model
