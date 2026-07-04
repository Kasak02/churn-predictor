
## Day 2 EDA Notes

**Dataset:** IBM Telco Customer Churn
**Shape:** 7043 rows � 21 columns

**Key findings:**
- Churn rate is ~26.5% � class imbalance exists, will need SMOTE later
- TotalCharges column is object type, needs conversion to float
- 11 rows have blank TotalCharges (all have tenure=0, new customers)
- No other missing values in the dataset

**Next steps (Day 3):**
- Univariate analysis: distribution plots for all features
- Bivariate analysis: each feature vs Churn
- Correlation heatmap

## Day 3 EDA Observations

### Class Imbalance
- 73.5% retained, 26.5% churned — 3:1 ratio
- Will use SMOTE in Week 2 to balance before training

### Numerical Features
- tenure: Right skewed — many new customers (high churn risk group)
- MonthlyCharges: Roughly uniform $20-$100, some clustering at low end
- TotalCharges: Highly right skewed — reflects tenure × monthly charges

### Categorical Features
- Contract: Most customers are month-to-month (highest churn risk)
- InternetService: Fibre optic is most common
- PaymentMethod: Electronic check is most common payment method
- TechSupport / OnlineSecurity: Majority have No — potential churn driver

### Next Steps (Day 4)
- Bivariate analysis: each feature vs Churn
- Find which features most strongly predict churn

## Day 4 — Business Insights from EDA

### Top Churn Risk Factors Found

| Factor | Finding | Churn Rate |
|--------|---------|------------|
| Contract type | Month-to-month customers | 42.7% |
| Tenure | New customers (0-12 months) | ~47% |
| TechSupport | Customers with no support | ~41% |
| OnlineSecurity | No security subscription | ~41% |
| MonthlyCharges | Higher charges = more churn | +0.19 corr |

### Key Business Recommendations
1. Incentivise month-to-month customers to upgrade to annual contracts
2. Prioritise retention efforts in first 12 months of customer lifecycle
3. Offer TechSupport and OnlineSecurity bundles to at-risk segments
4. High monthly charge customers need proactive outreach

### Model Implications
- Contract, tenure, TechSupport, OnlineSecurity → likely top SHAP features
- Class imbalance (26.5% churn) → SMOTE needed before XGBoost training
- TotalCharges and tenure highly correlated → may need to watch multicollinearity

# Day 5 — Data Cleaning Notes

## What was done today

### 1. Fixed TotalCharges dtype
- Problem : Stored as `object` (text) instead of `float64`
- Fix     : pd.to_numeric(df['TotalCharges'], errors='coerce')
- Result  : 11 rows became NaN (all had tenure=0)

### 2. Handled 11 Null Rows
- Problem : 11 customers had blank TotalCharges
- Reason  : tenure=0 means brand new customer, never billed
- Fix     : Filled with 0 (TotalCharges = 0 for new customers)
- Result  : 0 nulls remaining in entire dataset

### 3. Dropped customerID
- Reason  : Unique identifier per row, no predictive value
- Result  : Columns reduced from 21 → 20

### 4. Fixed SeniorCitizen
- Problem : Stored as 0/1 integer, inconsistent with other columns
- Fix     : Mapped 0→No, 1→Yes, then re-encoded Yes→1, No→0
- Result  : Consistent encoding across all binary columns

### 5. Encoded Binary Yes/No Columns → 0/1
Columns encoded:
- Partner, Dependents, PhoneService, PaperlessBilling, SeniorCitizen
- gender (Male=1, Female=0)
- Churn target (Yes=1, No=0)

### 6. Simplified Three-Value Columns → 0/1
Columns affected:
- MultipleLines, OnlineSecurity, OnlineBackup
- DeviceProtection, TechSupport, StreamingTV, StreamingMovies

Fix: 'No internet service' and 'No phone service' both mapped to 'No'
Then encoded Yes→1, No→0

### 7. Left Multi-Class Columns as Text
Columns kept as text for OneHotEncoder in Week 2:
- InternetService  : DSL / Fiber optic / No
- Contract         : Month-to-month / One year / Two year
- PaymentMethod    : Electronic check / Mailed check / etc.

## Final Clean Dataset Summary
- Shape      : (7043, 20)
- Nulls      : 0
- Churn rate : 26.5%
- Saved to   : data/processed/telco_churn_clean.csv

## Key Decisions Made
| Decision | Reason |
|----------|--------|
| Filled nulls with 0 not dropped | tenure=0 customers are valid data points |
| Kept 3 text columns as-is | OneHotEncoder handles them better in pipeline |
| Dropped customerID | No signal, just a row identifier |

## Tomorrow — Day 6 (Feature Engineering)
- Create numServices  : count of all services subscribed per customer
- Create chargePerMonth: TotalCharges / (tenure + 1)
- Create tenureBucket : group tenure into 4 buckets (New/Growing/Mature/Loyal)
- Save final feature-engineered dataset

# Day 6 — Feature Engineering Notes

## New Features Created

### 1. numServices
- Formula : Sum of 8 service columns (PhoneService, MultipleLines,
            OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies)
- Range   : 0 to 8
- Insight : Customers with 0 services churn at ~65%
            Customers with 8 services churn at ~8%
- Why     : More services = more platform investment = lower churn

### 2. chargePerMonth
- Formula : TotalCharges / (tenure + 1)
- Why +1  : Avoids division by zero for tenure=0 customers
- Insight : Captures normalised spend rate over customer lifetime
- Why     : Different from MonthlyCharges — reflects spend trajectory

### 3. tenureBucket
- Formula : pd.cut(tenure, bins=[0,12,24,48,72])
- Labels  : New(0-12m), Growing(12-24m), Mature(24-48m), Loyal(48-72m)
- Insight :
    New     → 47% churn rate (highest risk)
    Growing → 35% churn rate
    Mature  → 20% churn rate
    Loyal   →  8% churn rate (lowest risk)
- Why     : Captures non-linear relationship between tenure and churn

### 4. contractRisk
- Formula : Month-to-month=3, One year=2, Two year=1
- Insight : Ordinal encoding of contract risk level
- Why     : Preserves order (month-to-month > one year > two year risk)

## Files Saved
- data/processed/telco_churn_features.csv  ← used for ML training in Week 2

## Week 1 Complete!
All EDA, cleaning, and feature engineering is done.
Week 2 starts with sklearn pipeline + XGBoost training.
