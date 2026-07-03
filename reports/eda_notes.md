
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
