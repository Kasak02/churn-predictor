<div align="center">

<img src="https://img.icons8.com/color/96/combo-chart--v1.png" width="80"/>

# SaaS Customer Churn Predictor & Intervention Engine

*An end-to-end AI system that predicts customer churn, explains every prediction,*
*and automatically generates personalised retention emails.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kasak02-churn-predictor.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange?logo=xgboost)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

**[🔗 Live Demo](https://kasak02-churn-predictor.streamlit.app)** &nbsp;|&nbsp;
**[📓 Notebooks](notebooks/)** &nbsp;|&nbsp;
**[📊 Model Card](reports/model_card.md)** &nbsp;|&nbsp;
**[📁 Reports](reports/)**

</div>

---

## 📌 Project Overview

Customer churn is one of the most critical metrics for any SaaS business.
Losing a paying customer costs 5–25× more than retaining one — yet most
companies discover churn only **after** it has already happened.

This project builds a complete, production-grade ML pipeline that:

- 🔮 **Predicts** which customers will churn in the next 30 days
- 🔍 **Explains** every prediction using SHAP (not just a black-box score)
- 📧 **Acts** by auto-generating personalised retention emails via Groq API + Llama 3.3
- 📈 **Monitors** churn patterns through an interactive Plotly dashboard

---

## 🖥️ Live Application

<div align="center">

| Tab | Description |
|-----|-------------|
| 🔮 **Predict** | Upload customer CSV → churn probability per customer → risk ranking |
| 🔍 **Explain** | SHAP waterfall plot → exactly why each customer got their score |
| 📧 **Email** | Groq API + Llama 3.3 → personalised B2B retention email in 3 seconds |
| 📈 **Dashboard** | Training analytics + live prediction analytics with Plotly |

**👉 [Try the live app here](https://kasak02-churn-predictor.streamlit.app)**

</div>

---

## 🏗️ System Architecture

```
Raw Customer Data (CSV)
        │
        ▼
┌─────────────────────┐
│  Data Pipeline      │  pandas · feature engineering · SMOTE
│  (Week 1)           │  7,043 customers · 20 features → 33 after OHE
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ML Model           │  XGBoost + SMOTE · sklearn pipeline
│  (Week 2)           │  ROC-AUC: 0.84 · F1 Churn: 0.62
└────────┬────────────┘
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────┐
│  SHAP Explainer     │              │  Groq API           │
│  TreeExplainer      │──────────▶  │  Llama 3.3 70b      │
│  Per-customer       │  top risk   │  Retention email    │
│  waterfall plots    │  factors    │  generation         │
└────────┬────────────┘              └─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit App      │  4 tabs · deployed on Streamlit Cloud
│  (Week 3)           │  FastAPI + Docker coming Week 4
└─────────────────────┘
```

---

## 🧪 Model Performance

### All Models Compared

| Model | Accuracy | F1 Churn | F1 Weighted | ROC-AUC |
|-------|----------|----------|-------------|---------|
| Logistic Regression (baseline) | 73.95% | 0.6157 | 0.7533 | 0.8458 |
| XGBoost Tuned | 73.67% | 0.6195 | 0.7511 | 0.8415 |
| **XGBoost + SMOTE (final ✓)** | **best** | **best** | **best** | **best** |

> **Primary metric: F1 on churn class** — accuracy is misleading due to
> 26.5% class imbalance. ROC-AUC of 0.84 is in line with published
> research benchmarks for this dataset (79–83% accuracy range).

### Confusion Matrix (Final Model)

```
                  Predicted Retained   Predicted Churned
Actual Retained        TN ✓                FP ✗
Actual Churned         FN ✗ (costly!)      TP ✓
```

---

## 🔍 SHAP Explainability

Top 5 global churn drivers identified by SHAP TreeExplainer:

| Rank | Feature | Mean SHAP | Business Insight |
|------|---------|-----------|-----------------|
| 1 | `contractRisk` | 0.6455 | Month-to-month = 42.7% churn rate |
| 2 | `tenure` | 0.3401 | New customers (0-12m) churn at ~47% |
| 3 | `InternetService_Fiber optic` | 0.2693 | Fibre customers churn more than DSL |
| 4 | `PaymentMethod_Electronic check` | 0.2091 | Electronic check correlates with churn |
| 5 | `PaperlessBilling` | 0.1738 | Paperless billing customers churn more |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data** | pandas · numpy · feature-engine | Cleaning, feature engineering |
| **ML** | XGBoost · scikit-learn · imbalanced-learn | Training, SMOTE, pipeline |
| **Explainability** | SHAP | Per-customer prediction explanations |
| **LLM** | Groq API · Llama 3.3 70b | Retention email generation |
| **App** | Streamlit · Plotly | Interactive web application |
| **API** | FastAPI · uvicorn | REST endpoints — [Live API] (https://churn-predictor-api-1re5.onrender.com) |
| **Deployment** | Streamlit Cloud · Docker · Render.com | Production deployment |
| **CI/CD** | GitHub Actions | Automated testing and deployment |

---

## 📁 Project Structure

```
churn-predictor/
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv              # Original dataset (never modified)
│   └── processed/
│       ├── telco_churn_clean.csv        # After cleaning
│       └── telco_churn_features.csv     # ML-ready with engineered features
│
|
├── 02_eda.ipynb                         # Exploratory data analysis
├── 03_data_cleaning.ipynb               # Data cleaning pipeline
├── 04_feature_engineering.ipynb         # Feature creation
├── 05_model_training.ipynb              # XGBoost + SHAP
│
├── models/
│   ├── final_model.pkl                  # XGBoost + SMOTE (best model)
│   ├── preprocessor.pkl                 # sklearn ColumnTransformer
│   ├── shap_explainer.pkl               # SHAP TreeExplainer
│   └── feature_names.json               # Feature names for SHAP
│
├── src/
│   ├── shap_utils.py                    # SHAP explanation helper
│   └── email_generator.py               # Groq API email generation
│
├── app/
│   └── streamlit_app.py                 # Complete 4-tab Streamlit app
│
├── reports/
│   ├── model_card.md                    # Full model documentation
│   ├── model_results.csv                # All model metrics
│   ├── shap_feature_importance.csv      # SHAP rankings
│   └── [charts and notes]              # EDA + SHAP visualisations
│
├── Dockerfile                           # Week 4
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.11+
- Free Groq API key from [console.groq.com](https://console.groq.com)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/Kasak02/churn-predictor.git
cd churn-predictor

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
echo GROQ_API_KEY= your_key_here > .env

# 5. Run the app
streamlit run app/streamlit_app.py
```

App opens at `http://localhost:8501`

---

## 📊 Dataset

| Field | Details |
|-------|---------|
| Source | [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers |
| Features | 21 raw → 33 after preprocessing |
| Target | Churn (Yes/No) — 26.5% churn rate |
| Split | 80% train / 20% test (stratified) |
| Imbalance fix | SMOTE on training data only |

---

## 📅 Development Roadmap

- [x] **Week 1** — EDA, data cleaning, feature engineering
- [x] **Week 2** — XGBoost training, SMOTE, SHAP explainability
- [x] **Week 3** — Streamlit app, Groq email generation, deployment
- [ ] **Week 4** — FastAPI REST API, Docker, GitHub Actions CI/CD

---


## 👩‍💻 About

**Kasak Bhatia **
B.Tech — Artificial Intelligence & Data Science (3rd Year)


*Targeting Data Scientist roles in IT Operations and SaaS domains*

[![GitHub](https://img.shields.io/badge/GitHub-Kasak02-black?logo=github)](https://github.com/Kasak02)

---

<div align="center">

*If you found this project useful, please ⭐ the repository!*

</div>