<div align="center">

<img src="https://img.icons8.com/color/96/combo-chart--v1.png" width="80"/>

# SaaS Customer Churn Predictor & Intervention Engine

*An end-to-end AI system that predicts customer churn, explains every prediction,*
*and automatically generates personalised retention emails.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churn-predictor-4j4exkmmyywrhbulwqsuuu.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/FastAPI-0.13x-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black?logo=github)
![License](https://img.shields.io/badge/License-MIT-green)

**[🔗 Live App](https://churn-predictor-4j4exkmmyywrhbulwqsuuu.streamlit.app/)** &nbsp;|&nbsp;
**[⚡ Live API](https://churn-predictor-api-1re5.onrender.com)** &nbsp;|&nbsp;
**[📖 API Docs](https://churn-predictor-api-1re5.onrender.com/docs)** &nbsp;|&nbsp;
**[📁 GitHub](https://github.com/Kasak02/churn-predictor)**

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
- ⚡ **Serves** predictions via a FastAPI REST API containerised with Docker

---

## 🖥️ Live Deployments

| Service | URL | Description |
|---------|-----|-------------|
| 🔗 Streamlit App | [churn-predictor.streamlit.app](https://churn-predictor-4j4exkmmyywrhbulwqsuuu.streamlit.app/) | Interactive 4-tab web application |
| ⚡ FastAPI | [churn-predictor-api.onrender.com](https://churn-predictor-api-1re5.onrender.com) | REST API health check |
| 📖 API Docs | [/docs](https://churn-predictor-api-1re5.onrender.com/docs) | Interactive Swagger UI |
| 📁 GitHub | [Kasak02/churn-predictor](https://github.com/Kasak02/churn-predictor) | Source code + notebooks |

> **Note:** Render.com free tier sleeps after 15 min inactivity.
> First request may take ~30 seconds to wake up.

---

## 🖥️ Application Tabs

| Tab | Description |
|-----|-------------|
| 🔮 **Predict** | Upload customer CSV → churn probability per customer → risk ranking |
| 🔍 **Explain** | SHAP waterfall plot → exactly why each customer got their score |
| 📧 **Email** | Groq API + Llama 3.3 → personalised B2B retention email in ~3 seconds |
| 📈 **Dashboard** | Training analytics + live prediction analytics with Plotly |

---

## 🏗️ System Architecture

```
Raw Customer Data (CSV)
        │
        ▼
┌─────────────────────┐
│  Data Pipeline      │  pandas · feature engineering · SMOTE
│                     │  7,043 customers · 20 features → 33 after OHE
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ML Model           │  XGBoost + SMOTE · sklearn pipeline
│                     │  ROC-AUC: 0.84 · F1 Churn: 0.62
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
┌──────────────────────────────────────────────────────────┐
│  Streamlit App (4 tabs) + FastAPI REST API               │
│  Deployed: Streamlit Cloud + Docker + Render.com         │
│  CI/CD: GitHub Actions (lint + pytest + auto-deploy)     │
└──────────────────────────────────────────────────────────┘
```

---

## ⚡ API Endpoints

Base URL: `https://churn-predictor-api-1re5.onrender.com`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/info` | Model metadata and metrics |
| POST | `/predict` | Single customer churn prediction |
| POST | `/predict/batch` | Batch prediction (up to 1,000 customers) |
| POST | `/explain` | Top 5 SHAP features per customer |
| GET | `/explain/global` | Global feature importance |

**Quick test:**
```bash
curl -X POST "https://churn-predictor-api-1re5.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 3, "MonthlyCharges": 89.10, "TotalCharges": 267.30,
       "Contract": "Month-to-month", "InternetService": "Fiber optic",
       "PaymentMethod": "Electronic check", "gender": 0, "SeniorCitizen": 0,
       "Partner": 0, "Dependents": 0, "PhoneService": 1, "MultipleLines": 0,
       "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
       "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0,
       "PaperlessBilling": 1}'
```

---

## 🧪 Model Performance

| Model | Accuracy | F1 Churn | F1 Weighted | ROC-AUC |
|-------|----------|----------|-------------|---------|
| Logistic Regression (baseline) | 73.95% | 0.6157 | 0.7533 | 0.8458 |
| XGBoost Tuned | 73.67% | 0.6195 | 0.7511 | 0.8415 |
| **XGBoost + SMOTE (final ✓)** | **best** | **best** | **best** | **best** |

> **Primary metric: F1 on churn class** — accuracy is misleading due to
> 26.5% class imbalance. ROC-AUC of 0.84 is consistent with published
> research benchmarks for this dataset.

---

## 🔍 SHAP Explainability

Top 5 global churn drivers:

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
| **API** | FastAPI · uvicorn · Pydantic | REST endpoints with auto docs |
| **Container** | Docker · docker-compose | Containerisation |
| **Deployment** | Streamlit Cloud · Render.com | Production deployment |
| **CI/CD** | GitHub Actions · pytest · flake8 | Automated testing + deployment |

---

## 📁 Project Structure

```
churn-predictor/
│
├── 02_eda.ipynb                     # Exploratory data analysis
├── 03_data_cleaning.ipynb           # Data cleaning pipeline
├── 04_feature_engineering.ipynb     # Feature creation + engineering
├── 05_model_training.ipynb          # XGBoost + SHAP training
│
├── data/
│   ├── raw/
│   │   └── telco_churn.csv          # Original dataset (never modified)
│   └── processed/
│       ├── telco_churn_clean.csv    # After cleaning
│       └── telco_churn_features.csv # ML-ready with engineered features
│
├── models/
│   ├── final_model.pkl              # XGBoost + SMOTE (best model)
│   ├── preprocessor.pkl             # sklearn ColumnTransformer
│   ├── shap_explainer.pkl           # SHAP TreeExplainer
│   └── feature_names.json           # Feature names for SHAP
│
├── src/
│   ├── shap_utils.py                # SHAP explanation helper
│   └── email_generator.py           # Groq API email generation
│
├── app/
│   └── streamlit_app.py             # Complete 4-tab Streamlit app
│
├── api/
│   └── main.py                      # FastAPI endpoints
│
├── tests/
│   └── test_api.py                  # 9 pytest unit tests
│
├── reports/
│   ├── model_card.md                # Full model documentation
│   ├── model_results.csv            # All model metrics
│   └── shap_feature_importance.csv  # SHAP rankings
│
├── .github/
│   └── workflows/
│       └── ci.yml                   # GitHub Actions CI/CD
│
├── Dockerfile
├── docker-compose.yml
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
# 1. Clone
git clone https://github.com/Kasak02/churn-predictor.git
cd churn-predictor

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add Groq API key
echo GROQ_API_KEY=your_key_here > .env

# 5. Run Streamlit app
streamlit run app/streamlit_app.py

# 6. Run FastAPI (separate terminal)
uvicorn api.main:app --reload

# 7. Run with Docker
docker build -t churn-predictor .
docker run -p 8000:8000 churn-predictor

# 8. Run tests
python -m pytest tests/ -v
```

---

## 📊 Dataset

| Field | Details |
|-------|---------|
| Source | [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers |
| Features | 21 raw → 33 after preprocessing |
| Target | Churn (Yes/No) — 26.5% churn rate |
| Split | 80% train / 20% test (stratified) |
| Imbalance fix | SMOTE on training data only |

---

## 📅 Development Roadmap

- [x] **Week 1** — EDA, data cleaning, feature engineering
- [x] **Week 2** — XGBoost training, SMOTE, SHAP explainability
- [x] **Week 3** — Streamlit app, Groq email generation, Streamlit Cloud deployment
- [x] **Week 4** — FastAPI REST API, Docker, Render.com, GitHub Actions CI/CD

---

## 👩‍💻 About

**Kasak Bhatia **
B.Tech — Artificial Intelligence & Data Science Student

[![GitHub](https://img.shields.io/badge/GitHub-Kasak02-black?logo=github)](https://github.com/Kasak02)

---

<div align="center">

*If you found this project useful, please ⭐ the repository!*

</div>