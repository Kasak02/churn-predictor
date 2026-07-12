# api/main.py
# FastAPI REST API for Churn Predictor
# Run with: uvicorn api.main:app --reload

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SaaS Churn Predictor API",
    description = """
    End-to-end churn prediction API built with XGBoost + SHAP.

    ## Endpoints
    - **GET /**         — Health check
    - **GET /info**     — Model information and metrics
    - **POST /predict** — Predict churn probability for one customer
    - **POST /predict/batch** — Predict churn for multiple customers
    - **POST /explain** — Get top SHAP features for a customer
    """,
    version     = "1.0.0",
    contact     = {
        "name" : "Kasak",
        "url"  : "https://github.com/Kasak02/churn-predictor"
    }
)

# ── CORS — allows browser apps to call this API ───────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"]
)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS — once at startup
# ══════════════════════════════════════════════════════════════════════════════
model         = None
preprocessor  = None
explainer     = None
feature_names = []
model_results = []

try:
    model        = joblib.load('models/final_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')
    explainer    = joblib.load('models/shap_explainer.pkl')

    with open('models/feature_names.json') as f:
        feature_names = json.load(f)

    with open('reports/model_results.csv') as f:
        import csv
        reader        = csv.DictReader(f)
        model_results = list(reader)

    print("✓ All models loaded successfully")

except FileNotFoundError as e:
    print(f"⚠ Model files not found: {e}")
    print("  Tests will run in mock mode")

def check_models_loaded():
    """Raise error if models not loaded."""
    if model is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Models not loaded. Run from project root."
        )

# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class CustomerData(BaseModel):
    """Single customer data for prediction."""

    # Core features
    tenure          : float = Field(..., ge=0, le=72,
                                    description="Months as customer (0-72)")
    MonthlyCharges  : float = Field(..., ge=0,
                                    description="Monthly bill in USD")
    TotalCharges    : float = Field(0.0, ge=0,
                                    description="Total charges paid")

    # Contract and service
    Contract        : str   = Field(...,
                                    description="Month-to-month / One year / Two year")
    InternetService : str   = Field("DSL",
                                    description="DSL / Fiber optic / No")
    PaymentMethod   : str   = Field("Electronic check",
                                    description="Payment method")

    # Binary features (0 or 1)
    gender          : int   = Field(0, ge=0, le=1,
                                    description="0=Female, 1=Male")
    SeniorCitizen   : int   = Field(0, ge=0, le=1,
                                    description="0=No, 1=Yes")
    Partner         : int   = Field(0, ge=0, le=1)
    Dependents      : int   = Field(0, ge=0, le=1)
    PhoneService    : int   = Field(1, ge=0, le=1)
    MultipleLines   : int   = Field(0, ge=0, le=1)
    OnlineSecurity  : int   = Field(0, ge=0, le=1)
    OnlineBackup    : int   = Field(0, ge=0, le=1)
    DeviceProtection: int   = Field(0, ge=0, le=1)
    TechSupport     : int   = Field(0, ge=0, le=1)
    StreamingTV     : int   = Field(0, ge=0, le=1)
    StreamingMovies : int   = Field(0, ge=0, le=1)
    PaperlessBilling: int   = Field(0, ge=0, le=1)

    model_config ={
        "json_schema_extra" : {
            "example": {
                "tenure"          : 3,
                "MonthlyCharges"  : 89.10,
                "TotalCharges"    : 267.30,
                "Contract"        : "Month-to-month",
                "InternetService" : "Fiber optic",
                "PaymentMethod"   : "Electronic check",
                "gender"          : 0,
                "SeniorCitizen"   : 0,
                "Partner"         : 0,
                "Dependents"      : 0,
                "PhoneService"    : 1,
                "MultipleLines"   : 0,
                "OnlineSecurity"  : 0,
                "OnlineBackup"    : 0,
                "DeviceProtection": 0,
                "TechSupport"     : 0,
                "StreamingTV"     : 0,
                "StreamingMovies" : 0,
                "PaperlessBilling": 1
            }
        }
            }

class PredictionResponse(BaseModel):
    """Response from /predict endpoint."""
    churn_probability : float
    churn_percent     : str
    risk_level        : str
    prediction        : str
    confidence        : str


class ExplainResponse(BaseModel):
    """Response from /explain endpoint."""
    churn_probability : float
    risk_level        : str
    top_features      : list
    summary           : str


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features — same as Week 1."""
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']

    existing_svc       = [c for c in service_cols if c in df.columns]
    df['numServices']  = df[existing_svc].sum(axis=1)
    df['chargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['tenureBucket'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['New', 'Growing', 'Mature', 'Loyal'],
        include_lowest=True
    ).astype(str)
    df['contractRisk'] = df['Contract'].map({
        'Month-to-month': 3,
        'One year'      : 2,
        'Two year'      : 1
    }).fillna(2).astype(int)

    return df


def get_risk_level(prob: float) -> str:
    if prob >= 0.7:  return "High"
    elif prob >= 0.4: return "Medium"
    else:             return "Low"


def preprocess_customer(customer: CustomerData) -> np.ndarray:
    """Convert CustomerData to preprocessed numpy array."""
    df = pd.DataFrame([customer.model_dump()])
    df = engineer_features(df)
    return preprocessor.transform(df)


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── GET / — Health check ──────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Health check — confirms API is running."""
    return {
        "status" : "✓ API is running",
        "name"   : "SaaS Churn Predictor API",
        "version": "1.0.0",
        "docs"   : "/docs"
    }


# ── GET /info — Model information ─────────────────────────────────────────────
@app.get("/info", tags=["Model"])
def model_info():
    """Returns model metadata and performance metrics."""
    return {
        "model"   : "XGBoost + SMOTE",
        "version" : "1.0.0",
        "dataset" : {
            "name"    : "IBM Telco Customer Churn",
            "rows"    : 7043,
            "features": len(feature_names)
        },
        "metrics" : {
            "roc_auc"    : 0.8374,
            "f1_churn"   : 0.6195,
            "f1_weighted": 0.7511,
            "accuracy"   : 73.67
        },
        "top_features": feature_names[:5],
        "endpoints": ["/predict", "/predict/batch", "/explain"]
    }


# ── POST /predict — Single customer prediction ────────────────────────────────
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerData):
    check_models_loaded() 
    """
    Predict churn probability for a single customer.

    Returns churn probability, risk level and prediction.
    """
    try:
        X_proc = preprocess_customer(customer)
        prob   = float(model.predict_proba(X_proc)[0][1])
        pred   = int(model.predict(X_proc)[0])
        risk   = get_risk_level(prob)

        return PredictionResponse(
            churn_probability = round(prob, 4),
            churn_percent     = f"{prob*100:.1f}%",
            risk_level        = risk,
            prediction        = "Churn" if pred == 1 else "Retain",
            confidence        = (
                "High confidence"   if prob > 0.8 or prob < 0.2 else
                "Medium confidence" if prob > 0.6 or prob < 0.4 else
                "Low confidence — borderline case"
            )
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/batch — Multiple customers ──────────────────────────────────
@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(customers: list[CustomerData]):
    check_models_loaded()
    """
    Predict churn for multiple customers at once.

    Returns list of predictions sorted by churn risk (highest first).
    """
    if len(customers) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 customers per batch request"
        )

    try:
        results = []
        for i, customer in enumerate(customers):
            X_proc = preprocess_customer(customer)
            prob   = float(model.predict_proba(X_proc)[0][1])
            pred   = int(model.predict(X_proc)[0])
            risk   = get_risk_level(prob)

            results.append({
                "customer_index"   : i,
                "churn_probability": round(prob, 4),
                "churn_percent"    : f"{prob*100:.1f}%",
                "risk_level"       : risk,
                "prediction"       : "Churn" if pred == 1 else "Retain"
            })

        # Sort by churn probability descending
        results.sort(key=lambda x: x['churn_probability'], reverse=True)

        # Summary stats
        probs   = [r['churn_probability'] for r in results]
        n_high  = sum(1 for p in probs if p >= 0.7)
        n_med   = sum(1 for p in probs if 0.4 <= p < 0.7)
        n_low   = sum(1 for p in probs if p < 0.4)

        return {
            "total_customers": len(customers),
            "summary": {
                "high_risk"  : n_high,
                "medium_risk": n_med,
                "low_risk"   : n_low,
                "avg_churn_probability": round(np.mean(probs), 4)
            },
            "predictions": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ── POST /explain — SHAP explanation ─────────────────────────────────────────
@app.post("/explain", response_model=ExplainResponse, tags=["Explanation"])
def explain(customer: CustomerData):
    check_models_loaded()
    """
    Returns top 5 SHAP features explaining the churn prediction.
    Shows which features pushed the score up or down.
    """
    try:
        X_proc = preprocess_customer(customer)
        prob   = float(model.predict_proba(X_proc)[0][1])
        risk   = get_risk_level(prob)

        # Compute SHAP values
        shap_vals = explainer.shap_values(X_proc)[0]

        # Get top 5 features by absolute SHAP value
        feat_shap = sorted(
            zip(feature_names, shap_vals),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        top_features = [
            {
                "rank"     : i + 1,
                "feature"  : feat,
                "shap_value"   : round(float(val), 4),
                "direction": "increases churn risk" if val > 0
                              else "decreases churn risk",
                "impact"   : "HIGH" if abs(val) > 0.2
                              else "MEDIUM" if abs(val) > 0.1
                              else "LOW"
            }
            for i, (feat, val) in enumerate(feat_shap)
        ]

        # Auto-generate summary sentence
        top_positive = [
            f['feature'].replace('_', ' ')
            for f in top_features if f['shap_value'] > 0
        ][:2]

        summary = (
            f"This customer has a {prob*100:.1f}% churn probability. "
            f"Key risk factors: {', '.join(top_positive)}."
            if top_positive else
            f"This customer has a {prob*100:.1f}% churn probability "
            f"with low overall risk."
        )

        return ExplainResponse(
            churn_probability = round(prob, 4),
            risk_level        = risk,
            top_features      = top_features,
            summary           = summary
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /explain/global — Global SHAP importance ──────────────────────────────
@app.get("/explain/global", tags=["Explanation"])
def explain_global():
    """
    Returns global feature importance from SHAP
    across all training data.
    """
    try:
        import csv
        with open('reports/shap_feature_importance.csv') as f:
            reader   = csv.DictReader(f)
            features = list(reader)

        return {
            "model"          : "XGBoost + SMOTE",
            "total_features" : len(features),
            "top_10_features": [
                {
                    "rank"      : i + 1,
                    "feature"   : row['Feature'],
                    "mean_shap" : round(float(row['Mean_SHAP']), 4)
                }
                for i, row in enumerate(features[:10])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))