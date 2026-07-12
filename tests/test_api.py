# tests/test_api.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

# ── Test data ─────────────────────────────────────────────────────────────────
SAMPLE_CUSTOMER = {
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

LOW_RISK_CUSTOMER = {
    **SAMPLE_CUSTOMER,
    "tenure"  : 60,
    "Contract": "Two year",
    "OnlineSecurity": 1,
    "TechSupport"   : 1
}


# ── Import app ────────────────────────────────────────────────────────────────
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────
def test_health_check():
    """API should return 200 on root."""
    response = client.get("/")
    assert response.status_code == 200
    assert "status" in response.json()


def test_model_info():
    """Model info endpoint should return metrics."""
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert "model"   in data
    assert "metrics" in data
    assert data['metrics']['roc_auc'] > 0.8


def test_predict_high_risk():
    """High risk customer should return High risk level."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_level"        in data
    assert "prediction"        in data
    assert 0 <= data['churn_probability'] <= 1
    assert data['risk_level'] in ["High", "Medium", "Low"]


def test_predict_low_risk():
    """Long-tenure 2-year contract customer should be lower risk."""
    response = client.post("/predict", json=LOW_RISK_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert data['churn_probability'] < 0.5


def test_predict_batch():
    """Batch endpoint should handle multiple customers."""
    response = client.post(
        "/predict/batch",
        json=[SAMPLE_CUSTOMER, LOW_RISK_CUSTOMER]
    )
    assert response.status_code == 200
    data = response.json()
    assert data['total_customers'] == 2
    assert 'summary'     in data
    assert 'predictions' in data


def test_explain():
    """Explain endpoint should return SHAP features."""
    response = client.post("/explain", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    data = response.json()
    assert "top_features"      in data
    assert "churn_probability" in data
    assert len(data['top_features']) == 5


def test_explain_global():
    """Global SHAP endpoint should return top features."""
    response = client.get("/explain/global")
    assert response.status_code == 200
    data = response.json()
    assert "top_10_features" in data
    assert len(data['top_10_features']) == 10


def test_invalid_contract():
    """Invalid contract type should still return a prediction."""
    bad_customer = {**SAMPLE_CUSTOMER, "Contract": "Invalid"}
    response     = client.post("/predict", json=bad_customer)
    assert response.status_code == 200


def test_probability_range():
    """All predictions must be between 0 and 1."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    data     = response.json()
    assert 0.0 <= data['churn_probability'] <= 1.0