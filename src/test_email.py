# src/test_email.py
# Run with: python src/test_email.py

from email_generator import generate_retention_email
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test with a sample high-risk customer
test_features = [
    {'feature': 'contractRisk',   'shap_value': 0.45},
    {'feature': 'tenure',         'shap_value': 0.34},
    {'feature': 'OnlineSecurity', 'shap_value': 0.19},
]

print("Testing email generator...")
print("="*50)

email = generate_retention_email(
    churn_prob   = 82.3,
    risk_level   = "High",
    top_features = test_features
)

print(email)
print("="*50)
print("\nEmail generation working!")