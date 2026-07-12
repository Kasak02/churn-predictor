

def explain_customer(customer_idx, X_test_proc, shap_values,
                      feature_names, y_pred_prob, explainer,
                      top_n=5):
    """
    Returns top N SHAP features for a given customer.
    Used by Streamlit app and LangChain email generator.
    """
    prob       = y_pred_prob[customer_idx]
    shap_vals  = shap_values[customer_idx]

    feat_shap  = list(zip(feature_names, shap_vals))
    feat_shap_sorted = sorted(feat_shap,
                               key=lambda x: abs(x[1]),
                               reverse=True)[:top_n]

    risk_level = "High"   if prob > 0.7 else \
                 "Medium" if prob > 0.4 else "Low"

    top_features = []
    for feat, shap_val in feat_shap_sorted:
        direction = "increases churn risk" if shap_val > 0 \
                    else "decreases churn risk"
        top_features.append({
            "feature"   : feat,
            "shap_value": round(float(shap_val), 4),
            "direction" : direction,
            "impact"    : "HIGH" if abs(shap_val) > 0.1 else "MEDIUM"
        })

    return {
        "churn_probability" : round(float(prob), 4),
        "churn_percent"     : round(float(prob)*100, 1),
        "risk_level"        : risk_level,
        "top_features"      : top_features
    }
