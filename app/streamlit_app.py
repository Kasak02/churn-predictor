# app/streamlit_app.py
# Run with: streamlit run app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page config — must be FIRST ──────────────────────────────────────────────
st.set_page_config(
    page_title            = "SaaS Churn Predictor",
    page_icon             = "📊",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem; font-weight: 700;
    color: #534AB7; margin-bottom: 0;
}
.sub-header {
    font-size: 1rem; color: #5F5E5A;
    margin-top: 0; margin-bottom: 1.5rem;
}
.risk-high   { color: #E05A2B; font-weight: 700; }
.risk-medium { color: #F0934A; font-weight: 700; }
.risk-low    { color: #0F6E56; font-weight: 700; }
.stTabs [data-baseweb="tab"] {
    background-color: #F1EFE8;
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: #EEEDFE !important;
    color: #534AB7 !important;
}
div[data-testid="metric-container"] {
    background-color: #F8F7F4;
    border: 0.5px solid #D3D1C7;
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    try:
        model        = joblib.load('models/final_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        explainer    = joblib.load('models/shap_explainer.pkl')
        with open('models/feature_names.json') as f:
            feature_names = json.load(f)
        return model, preprocessor, explainer, feature_names
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Run the app from the project root: churn-predictor/")
        st.stop()


@st.cache_data
def load_metrics():
    try:
        results_df  = pd.read_csv('reports/model_results.csv')
        shap_df     = pd.read_csv('reports/shap_feature_importance.csv')
        train_df    = pd.read_csv('data/processed/telco_churn_features.csv')

        with open('models/xgb_best_params.json') as f:
            best_params = json.load(f)
        with open('models/feature_names.json') as f:
            feature_names = json.load(f)

        best_row = results_df.loc[results_df['F1_Churn'].idxmax()]
        lr_row   = results_df[results_df['Model'] == 'Logistic Regression'].iloc[0]

        return {
            'results_df'   : results_df,
            'shap_df'      : shap_df,
            'train_df'     : train_df,
            'best_params'  : best_params,
            'feature_names': feature_names,
            'best_row'     : best_row,
            'lr_row'       : lr_row,
            'model_name'   : best_row['Model'],
            'f1_churn'     : round(float(best_row['F1_Churn']),  4),
            'roc_auc'      : round(float(best_row['ROC_AUC']),   4),
            'accuracy'     : round(float(best_row['Accuracy']),  2),
            'f1_weighted'  : round(float(best_row['F1_Weighted']),4),
            'tp'           : int(best_row['TP']),
            'fn'           : int(best_row['FN']),
            'fp'           : int(best_row['FP']),
            'tn'           : int(best_row['TN']),
            'lr_f1'        : round(float(lr_row['F1_Churn']), 4),
            'lr_roc'       : round(float(lr_row['ROC_AUC']),  4),
            'n_features'   : len(feature_names),
            'total_size'   : len(train_df),
            'train_size'   : int(len(train_df) * 0.8),
            'churn_rate'   : round(train_df['Churn'].mean() * 100, 1),
            'top5_features': shap_df.head(5)['Feature'].tolist(),
            'n_estimators' : best_params['n_estimators'],
            'max_depth'    : best_params['max_depth'],
        }
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        st.stop()


# ── Helper: risk badge ────────────────────────────────────────────────────────
def risk_badge(prob):
    if prob >= 0.7:
        return "🔴 High"
    elif prob >= 0.4:
        return "🟡 Medium"
    else:
        return "🟢 Low"


# ── Helper: preprocess uploaded data ─────────────────────────────────────────
def preprocess_and_predict(df_raw, preprocessor, model):
    """
    Handles BOTH raw Kaggle CSV and cleaned feature CSV.
    Applies all Week 1 cleaning steps before prediction.
    """
    df = df_raw.copy()

    # ── Step 1: Save customerID before dropping ───────────────────────────
    customer_ids = None
    if 'customerID' in df.columns:
        customer_ids = df['customerID'].values
        df = df.drop(columns=['customerID'])

    # ── Step 2: Drop target if present ───────────────────────────────────
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])

    # ── Step 3: Fix TotalCharges — blank strings → 0 ─────────────────────
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(
            df['TotalCharges'], errors='coerce'
        ).fillna(0)

    # ── Step 4: Fix SeniorCitizen — 0/1 int → Yes/No → back to 0/1 ───────
    if 'SeniorCitizen' in df.columns:
        if df['SeniorCitizen'].dtype in ['int64', 'float64']:
            df['SeniorCitizen'] = df['SeniorCitizen'].map(
                {0: 0, 1: 1}
            ).fillna(0).astype(int)

    # ── Step 5: Encode binary Yes/No columns → 0/1 ───────────────────────
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({
                'nan': 'No', 'NaN': 'No',
                '0.0': 'No', '1.0': 'Yes',
                '0'  : 'No', '1'  : 'Yes'
            })
            df[col] = df[col].map(
                {'Yes': 1, 'No': 0}
            ).fillna(0).astype(int)

    # ── Step 6: Encode gender ─────────────────────────────────────────────
    if 'gender' in df.columns and df['gender'].dtype == object:
        df['gender'] = (df['gender'] == 'Male').astype(int)

    # ── Step 7: Fix three-value columns ──────────────────────────────────
    three_val_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                      'DeviceProtection', 'TechSupport',
                      'StreamingTV', 'StreamingMovies']
    for col in three_val_cols:
        if col in df.columns:
            # Force to string first to handle any mixed types
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({
                'No internet service': 'No',
                'No phone service'   : 'No',
                'nan'                : 'No',
                'NaN'                : 'No',
                '0'                  : 'No',
                '1'                  : 'Yes',
                '0.0'                : 'No',
                '1.0'                : 'Yes'
            })
            df[col] = df[col].map(
                {'Yes': 1, 'No': 0}
            ).fillna(0).astype(int)

    # ── Step 8: Engineered features ───────────────────────────────────────
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']

    if 'numServices' not in df.columns:
        existing_svc   = [c for c in service_cols if c in df.columns]
        df['numServices'] = df[existing_svc].sum(axis=1) \
                            if existing_svc else 0

    if 'chargePerMonth' not in df.columns:
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['chargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
        else:
            df['chargePerMonth'] = 0

    if 'tenureBucket' not in df.columns:
        if 'tenure' in df.columns:
            df['tenureBucket'] = pd.cut(
                df['tenure'],
                bins=[0, 12, 24, 48, 72],
                labels=['New', 'Growing', 'Mature', 'Loyal'],
                include_lowest=True
            ).astype(str)
        else:
            df['tenureBucket'] = 'New'

    if 'contractRisk' not in df.columns:
        if 'Contract' in df.columns:
            df['contractRisk'] = df['Contract'].map({
                'Month-to-month': 3,
                'One year'      : 2,
                'Two year'      : 1
            }).fillna(2).astype(int)
        else:
            df['contractRisk'] = 2

    # ── Step 9: Final safety clean ────────────────────────────────────────
    # Catch any remaining string columns that slipped through
    # This handles edge cases in raw Kaggle CSV

    # Force-clean any column that is still object type
    for col in df.select_dtypes(include='object').columns:
        # Skip columns meant for OneHotEncoder
        ohe_cols = ['InternetService', 'Contract', 'PaymentMethod',
                    'tenureBucket']
        if col in ohe_cols:
            continue

        # Try Yes/No mapping first
        if df[col].isin(['Yes', 'No', 'No phone service',
                         'No internet service']).any():
            df[col] = df[col].replace({
                'No phone service'   : 'No',
                'No internet service': 'No'
            })
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)

        # Try Male/Female mapping
        elif df[col].isin(['Male', 'Female']).any():
            df[col] = (df[col] == 'Male').astype(int)

        # Any other string column — force to numeric, fill errors with 0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ── Step 10: Debug check (remove after testing) ───────────────────────
    remaining_strings = df.select_dtypes(include='object').columns.tolist()
    ohe_expected      = ['InternetService', 'Contract',
                         'PaymentMethod', 'tenureBucket']
    unexpected        = [c for c in remaining_strings
                         if c not in ohe_expected]
    if unexpected:
        st.warning(f"⚠️ These columns still have strings: {unexpected}")

    # ── Step 11: Predict ──────────────────────────────────────────────────
    X_proc = preprocessor.transform(df)
    probs  = model.predict_proba(X_proc)[:, 1]
    preds  = model.predict(X_proc)

    return X_proc, probs, preds, customer_ids


# ── Helper: SHAP waterfall for one customer ───────────────────────────────────
def plot_shap_waterfall(explainer, X_proc_row, feature_names, prob):
    shap_vals   = explainer.shap_values(X_proc_row.reshape(1, -1))[0]
    explanation = shap.Explanation(
        values        = shap_vals,
        base_values   = explainer.expected_value,
        data          = X_proc_row,
        feature_names = feature_names
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f'SHAP Explanation — Churn Probability: {prob*100:.1f}%',
              fontweight='bold', fontsize=12, pad=12)
    plt.tight_layout()
    return fig


# ── Load everything ───────────────────────────────────────────────────────────
# Load .env for API keys
from dotenv import load_dotenv
load_dotenv()
model, preprocessor, explainer, feature_names = load_models()
m = load_metrics()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Churn Predictor")
    st.markdown("*AI-powered customer retention*")
    st.divider()

    st.markdown("### Model Summary")
    st.markdown(f"""
    | Field | Value |
    |-------|-------|
    | Model | {m['model_name']} |
    | F1 Churn | {m['f1_churn']} |
    | ROC-AUC | {m['roc_auc']} |
    | Accuracy | {m['accuracy']}% |
    | Features | {m['n_features']} |
    | n_estimators | {m['n_estimators']} |
    | max_depth | {m['max_depth']} |
    """)

    st.divider()
    st.markdown("### Dataset")
    st.markdown(f"""
    - **Total customers** : {m['total_size']:,}
    - **Training set**    : {m['train_size']:,}
    - **Churn rate**      : {m['churn_rate']}%
    """)

    st.divider()
    st.markdown("### Top Churn Drivers")
    for i, feat in enumerate(m['top5_features'], 1):
        st.markdown(f"**{i}.** `{feat}`")

    st.divider()
    st.markdown("### Tech Stack")
    st.markdown("`XGBoost` `SHAP` `Streamlit`\n`Plotly` `Groq API` `Llama 3.3`\n`pandas` `scikit-learn`")
    st.divider()
    st.markdown("Built by **Kasak** · VIPS-TC")
    st.markdown("[GitHub ↗](https://github.com/Kasak02/churn-predictor)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-header">📊 SaaS Customer Churn Predictor</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">'
    'Predict churn · Explain predictions with SHAP · '
    'Generate AI retention emails · Explore analytics'
    '</p>',
    unsafe_allow_html=True
)

# ── Top KPI row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
roc_delta = round(m['roc_auc'] - m['lr_roc'], 4)
f1_delta  = round(m['f1_churn'] - m['lr_f1'], 4)

with c1:
    st.metric("Best Model",     m['model_name'],
              delta=f"Accuracy {m['accuracy']}%")
with c2:
    st.metric("ROC-AUC",        str(m['roc_auc']),
              delta=f"{roc_delta:+.4f} vs LR")
with c3:
    st.metric("F1 Churn",       str(m['f1_churn']),
              delta=f"{f1_delta:+.4f} vs LR")
with c4:
    st.metric("Training Data",  f"{m['train_size']:,} customers",
              delta=f"{m['churn_rate']}% churn rate")

st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Predict",
    "🔍  Explain",
    "📧  Email",
    "📈  Dashboard"
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload Customer Data")
    st.markdown(
        f"Upload a CSV of customers → model scores each one with a "
        f"churn probability using **{m['n_features']} features**."
    )

    # ── Sample CSV download ───────────────────────────────────────────────
    sample_df  = m['train_df'].drop(columns=['Churn']).head(20)
    sample_csv = sample_df.to_csv(index=False)

    col_up, col_dl = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Must have same columns as training data"
        )
    with col_dl:
        st.markdown("#### Don't have a file?")
        st.markdown(f"Download **{len(sample_df)} sample customers** to test:")
        st.download_button(
            label     = "⬇️ Download sample CSV",
            data      = sample_csv,
            file_name = "sample_customers.csv",
            mime      = "text/csv"
        )

    # ── Predictions ───────────────────────────────────────────────────────
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        st.success(
            f"✓ {len(df_raw)} customers loaded · "
            f"{len(df_raw.columns)} columns detected"
        )

        with st.spinner("Running predictions..."):
            try:
                X_proc, probs, preds, customer_ids = preprocess_and_predict(
                    df_raw, preprocessor, model
                )

                # Build results dataframe
                results = df_raw.copy()

                # ── Add Customer ID column ────────────────────────────────
                if customer_ids is not None:
                    results.insert(0, 'Customer ID', customer_ids)
                else:
                    results.insert(0, 'Customer ID',
                                   [f'CUST-{i+1:04d}'
                                    for i in range(len(results))])

                # Add engineered features if missing
                service_cols = ['PhoneService', 'MultipleLines',
                                'OnlineSecurity', 'OnlineBackup',
                                'DeviceProtection', 'TechSupport',
                                'StreamingTV', 'StreamingMovies']

                if 'numServices' not in results.columns:
                    existing_svc = [c for c in service_cols
                                    if c in results.columns]
                    results['numServices'] = results[existing_svc].sum(axis=1) \
                                             if existing_svc else 0

                if 'chargePerMonth' not in results.columns:
                    if 'TotalCharges' in results.columns \
                    and 'tenure' in results.columns:
                        results['TotalCharges'] = pd.to_numeric(
                            results['TotalCharges'], errors='coerce'
                        ).fillna(0)
                        results['chargePerMonth'] = (
                            results['TotalCharges'] / (results['tenure'] + 1)
                        )

                if 'tenureBucket' not in results.columns \
                and 'tenure' in results.columns:
                    results['tenureBucket'] = pd.cut(
                        results['tenure'],
                        bins=[0, 12, 24, 48, 72],
                        labels=['New', 'Growing', 'Mature', 'Loyal'],
                        include_lowest=True
                    ).astype(str)

                # Insert prediction columns at front
                results.insert(0, 'Churn Probability %',
                               (probs * 100).round(1))
                results.insert(1, 'Risk Level',
                               [risk_badge(p) for p in probs])
                results.insert(2, 'Prediction',
                               ['Churn' if p == 1 else 'Retain'
                                for p in preds])

                # Sort by risk
                results = results.sort_values(
                    'Churn Probability %', ascending=False
                ).reset_index(drop=True)

                # Store in session state for other tabs
                st.session_state['results']     = results
                st.session_state['X_proc']      = X_proc
                st.session_state['probs']       = probs
                st.session_state['df_raw']      = df_raw
                st.session_state['customer_ids']= customer_ids

                st.success("✓ Predictions complete!")

            except Exception as e:
                st.error(f"Prediction error: {e}")
                st.info("Make sure your CSV has the same columns as the sample.")
                st.stop()

        # ── Summary metrics ───────────────────────────────────────────────
        n_high   = (probs >= 0.7).sum()
        n_medium = ((probs >= 0.4) & (probs < 0.7)).sum()
        n_low    = (probs < 0.4).sum()
        avg_prob = probs.mean() * 100

        st.markdown("#### Prediction Summary")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("🔴 High Risk",   n_high,
                      delta=f"{n_high/len(probs)*100:.1f}% of customers",
                      delta_color="inverse")
        with m2:
            st.metric("🟡 Medium Risk", n_medium,
                      delta=f"{n_medium/len(probs)*100:.1f}% of customers",
                      delta_color="off")
        with m3:
            st.metric("🟢 Low Risk",    n_low,
                      delta=f"{n_low/len(probs)*100:.1f}% of customers")
        with m4:
            st.metric("Avg Churn Prob", f"{avg_prob:.1f}%",
                      delta=f"Baseline: {m['churn_rate']}%",
                      delta_color="off")

        # ── Risk distribution chart ───────────────────────────────────────
        fig_risk = go.Figure(go.Bar(
            x            = ['🔴 High Risk', '🟡 Medium Risk', '🟢 Low Risk'],
            y            = [n_high, n_medium, n_low],
            marker_color = ['#E05A2B', '#F0934A', '#0F6E56'],
            text         = [n_high, n_medium, n_low],
            textposition = 'outside'
        ))
        fig_risk.update_layout(
            title       = 'Customer Risk Distribution',
            xaxis_title = 'Risk Level',
            yaxis_title = 'Number of Customers',
            height      = 350,
            showlegend  = False,
            plot_bgcolor  = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        # ── Churn probability histogram ───────────────────────────────────
        fig_hist = px.histogram(
            x          = probs * 100,
            nbins      = 20,
            title      = 'Churn Probability Distribution',
            labels     = {'x': 'Churn Probability (%)', 'y': 'Count'},
            color_discrete_sequence = ['#534AB7']
        )
        fig_hist.add_vline(x=50, line_dash='dash', line_color='#E05A2B',
                           annotation_text='Decision threshold (50%)')
        fig_hist.add_vline(x=70, line_dash='dot',  line_color='#F0934A',
                           annotation_text='High risk (70%)')
        fig_hist.update_layout(
            height        = 350,
            plot_bgcolor  = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # ── Results table ─────────────────────────────────────────────────
        st.markdown("#### All Customers — Ranked by Churn Risk")
        preferred_cols = ['Customer ID',
                          'Churn Probability %', 'Risk Level', 'Prediction',
                          'tenure', 'Contract', 'MonthlyCharges',
                          'numServices', 'InternetService', 'TechSupport']
        show_cols = [c for c in preferred_cols if c in results.columns]
        st.dataframe(
            results[show_cols],
            use_container_width=True,
            height=400
        )

        # ── High risk customers ───────────────────────────────────────────
        high_risk_df = results[results['Churn Probability %'] >= 70]
        if len(high_risk_df) > 0:
            st.markdown(
                f"#### 🔴 High Risk Customers — Immediate Action Needed"
                f" ({len(high_risk_df)} customers)"
            )
            preferred_hr = ['Customer ID',
                            'Churn Probability %', 'Risk Level',
                            'Contract', 'tenure', 'MonthlyCharges',
                            'numServices']
            show_hr = [c for c in preferred_hr if c in high_risk_df.columns]
            st.dataframe(
                high_risk_df[show_hr],
                use_container_width=True
            )

        # ── Download predictions ──────────────────────────────────────────
        st.download_button(
            label     = "⬇️ Download predictions CSV",
            data      = results.to_csv(index=False),
            file_name = "churn_predictions.csv",
            mime      = "text/csv"
        )

    else:
        st.info("👆 Upload a CSV file above or download the sample to get started.")
        st.markdown("#### What happens after upload:")
        for step in [
            f"Data preprocessed through sklearn pipeline ({m['n_features']} features)",
            f"**{m['model_name']}** predicts churn probability per customer",
            "Customers ranked by risk: High / Medium / Low",
            "Risk distribution chart + probability histogram shown",
            "Download predictions as CSV"
        ]:
            st.markdown(f"  ✓ {step}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLAIN
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### SHAP Explanation")
    st.markdown(
        "SHAP shows **why** the model gave each customer their churn score. "
        "Each feature's contribution is shown as a positive or negative impact."
    )

    # ── Global SHAP ───────────────────────────────────────────────────────
    st.markdown("#### Global Feature Importance")
    st.markdown(
        f"Top churn driver: **`{m['top5_features'][0]}`** "
        f"(Mean SHAP = {m['shap_df'].iloc[0]['Mean_SHAP']:.4f})"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if os.path.exists('reports/shap_global_importance.png'):
            st.image('reports/shap_global_importance.png',
                     caption='Top 15 features by mean |SHAP|',
                     use_container_width=True)
    with col_b:
        if os.path.exists('reports/shap_summary_beeswarm.png'):
            st.image('reports/shap_summary_beeswarm.png',
                     caption='Beeswarm — direction + magnitude per customer',
                     use_container_width=True)

    # ── Top 5 table ───────────────────────────────────────────────────────
    st.markdown("#### Top 5 Global Churn Drivers")
    top5_df = m['shap_df'].head(5)[['Feature', 'Mean_SHAP']].copy()
    top5_df.columns = ['Feature', 'Mean SHAP Value']
    top5_df.index   = range(1, 6)
    st.dataframe(top5_df, use_container_width=True)

    st.divider()

    # ── Per-customer SHAP ─────────────────────────────────────────────────
    st.markdown("#### Per-Customer SHAP Explanation")

    if 'results' not in st.session_state:
        st.warning(
            "⚠️ No predictions yet. "
            "Go to the **🔮 Predict** tab first and upload a CSV."
        )
    else:
        results  = st.session_state['results']
        X_proc   = st.session_state['X_proc']
        probs    = st.session_state['probs']

        # Customer selector
        n_customers = len(results)
        col_sel, col_info = st.columns([1, 2])
        with col_sel:
            customer_idx = st.selectbox(
                "Select customer to explain",
                options = range(n_customers),
                format_func = lambda i: (
                    f"Customer {i+1} — "
                    f"{results.iloc[i]['Churn Probability %']}% risk — "
                    f"{results.iloc[i]['Risk Level']}"
                )
            )

        selected_prob = probs[customer_idx]
        selected_risk = risk_badge(selected_prob)

        with col_info:
            st.markdown("#### Selected Customer Profile")
            profile_cols = ['tenure', 'Contract', 'MonthlyCharges',
                            'InternetService', 'TechSupport', 'numServices']
            existing = [c for c in profile_cols
                        if c in results.columns]
            st.dataframe(
                results.iloc[[customer_idx]][existing],
                use_container_width=True
            )

        # Risk level display
        risk_color = (
            "#E05A2B" if selected_prob >= 0.7 else
            "#F0934A" if selected_prob >= 0.4 else
            "#0F6E56"
        )
        st.markdown(
            f"**Churn Probability:** "
            f"<span style='color:{risk_color};font-size:1.3rem;"
            f"font-weight:700'>{selected_prob*100:.1f}%</span> "
            f"— Risk Level: {selected_risk}",
            unsafe_allow_html=True
        )

        # Waterfall plot
        with st.spinner("Generating SHAP explanation..."):
            try:
                fig_wf = plot_shap_waterfall(
                    explainer,
                    X_proc[customer_idx],
                    feature_names,
                    selected_prob
                )
                st.pyplot(fig_wf)
                plt.close()
            except Exception as e:
                st.error(f"SHAP plot error: {e}")

        # Top features table for this customer
        shap_vals_customer = explainer.shap_values(
            X_proc[customer_idx].reshape(1, -1)
        )[0]
        feat_shap = sorted(
            zip(feature_names, shap_vals_customer),
            key=lambda x: abs(x[1]), reverse=True
        )[:8]

        st.markdown("#### Top 8 Features for This Customer")
        customer_shap_df = pd.DataFrame(feat_shap,
                                         columns=['Feature', 'SHAP Value'])
        customer_shap_df['Impact'] = customer_shap_df['SHAP Value'].apply(
            lambda v: '↑ Increases churn risk' if v > 0
                      else '↓ Decreases churn risk'
        )
        customer_shap_df['SHAP Value'] = customer_shap_df['SHAP Value'].round(4)
        customer_shap_df.index = range(1, len(customer_shap_df) + 1)
        st.dataframe(customer_shap_df, use_container_width=True)



# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — EMAIL (Groq API + Llama 3.3)
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📧 AI Retention Email Generator")
    st.markdown(
        "Select a high-risk customer → AI reads their top SHAP risk factors "
        "→ **Groq API + Llama 3.3** writes a personalised retention email instantly."
    )

    # ── Check API key ─────────────────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        st.error(
            "⚠️ GROQ_API_KEY not found. "
            "Create a `.env` file in the project root with: "
            "`GROQ_API_KEY=your_key_here`"
        )
        st.info(
            "Get a free API key at https://console.groq.com — "
            "no credit card needed."
        )
        st.stop()

    # ── Check predictions exist ───────────────────────────────────────────
    if 'results' not in st.session_state:
        st.warning(
            "⚠️ No predictions yet. "
            "Go to **🔮 Predict** tab first and upload a CSV."
        )
    else:
        results = st.session_state['results']
        probs   = st.session_state['probs']
        X_proc  = st.session_state['X_proc']

        st.divider()

        # ── Customer selector ─────────────────────────────────────────────
        st.markdown("#### Step 1 — Select a Customer")

        at_risk_indices = [
            i for i, p in enumerate(probs) if p >= 0.4
        ]

        if not at_risk_indices:
            st.info("No medium or high risk customers found in uploaded data.")
        else:
            col_sel, col_profile = st.columns([1, 2])

            with col_sel:
                selected_idx = st.selectbox(
                    "Choose customer",
                    options     = at_risk_indices,
                    format_func = lambda i: (
                        f"Customer {results.iloc[i].get('Customer ID', i+1)} "
                        f"— {probs[i]*100:.1f}% risk "
                        f"— {risk_badge(probs[i])}"
                    )
                )

                selected_prob = probs[selected_idx]
                selected_risk = (
                    "High"   if selected_prob >= 0.7 else
                    "Medium" if selected_prob >= 0.4 else "Low"
                )

                risk_color = (
                    "#E05A2B" if selected_prob >= 0.7 else
                    "#F0934A" if selected_prob >= 0.4 else
                    "#0F6E56"
                )
                st.markdown(
                    f"**Churn Probability:** "
                    f"<span style='color:{risk_color};"
                    f"font-size:1.4rem;font-weight:700'>"
                    f"{selected_prob*100:.1f}%</span>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Risk Level:** {risk_badge(selected_prob)}")

            with col_profile:
                st.markdown("**Customer Profile:**")
                profile_cols = ['Customer ID', 'tenure', 'Contract',
                                'MonthlyCharges', 'InternetService',
                                'TechSupport', 'numServices']
                show_profile = [
                    c for c in profile_cols if c in results.columns
                ]
                st.dataframe(
                    results.iloc[[selected_idx]][show_profile],
                    use_container_width=True
                )

            st.divider()

            # ── SHAP features for this customer ───────────────────────────
            st.markdown("#### Step 2 — Top Risk Factors (from SHAP)")

            shap_vals = explainer.shap_values(
                X_proc[selected_idx].reshape(1, -1)
            )[0]

            feat_shap = sorted(
                zip(feature_names, shap_vals),
                key=lambda x: abs(x[1]), reverse=True
            )[:5]

            top_features_list = [
                {
                    'feature'   : f,
                    'shap_value': round(float(v), 4),
                    'direction' : '↑ Increases churn' if v > 0
                                  else '↓ Decreases churn'
                }
                for f, v in feat_shap
            ]

            shap_display         = pd.DataFrame(top_features_list)
            shap_display.columns = ['Feature', 'SHAP Value', 'Direction']
            shap_display.index   = range(1, len(shap_display) + 1)
            st.dataframe(shap_display, use_container_width=True)

            st.divider()

            # ── Email generation ──────────────────────────────────────────
            st.markdown("#### Step 3 — Generate Retention Email")

            col_btn, col_note = st.columns([1, 2])
            with col_btn:
                generate_btn = st.button(
                    "✉️ Generate Email with AI",
                    type             = "primary",
                    use_container_width = True
                )
            with col_note:
                st.caption(
                    "Powered by **Groq API + Llama 3.3**. "
                    "Each click generates a unique email. "
                    "Generation takes ~3-5 seconds."
                )

            if generate_btn:
                with st.spinner(
                    "Groq API is reading SHAP features and "
                    "writing your personalised email..."
                ):
                    try:
                        import sys
                        sys.path.insert(0, os.path.join(
                            os.path.dirname(
                                os.path.dirname(os.path.abspath(__file__))
                            ), 'src'
                        ))
                        from email_generator import generate_retention_email

                        email_text = generate_retention_email(
                            churn_prob   = selected_prob * 100,
                            risk_level   = selected_risk,
                            top_features = top_features_list
                        )

                        st.session_state['generated_email'] = email_text
                        st.session_state['email_customer']  = selected_idx

                    except Exception as e:
                        st.error(f"Email generation failed: {e}")
                        st.info(
                            "Check your GROQ_API_KEY in .env file "
                            "and make sure groq is installed: pip install groq"
                        )

            # ── Show generated email ──────────────────────────────────────
            if 'generated_email' in st.session_state:
                st.markdown("#### Generated Email")

                email_text = st.session_state['generated_email']
                lines      = email_text.strip().split('\n')

                subject = ""
                body    = email_text

                for i, line in enumerate(lines):
                    if line.lower().startswith('subject:'):
                        subject = line.replace('Subject:', '')\
                                      .replace('subject:', '').strip()
                        body    = '\n'.join(lines[i+1:]).strip()
                        break

                if subject:
                    st.markdown(f"**Subject:** {subject}")

                edited_email = st.text_area(
                    "Email body (you can edit before sending):",
                    value  = body,
                    height = 250
                )

                col_copy, col_dl, col_regen = st.columns(3)

                with col_copy:
                    st.download_button(
                        label     = "⬇️ Download as .txt",
                        data      = f"Subject: {subject}\n\n{edited_email}",
                        file_name = f"retention_email_customer"
                                    f"_{selected_idx+1}.txt",
                        mime      = "text/plain"
                    )

                with col_dl:
                    st.download_button(
                        label     = "⬇️ Download as .csv",
                        data      = pd.DataFrame([{
                            'Customer ID'      : results.iloc[
                                selected_idx
                            ].get('Customer ID', selected_idx + 1),
                            'Churn Probability': f"{selected_prob*100:.1f}%",
                            'Risk Level'       : selected_risk,
                            'Subject'          : subject,
                            'Email Body'       : edited_email
                        }]).to_csv(index=False),
                        file_name = "retention_email.csv",
                        mime      = "text/csv"
                    )

                with col_regen:
                    if st.button("🔄 Regenerate Email",
                                 use_container_width=True):
                        del st.session_state['generated_email']
                        st.rerun()

                st.divider()

                # ── Why this email was written this way ───────────────────
                st.markdown("#### Why this email was written this way")
                st.markdown(
                    "Groq API + Llama 3.3 read these specific SHAP risk "
                    "factors and framed the email around them:"
                )

                for f in top_features_list[:3]:
                    if f['shap_value'] > 0:
                        st.markdown(
                            f"- **{f['feature'].replace('_', ' ')}** "
                            f"pushed churn risk UP "
                            f"(SHAP = +{f['shap_value']:.3f}) → "
                            f"email addresses this pain point"
                        )

                st.caption(
                    "Every email is dynamically generated by Llama 3.3 "
                    "based on this specific customer's SHAP values — "
                    "not a template. This connects ML prediction directly "
                    "to business action."
                )

            # ── How it works under the hood ───────────────────────────────
            st.divider()
            st.markdown("#### How email generation works under the hood")
            st.code("""
# src/email_generator.py — Direct Groq SDK
from groq import Groq
import os

def generate_retention_email(churn_prob, risk_level, top_features):
    # Format SHAP features as readable risk factors
    risk_factors = [
        f['feature'].replace('_', ' ')
        for f in top_features[:3]
        if f['shap_value'] > 0
    ]

    prompt = f\"\"\"
    You are a Customer Success Manager.
    Write a retention email for a {risk_level} risk customer.
    Churn probability : {churn_prob}%
    Key risk factors  : {', '.join(risk_factors)}
    Rules: under 120 words, warm tone, one clear next step.
    Start with Subject: [subject line]
    \"\"\"

    client   = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model       = "llama-3.3-70b-versatile",
        messages    = [{"role": "user", "content": prompt}],
        temperature = 0.7,
        max_tokens  = 300
    )
    return response.choices[0].message.content.strip()
            """, language='python')

            
# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 Analytics Dashboard")

    dash_mode = st.radio(
        "Select view",
        ["Training Data Analytics", "Live Prediction Analytics"],
        horizontal=True
    )

    # ════════════════════════════════════════════════════════════════════
    # TRAINING DATA ANALYTICS
    # ════════════════════════════════════════════════════════════════════
    if dash_mode == "Training Data Analytics":

        train_df = m['train_df']
        st.markdown(
            f"Analysing **{len(train_df):,} customers** from training dataset "
            f"with **{m['churn_rate']}% churn rate**."
        )

        # ── Row 1: Churn overview ─────────────────────────────────────────
        st.markdown("#### Churn Overview")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)

        churned    = train_df['Churn'].sum()
        retained   = len(train_df) - churned
        churn_pct  = round(churned / len(train_df) * 100, 1)

        with r1c1:
            st.metric("Total Customers", f"{len(train_df):,}")
        with r1c2:
            st.metric("Churned",  f"{churned:,}",
                      delta=f"{churn_pct}%", delta_color="inverse")
        with r1c3:
            st.metric("Retained", f"{retained:,}",
                      delta=f"{100-churn_pct}%")
        with r1c4:
            st.metric("Avg Monthly Charge",
                      f"${train_df['MonthlyCharges'].mean():.1f}")

        # ── Row 2: Distribution charts ────────────────────────────────────
        col1, col2 = st.columns(2)

        with col1:
            # Churn pie chart
            fig_pie = px.pie(
                values = [churned, retained],
                names  = ['Churned', 'Retained'],
                title  = 'Overall Churn Split',
                color_discrete_sequence = ['#E05A2B', '#534AB7'],
                hole   = 0.4
            )
            fig_pie.update_traces(textposition='inside',
                                   textinfo='percent+label')
            fig_pie.update_layout(height=350,
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Churn by contract type
            contract_churn = train_df.groupby('Contract')['Churn']\
                                     .mean() * 100
            contract_churn = contract_churn.reset_index()
            contract_churn.columns = ['Contract', 'Churn Rate %']
            contract_churn = contract_churn.sort_values(
                'Churn Rate %', ascending=False
            )
            fig_contract = px.bar(
                contract_churn,
                x     = 'Contract',
                y     = 'Churn Rate %',
                title = 'Churn Rate by Contract Type',
                color = 'Churn Rate %',
                color_continuous_scale = 'RdYlGn_r',
                text  = 'Churn Rate %'
            )
            fig_contract.update_traces(
                texttemplate='%{text:.1f}%', textposition='outside'
            )
            fig_contract.add_hline(
                y=churn_pct, line_dash='dash',
                line_color='gray',
                annotation_text=f'Avg {churn_pct}%'
            )
            fig_contract.update_layout(
                height=350, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_contract, use_container_width=True)

        # ── Row 3: Tenure + Monthly charges ──────────────────────────────
        col3, col4 = st.columns(2)

        with col3:
            # Tenure distribution by churn
            fig_tenure = px.histogram(
                train_df,
                x     = 'tenure',
                color = 'Churn',
                title = 'Tenure Distribution by Churn',
                nbins = 30,
                barmode = 'overlay',
                color_discrete_map = {1: '#E05A2B', 0: '#534AB7'},
                labels = {'Churn': 'Churned', 'tenure': 'Tenure (months)'}
            )
            fig_tenure.update_traces(opacity=0.7)
            fig_tenure.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_tenure, use_container_width=True)

        with col4:
            # Monthly charges box plot
            train_df_plot = train_df.copy()
            train_df_plot['Churn Label'] = train_df_plot['Churn'].map(
                {1: 'Churned', 0: 'Retained'}
            )
            fig_charges = px.box(
                train_df_plot,
                x     = 'Churn Label',
                y     = 'MonthlyCharges',
                color = 'Churn Label',
                title = 'Monthly Charges — Churned vs Retained',
                color_discrete_map = {
                    'Churned': '#E05A2B', 'Retained': '#534AB7'
                }
            )
            fig_charges.update_layout(
                height=350, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_charges, use_container_width=True)

        # ── Row 4: Services + Senior citizen ─────────────────────────────
        col5, col6 = st.columns(2)

        with col5:
            # numServices vs churn
            svc_churn = train_df.groupby('numServices')['Churn']\
                                 .mean() * 100
            svc_churn = svc_churn.reset_index()
            svc_churn.columns = ['Num Services', 'Churn Rate %']
            fig_svc = px.bar(
                svc_churn,
                x     = 'Num Services',
                y     = 'Churn Rate %',
                title = 'Churn Rate by Number of Services',
                color = 'Churn Rate %',
                color_continuous_scale = 'RdYlGn_r',
                text  = 'Churn Rate %'
            )
            fig_svc.update_traces(
                texttemplate='%{text:.1f}%', textposition='outside'
            )
            fig_svc.update_layout(
                height=350, showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_svc, use_container_width=True)

        with col6:
            # Tenure bucket churn
            if 'tenureBucket' in train_df.columns:
                bucket_order = ['New', 'Growing', 'Mature', 'Loyal']
                bucket_churn = train_df.groupby('tenureBucket')['Churn']\
                                        .mean() * 100
                bucket_churn = bucket_churn.reindex(
                    [b for b in bucket_order
                     if b in bucket_churn.index]
                ).reset_index()
                bucket_churn.columns = ['Tenure Bucket', 'Churn Rate %']
                fig_bucket = px.bar(
                    bucket_churn,
                    x     = 'Tenure Bucket',
                    y     = 'Churn Rate %',
                    title = 'Churn Rate by Tenure Bucket',
                    color = 'Tenure Bucket',
                    color_discrete_sequence = [
                        '#E05A2B','#F0934A','#534AB7','#0F6E56'
                    ],
                    text  = 'Churn Rate %'
                )
                fig_bucket.update_traces(
                    texttemplate='%{text:.1f}%',
                    textposition='outside'
                )
                fig_bucket.update_layout(
                    height=350, showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_bucket, use_container_width=True)

        # ── Row 5: Model comparison ───────────────────────────────────────
        st.markdown("#### Model Performance Comparison")
        results_plot = m['results_df'].copy()
        fig_models   = px.bar(
            results_plot,
            x          = 'Model',
            y          = ['F1_Churn', 'F1_Weighted', 'ROC_AUC'],
            barmode    = 'group',
            title      = 'All Models — Key Metrics Comparison',
            color_discrete_sequence = ['#E05A2B', '#534AB7', '#0F6E56'],
            labels     = {'value': 'Score', 'variable': 'Metric'}
        )
        fig_models.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_models, use_container_width=True)

        # Full results table
        st.markdown("#### Full Model Results Table")
        st.dataframe(
            m['results_df'].set_index('Model'),
            use_container_width=True
        )

    # ════════════════════════════════════════════════════════════════════
    # LIVE PREDICTION ANALYTICS
    # ════════════════════════════════════════════════════════════════════
    else:
        if 'results' not in st.session_state:
            st.warning(
                "⚠️ No live predictions yet. "
                "Go to **🔮 Predict** tab and upload a CSV first."
            )
            st.info(
                "Once predictions are made, this tab will show "
                "interactive analytics for your specific customers."
            )
        else:
            results = st.session_state['results']
            probs   = st.session_state['probs']
            df_raw  = st.session_state['df_raw']

            st.markdown(
                f"Showing analytics for **{len(results)} uploaded customers**."
            )

            # ── Summary metrics ───────────────────────────────────────────
            n_high   = (probs >= 0.7).sum()
            n_medium = ((probs >= 0.4) & (probs < 0.7)).sum()
            n_low    = (probs < 0.4).sum()

            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                st.metric("Total Customers", len(results))
            with lc2:
                st.metric("🔴 High Risk", n_high,
                          delta=f"{n_high/len(probs)*100:.1f}%",
                          delta_color="inverse")
            with lc3:
                st.metric("🟡 Medium Risk", n_medium,
                          delta=f"{n_medium/len(probs)*100:.1f}%",
                          delta_color="off")
            with lc4:
                st.metric("🟢 Low Risk", n_low,
                          delta=f"{n_low/len(probs)*100:.1f}%")

            # ── Risk breakdown pie ────────────────────────────────────────
            col1, col2 = st.columns(2)
            with col1:
                fig_live_pie = px.pie(
                    values = [n_high, n_medium, n_low],
                    names  = ['High Risk', 'Medium Risk', 'Low Risk'],
                    title  = 'Risk Level Distribution (Uploaded Customers)',
                    color_discrete_sequence = ['#E05A2B','#F0934A','#0F6E56'],
                    hole   = 0.4
                )
                fig_live_pie.update_traces(
                    textposition='inside', textinfo='percent+label'
                )
                fig_live_pie.update_layout(
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_live_pie, use_container_width=True)

            with col2:
                # Churn probability histogram
                fig_live_hist = px.histogram(
                    x      = probs * 100,
                    nbins  = 15,
                    title  = 'Churn Probability Distribution',
                    labels = {'x': 'Churn Probability (%)', 'y': 'Count'},
                    color_discrete_sequence = ['#534AB7']
                )
                fig_live_hist.add_vline(
                    x=70, line_dash='dash', line_color='#E05A2B',
                    annotation_text='High risk threshold'
                )
                fig_live_hist.add_vline(
                    x=50, line_dash='dot', line_color='#F0934A',
                    annotation_text='Decision threshold'
                )
                fig_live_hist.update_layout(
                    height=350,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_live_hist, use_container_width=True)

            # ── Contract type breakdown if available ──────────────────────
            if 'Contract' in results.columns:
                col3, col4 = st.columns(2)
                with col3:
                    contract_counts = results['Contract'].value_counts()
                    fig_ct = px.bar(
                        x     = contract_counts.index,
                        y     = contract_counts.values,
                        title = 'Customers by Contract Type',
                        labels= {'x': 'Contract', 'y': 'Count'},
                        color = contract_counts.values,
                        color_continuous_scale = 'Blues',
                        text  = contract_counts.values
                    )
                    fig_ct.update_traces(textposition='outside')
                    fig_ct.update_layout(
                        height=350, showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_ct, use_container_width=True)

                with col4:
                    # Average churn probability by contract
                    results_copy = results.copy()
                    results_copy['prob'] = probs
                    avg_by_contract = results_copy.groupby('Contract')['prob']\
                                                   .mean() * 100
                    avg_by_contract = avg_by_contract.reset_index()
                    avg_by_contract.columns = ['Contract',
                                                'Avg Churn Prob %']
                    fig_avg = px.bar(
                        avg_by_contract,
                        x     = 'Contract',
                        y     = 'Avg Churn Prob %',
                        title = 'Avg Churn Probability by Contract',
                        color = 'Avg Churn Prob %',
                        color_continuous_scale = 'RdYlGn_r',
                        text  = 'Avg Churn Prob %'
                    )
                    fig_avg.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='outside'
                    )
                    fig_avg.update_layout(
                        height=350, showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig_avg, use_container_width=True)

            # ── Full ranked table ─────────────────────────────────────────
            st.markdown("#### All Customers Ranked by Churn Risk")
            display_cols = ['Churn Probability %', 'Risk Level',
                            'Prediction']
            extra_cols   = ['tenure', 'Contract', 'MonthlyCharges',
                            'numServices']
            show_cols    = display_cols + [
                c for c in extra_cols if c in results.columns
            ]
            st.dataframe(
                results[show_cols],
                use_container_width=True,
                height=400
            )

            # Download
            st.download_button(
                label     = "⬇️ Download full analytics CSV",
                data      = results.to_csv(index=False),
                file_name = "churn_analytics.csv",
                mime      = "text/csv"
            )