import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_utils import (
    generate_qbd_formulations, 
    generate_stability_data, 
    calculate_shelf_life, 
    calculate_arrhenius,
    train_ml_model
)

st.set_page_config(page_title="Pioglitazone Stability Predictor", layout="wide")

# iOS Style CSS Injection
st.markdown("""
<style>
    /* iOS light mode styling */
    .stApp {
        background-color: #F2F2F7;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #E5E5EA;
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #1C1C1E;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #007AFF !important;
        font-weight: 600;
    }
    div[data-testid="stVerticalBlock"] > div {
        background-color: transparent;
    }
    /* Cards */
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #1C1C1E; font-weight: 600; }
    h1 { font-size: 2.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>💊 Pioglitazone Formulation Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #8E8E93; font-size: 1.1rem; margin-bottom: 20px;'>Advanced Stability Analytics & ML Predictions</p>", unsafe_allow_html=True)

tabs = st.tabs(["1. Data Ingestion", "2. QbD & RSM", "3. Shelf-Life (ICH)", "4. Kinetics", "5. ML Prediction"])

# Initialize session state for data
if 'stability_data' not in st.session_state:
    st.session_state['stability_data'] = generate_stability_data('F1')

with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Automated Data Ingestion")
    st.markdown("Upload raw stability data (CSV/Excel) from HPLC or LIMS.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['stability_data'] = df
            st.success("Data successfully ingested and parsed!")
        except Exception as e:
            st.error(f"Error parsing file: {e}")
    
    st.dataframe(st.session_state['stability_data'], use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Response Surface Methodology (RSM)")
    st.markdown("Analyze Design of Experiments (DoE) formulations to find the optimal Arginine ratio.")
    
    qbd_df = generate_qbd_formulations()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(qbd_df, use_container_width=True)
    with col2:
        # Create a 3D Surface Plot for RSM
        x = np.linspace(1, 10, 20)
        y = np.linspace(2, 5, 20)
        X, Y = np.meshgrid(x, y)
        Z = 100 - (X*0.5 + Y*1.2) + np.random.normal(0, 1, X.shape) # Mock response
        
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues')])
        fig.update_layout(
            title="RSM: Dissolution vs Arginine vs Binder",
            scene=dict(xaxis_title='Arginine (%)', yaxis_title='Binder (%)', zaxis_title='Dissolution (%)'),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Shelf-Life Extrapolation (ICH Q1E)")
    st.markdown("Calculate statistical shelf-life using linear regression and 95% confidence intervals.")
    
    df = st.session_state['stability_data']
    param = st.selectbox("Select Parameter:", ["Assay (%)", "Degradation (%)"])
    limit = st.number_input("Acceptance Limit:", value=95.0 if "Assay" in param else 1.0)
    is_decreasing = "Assay" in param
    
    shelf_life, ci_df = calculate_shelf_life(df, param, limit, is_decreasing)
    
    if shelf_life:
        st.info(f"**Predicted Shelf-Life:** {shelf_life} Months")
        
        fig = go.Figure()
        # Raw Data
        fig.add_trace(go.Scatter(x=df['Timepoint (Months)'], y=df[param], mode='markers', name='Actual Data'))
        # Prediction Line
        fig.add_trace(go.Scatter(x=ci_df['Time'], y=ci_df['Predicted'], mode='lines', name='Linear Fit', line=dict(color='#007AFF')))
        # Confidence Bound
        fig.add_trace(go.Scatter(x=ci_df['Time'], y=ci_df['Bound'], mode='lines', name='95% Confidence Bound', line=dict(dash='dash', color='#FF3B30')))
        # Limit Line
        fig.add_hline(y=limit, line_dash="dot", line_color="#34C759", annotation_text="Limit")
        
        fig.update_layout(title=f"{param} Extrapolation", xaxis_title="Months", yaxis_title=param)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data to calculate shelf-life.")
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Degradation Kinetics (Arrhenius Equation)")
    st.markdown("Calculate Activation Energy (Ea) based on degradation rates at multiple temperatures.")
    
    col1, col2, col3 = st.columns(3)
    k_25 = col1.number_input("Degradation Rate k (25°C):", value=0.015, format="%.3f")
    k_30 = col2.number_input("Degradation Rate k (30°C):", value=0.025, format="%.3f")
    k_40 = col3.number_input("Degradation Rate k (40°C):", value=0.065, format="%.3f")
    
    Ea, inv_T, ln_k, inv_T_line, ln_k_line = calculate_arrhenius(k_25, k_30, k_40)
    
    st.success(f"**Calculated Activation Energy (Ea):** {Ea} kJ/mol")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=inv_T, y=ln_k, mode='markers', name='Data Points', marker=dict(size=10, color='#007AFF')))
    fig.add_trace(go.Scatter(x=inv_T_line, y=ln_k_line, mode='lines', name='Arrhenius Fit', line=dict(color='#FF9500')))
    fig.update_layout(title="Arrhenius Plot: ln(k) vs 1/T", xaxis_title="1/T (K^-1)", yaxis_title="ln(k)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[4]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Machine Learning Risk Prediction")
    st.markdown("Predict formulation failure risk using a trained Random Forest classifier.")
    
    model = train_ml_model()
    
    st.markdown("### Test a New Formulation")
    col1, col2, col3 = st.columns(3)
    arg = col1.slider("Arginine (%)", 1.0, 10.0, 4.0)
    moist = col2.slider("Moisture (%)", 1.0, 4.0, 2.6)
    hard = col3.slider("Hardness (kp)", 5.0, 15.0, 10.0)
    
    features = pd.DataFrame({'Arginine (%)': [arg], 'Moisture (%)': [moist], 'Hardness (kp)': [hard]})
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        st.error("🔴 **High Risk Prediction:** This formulation has a high probability of failing stability criteria.")
    else:
        st.success("🟢 **Low Risk Prediction:** This formulation is predicted to be stable.")
    
    # Feature Importance
    importance = pd.DataFrame({'Feature': features.columns, 'Importance': model.feature_importances_})
    fig = px.bar(importance, x='Feature', y='Importance', title="ML Feature Importance", color_discrete_sequence=['#5856D6'])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
