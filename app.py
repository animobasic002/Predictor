import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_utils import (
    generate_amino_acid_data,
    generate_qbd_formulations, 
    generate_stability_data, 
    calculate_shelf_life, 
    calculate_arrhenius,
    train_ml_model
)

st.set_page_config(page_title="Pioglitazone Stability Predictor", layout="wide")

st.markdown("""
<style>
    /* iOS Mobile App Feel: Hide Streamlit Header & Footer */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Maximize screen real estate on mobile by reducing padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
    }
    
    /* Dynamic Typography for smaller screens */
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.3rem !important; margin-bottom: 0.5rem !important; }
    p { font-size: 0.95rem !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>💊 Pioglitazone Formulation Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: gray; font-size: 1.1rem; margin-bottom: 20px;'>Advanced Stability Analytics & ML Predictions</p>", unsafe_allow_html=True)

# Shortened tab names for mobile responsiveness
tabs = st.tabs(["1. Screening", "2. Data", "3. QbD", "4. Shelf-Life", "5. Kinetics", "6. ML Risk"])

# Mobile friendly plotly config (disables panning/zooming which interferes with phone scrolling)
plotly_config = {
    'scrollZoom': False,
    'displayModeBar': False, # Hides the top toolbar which clutters mobile screens
}

# Initialize session state for data
if 'stability_data' not in st.session_state:
    st.session_state['stability_data'] = generate_stability_data('F1')

with tabs[0]:
    with st.container(border=True):
        st.header("Amino Acid Screening")
        st.markdown("Testing 8 different amino acids to select the most promising one for stability.")
        
        aa_data = generate_amino_acid_data()
        st.dataframe(aa_data.style.highlight_max(subset=['Stability Score (0-100)'], color='lightgreen'), use_container_width=True)
        
        fig = px.bar(aa_data, x='Amino Acid', y='Stability Score (0-100)', title="Stability Score by Amino Acid", color='Stability Score (0-100)')
        fig.update_layout(dragmode=False, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        
        st.success("**Selection:** Arginine showed the highest stability score. Moving forward with Arginine.")

with tabs[1]:
    with st.container(border=True):
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

with tabs[2]:
    with st.container(border=True):
        st.header("Response Surface Methodology")
        st.markdown("Analyze Design of Experiments (DoE) formulations to find the optimal Arginine ratio.")
        
        qbd_df = generate_qbd_formulations()
        
        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.dataframe(qbd_df, use_container_width=True)
        with col2:
            x = np.linspace(1, 10, 20)
            y = np.linspace(2, 5, 20)
            X, Y = np.meshgrid(x, y)
            Z = 100 - (X*0.5 + Y*1.2) + np.random.normal(0, 1, X.shape)
            
            enable_3d = st.toggle("Unlock 3D Rotation & Zoom", value=True, help="Turn this off if the graph gets in the way of scrolling on your phone.")
            
            fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Blues')])
            fig.update_layout(
                title="Dissolution vs Arginine vs Binder",
                scene=dict(xaxis_title='Arginine', yaxis_title='Binder', zaxis_title='Dissolution'),
                margin=dict(l=0, r=0, b=0, t=40),
                dragmode="turntable" if enable_3d else False 
            )
            
            # Allow zooming and show Plotly controls ONLY for this 3D graph when unlocked
            config_3d = {
                'scrollZoom': enable_3d,
                'displayModeBar': enable_3d,
                'displaylogo': False
            }
            st.plotly_chart(fig, use_container_width=True, config=config_3d)

with tabs[3]:
    with st.container(border=True):
        st.header("Shelf-Life Extrapolation")
        
        df = st.session_state['stability_data']
        param = st.selectbox("Select Parameter:", ["Assay (%)", "Degradation (%)"])
        limit = st.number_input("Acceptance Limit:", value=95.0 if "Assay" in param else 1.0)
        is_decreasing = "Assay" in param
        
        shelf_life, ci_df = calculate_shelf_life(df, param, limit, is_decreasing)
        
        if shelf_life:
            st.info(f"**Predicted Shelf-Life:** {shelf_life} Months")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Timepoint (Months)'], y=df[param], mode='markers', name='Actual Data'))
            fig.add_trace(go.Scatter(x=ci_df['Time'], y=ci_df['Predicted'], mode='lines', name='Linear Fit', line=dict(color='#007AFF')))
            fig.add_trace(go.Scatter(x=ci_df['Time'], y=ci_df['Bound'], mode='lines', name='95% Bound', line=dict(dash='dash', color='#FF3B30')))
            fig.add_hline(y=limit, line_dash="dot", line_color="#34C759", annotation_text="Limit")
            
            fig.update_layout(
                title=f"{param} Extrapolation", 
                xaxis_title="Months", 
                yaxis_title=param,
                dragmode=False, # Disable panning to allow mobile scrolling
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        else:
            st.warning("Not enough data to calculate shelf-life.")

with tabs[4]:
    with st.container(border=True):
        st.header("Degradation Kinetics")
        
        col1, col2, col3 = st.columns(3)
        k_25 = col1.number_input("k (25°C):", value=0.015, format="%.3f")
        k_30 = col2.number_input("k (30°C):", value=0.025, format="%.3f")
        k_40 = col3.number_input("k (40°C):", value=0.065, format="%.3f")
        
        Ea, inv_T, ln_k, inv_T_line, ln_k_line = calculate_arrhenius(k_25, k_30, k_40)
        
        st.success(f"**Calculated Activation Energy (Ea):** {Ea} kJ/mol")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=inv_T, y=ln_k, mode='markers', name='Data Points', marker=dict(size=10, color='#007AFF')))
        fig.add_trace(go.Scatter(x=inv_T_line, y=ln_k_line, mode='lines', name='Arrhenius Fit', line=dict(color='#FF9500')))
        fig.update_layout(
            title="Arrhenius Plot: ln(k) vs 1/T", 
            xaxis_title="1/T (K^-1)", 
            yaxis_title="ln(k)",
            dragmode=False,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

with tabs[5]:
    with st.container(border=True):
        st.header("ML Risk Prediction")
        
        try:
            model = train_ml_model()
            
            col1, col2, col3 = st.columns(3)
            arg = col1.slider("Arginine (%)", 1.0, 10.0, 4.0)
            moist = col2.slider("Moisture (%)", 1.0, 4.0, 2.6)
            hard = col3.slider("Hardness (kp)", 5.0, 15.0, 10.0)
            
            features = pd.DataFrame({'Arginine (%)': [arg], 'Moisture (%)': [moist], 'Hardness (kp)': [hard]})
            prob = model.predict_proba(features)[0]
            prediction = 1 if prob > 0.5 else 0
            
            st.metric("Probability of Failure", f"{prob*100:.1f}%")
            
            if prediction == 1:
                st.error("🔴 **High Risk Prediction:** This formulation has a high probability of failing stability criteria.")
            else:
                st.success("🟢 **Low Risk Prediction:** This formulation is predicted to be stable.")
            
            importance = pd.DataFrame({'Feature': features.columns, 'Importance': model.feature_importances_})
            fig = px.bar(importance, x='Feature', y='Importance', title="ML Feature Importance", color_discrete_sequence=['#5856D6'])
            fig.update_layout(
                dragmode=False,
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig, use_container_width=True, config=plotly_config)
        except Exception as e:
            st.error(f"Error running ML model: {e}")
