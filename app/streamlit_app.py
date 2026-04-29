import streamlit as st
import pandas as pd
import sys
import os
import requests
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import random
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

from src.pipeline.predict_pipeline import PredictPipeline
from src.explanability.shap_explainer import ShapExplainer
from src.monitoring.db import save_to_db

st.set_page_config(page_title="Fraud Guard", layout="wide")

# --- Initialize Session State for Auto-Fill Buttons ---
if "t_time" not in st.session_state: st.session_state.t_time = 10000.0
if "t_amount" not in st.session_state: st.session_state.t_amount = 100.0
for i in range(1, 29):
    if f"v_{i}" not in st.session_state: st.session_state[f"v_{i}"] = 0.0

import random

def generate_sample(is_fraud=False):
    """Fills the UI with either a normal transaction or a simulated fraud attack"""
    if is_fraud:
        # A mini-database of 3 completely different, real fraud signatures
        fraud_database = [
            {
                "Time": 406.0, "Amount": 0.00,
                "V": [-2.312, 1.951, -1.609, 3.997, -0.522, -1.426, -2.537, 1.391, -2.770, -2.772,
                       3.202, -2.899, -0.595, -4.289, 0.389, -1.140, -2.830, -0.016, 0.416, 0.126,
                       0.517, -0.035, -0.465, 0.320, 0.044, 0.177, 0.261, -0.143]
            },
            {
                # A "Borderline" Fraud Signature to test the Threshold Slider
                "Time": 12500.0, "Amount": 99.99,
                "V": [-0.95, 0.52, -1.53, 0.85, -0.21, 0.11, -0.45, 0.22, -0.63, -1.05,
                       1.20, -1.55, 0.30, -2.01, 0.10, -0.55, -1.22, 0.20, 0.45, -0.10,
                       0.25, 0.15, -0.12, 0.05, 0.22, -0.15, 0.02, 0.05]
            },
            {
                "Time": 4462.0, "Amount": 1.00,
                "V": [-2.303, 1.759, -0.359, 2.330, -0.821, -0.075, -0.560, 1.214, -1.385, -2.776,
                       3.231, -2.719, -1.059, -3.535, -1.583, -1.488, -2.573, -0.739, 0.380, -0.430,
                      -0.294, -0.932, 0.172, -0.087, -0.156, -0.542, 0.039, -0.153]
            }
        ]
        
        # Randomly select one of the real fraud signatures
        chosen_fraud = random.choice(fraud_database)
        
        st.session_state.t_time = chosen_fraud["Time"]
        st.session_state.t_amount = chosen_fraud["Amount"]
        for i in range(1, 29):
            st.session_state[f"v_{i}"] = chosen_fraud["V"][i-1]
            
    else:
        # Normal baseline simulation (This stays random because normal transactions are easy to mimic)
        st.session_state.t_time = random.uniform(100, 150000)
        st.session_state.t_amount = random.uniform(5, 150)
        for i in range(1, 29):
            st.session_state[f"v_{i}"] = random.uniform(-1.0, 1.0)

# --- UI Sidebar & Navigation ---
page = st.sidebar.selectbox("📌 Choose Section", ["Prediction", "Drift Monitoring"])

if page == "Prediction":
    st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='color: #1E3A8A;'>💳 Fraud Guard Intelligence</h1>
            <p style='color: #6B7280; font-size: 1.2rem;'>Real-Time Transaction Risk Analysis</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 🛠️ Demo Controls")
        # 🔥 UI UPGRADE: One-click auto-fill buttons
        demo_col1, demo_col2 = st.columns(2)
        with demo_col1:
            if st.button("✅ Simulate Normal User", use_container_width=True):
                generate_sample(is_fraud=False)
        with demo_col2:
            if st.button("🚨 Simulate Fraud Attack", type="primary", use_container_width=True):
                generate_sample(is_fraud=True)

        st.markdown("### 📥 Transaction Input")
        with st.container(border=True):
            with st.form("transaction_form"):
                
                # Tie sliders to session state keys
                t_time = st.slider("Time (Sec)", 0.0, 172800.0, key="t_time")
                t_amount = st.slider("Amount ($)", 0.0, 5000.0, key="t_amount")
                
                with st.expander("PCA Feature Vectors (V1 - V28)", expanded=False):
                    v_data = {}
                    for i in range(1, 29):
                        # Tie number inputs to session state, making them instantly update
                        v_data[f"V{i}"] = st.number_input(f"V{i}", key=f"v_{i}", format="%.4f")
                
                st.markdown("---")
                threshold = st.slider("AI Sensitivity (Threshold)", 0.05, 0.95, 0.15)
                submit_btn = st.form_submit_button("🔍 Run Analysis", use_container_width=True)

    with col2:
        st.markdown("### 📊 Live Telemetry & Assessment")
        if not submit_btn:
            st.info("Awaiting transaction payload. Click 'Simulate' then 'Run Analysis'.")

        if submit_btn:
            payload = {"Time": st.session_state.t_time, "Amount": st.session_state.t_amount, **v_data}
            
            try:
                with st.spinner("Analyzing threat vectors..."):
                    response = requests.post(API_URL, json=payload, timeout=30)
                    result = response.json()
                    
                prob = result["fraud_probability"]
                pred = 1 if prob > threshold else 0
                action = "🚫 Block Transaction" if pred == 1 else "✅ Allow Transaction"
                
                if pred == 1:
                    st.error(f"🚨 FRAUD DETECTED: {action}")
                else:
                    st.success(f"✅ TRANSACTION SAFE: {action}")
                
                m_col1, m_col2 = st.columns(2)
                m_col1.metric("Risk Level", f"{prob:.4%}")
                m_col2.metric("Prediction Output", pred)
                
                st.progress(float(prob))
                st.markdown("---")
                
                st.subheader("🧠 Explainable AI (SHAP)")
                with st.spinner("Generating explanations..."):
                    input_df = pd.DataFrame([payload])
                    
                    pipeline = PredictPipeline()
                    processed_df = pipeline.preprocess(input_df)
                    
                    explainer = ShapExplainer()
                    shap_values = explainer.explain(processed_df)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    shap.plots.waterfall(shap_values[0], show=False) 
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"⏳ Error connecting to API: {e}")

elif page == "Drift Monitoring":
    st.title("📉 Data Drift Monitoring")
    st.markdown("### Monitor model health & trigger verified retraining")
    
    # 🔥 UI UPGRADE: Drift Injection Button for easy testing
    with st.expander("🛠️ Demo Tools: Force Data Drift", expanded=True):
        st.write("Inject 50 heavily skewed transactions into the database to trigger a statistical drift warning.")
        if st.button("💉 Inject Synthetic Drift Data", type="primary"):
            with st.spinner("Injecting bad data into Cloud DB..."):
                for _ in range(50):
                    skewed_data = {"Time": random.uniform(10, 50000), "Amount": random.uniform(1000, 5000)}
                    for i in range(1, 29):
                        skewed_data[f"V{i}"] = random.uniform(-15.0, 15.0) # Massive deviation
                    save_to_db(skewed_data, pred=1, prob=0.99)
                st.success("✅ 50 Skewed rows injected! Now click 'Run Drift Detection' below.")

    st.markdown("---")

    try:
        from src.monitoring.drift import detect_drift
        from src.pipeline.retrain_pipeline import retrain
    except:
        st.error("⚠️ Drift feature not supported in this environment")
        st.stop()

    if st.button("🚀 Run Drift Detection"):
        with st.spinner("Running statistical drift analysis..."):
            try:
                report_path = detect_drift("data/creditcard.csv")
                if report_path:
                    st.success("✅ Drift report generated!")
                    st.session_state["drift_done"] = True
                else:
                    st.warning("⚠️ Not enough data in live DB (Needs 50 rows). Use the Demo Injector above!")
            except Exception as e:
                st.error(f"⚠️ Error running drift: {e}")

    report_path = "reports/drift_report.html"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=800, scrolling=True)

    st.markdown("---")
    st.subheader("🔁 Human-in-the-Loop Retraining")
    st.write("Ensure database contains human-verified `Actual_Class` labels before retraining.")

    if st.session_state.get("drift_done", False):
        if st.button("⚡ Retrain Model (Requires Verified Data)"):
            with st.spinner("Retraining model..."):
                try:
                    retrain()
                    st.success("✅ Model retrained successfully with verified data!")
                except ValueError as ve:
                    st.error(f"❌ {ve}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")