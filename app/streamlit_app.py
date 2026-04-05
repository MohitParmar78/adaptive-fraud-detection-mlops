import streamlit as st
import pandas as pd
import sys
import os
import requests
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.predict_pipeline import PredictPipeline
from src.explanability.shap_explainer import ShapExplainer

# Load pipeline
pipeline = PredictPipeline()

# Page config
st.set_page_config(page_title="Fraud Detection System", layout="wide")

# Title
st.title("💳 Fraud Detection System")
st.markdown("### Real-Time Transaction Risk Analysis")

page = st.sidebar.selectbox(
    "📌 Choose Section",
    ["Prediction", "Drift Monitoring"]
)

if page == "Prediction":

    # Sidebar input
    st.sidebar.header("🔧 Enter Transaction Details")

    def user_input():
        data = {}

        data["Time"] = st.sidebar.number_input("Time", value=10000)

        # PCA features
        for i in range(1, 29):
            data[f"V{i}"] = st.sidebar.number_input(f"V{i}", value=0.0)

        data["Amount"] = st.sidebar.number_input("Amount", value=100.0)

        return pd.DataFrame([data])


    input_df = user_input()

    # Show input
    st.subheader("📥 Input Data")
    st.write(input_df)

    # Predict button
    if st.button("🔍 Predict Fraud"):

        ## pred, prob = pipeline.predict(input_df)
    
        # Convert input to JSON
        data = input_df.to_dict(orient="records")[0]

        # Call FastAPI
        try:
            response = requests.post(
            "https://your-api.onrender.com/predict",
            json=data,
            timeout=60
            )

            result = response.json()

        except Exception as e:
            st.error("⏳ Server is waking up or unreachable. Try again.")
            st.stop()

        pred = result["fraud_prediction"]
        prob = result["fraud_probability"]

        st.subheader("📊 Prediction Result")

        # 🔹 Result message
        if pred == 1:
            st.error("🚨 Fraud Detected!")
        else:
            st.success("✅ Transaction is Safe")

        # 🏆 1. METRIC DISPLAY (ADD HERE)
        st.metric("Fraud Probability", f"{prob:.2%}")

        # 🏆 2. COLUMNS LAYOUT (ADD HERE)
        col1, col2 = st.columns(2)
        col1.metric("Prediction", pred)
        col2.metric("Fraud Probability", f"{prob:.2%}")

        # 🏆 3. RISK BAR (keep this)
        st.subheader("📈 Risk Level")
        st.progress(float(prob))

        # 🏆 4. CHART (ADD HERE)
        st.subheader("📊 Risk Distribution")
        st.bar_chart({
            "Fraud": prob,
            "Safe": 1 - prob
        })
    
        # 🏆 ---------------- SHAP START ----------------

        st.subheader("🧠 Model Explanation")

        explainer = ShapExplainer()

        shap_values_raw = explainer.explainer.shap_values(input_df)

        explanation = shap.Explanation(
            values=shap_values_raw[0][:, 1],
            base_values=explainer.explainer.expected_value[1],
            data=input_df.values[0],
            feature_names=input_df.columns
        )

        # 🔹 Plot SHAP
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)

        # 🏆 ---------------- SHAP END ----------------

        # 🔹 Decision logic
        if prob > 0.8:
            st.error("🚫 High Risk → Block Transaction")
        elif prob > 0.4:
            st.warning("⚠️ Medium Risk → Review Needed")
        else:
            st.success("✅ Low Risk → Allow")
            
elif page == "Drift Monitoring":

    st.title("📉 Data Drift Monitoring")

    st.markdown("### Monitor model health & retrain")

    import os
    from src.monitoring.drift import detect_drift
    from src.pipeline.retrain_pipeline import retrain

    # 🧠 RUN DRIFT
    if st.button("🚀 Run Drift Detection"):

        with st.spinner("Running drift detection..."):

            # NOTE: path fix (since streamlit runs from app/)
            report_path = detect_drift("data/creditcard.csv")

        if report_path:
            st.success("✅ Drift report generated!")
            st.session_state["drift_done"] = True
        else:
            st.warning("⚠️ Not enough data")

    # 📄 OPEN REPORT
    st.subheader("📄 Drift Report")

    report_path = "reports/drift_report.html"

    if os.path.exists(report_path):

        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Show inside app
        components.html(html, height=800, scrolling=True)

        # Download option
        st.download_button(
            "📥 Download Report",
            html,
            "drift_report.html"
        )

    # 🔁 RETRAIN
    st.markdown("---")
    st.subheader("🔁 Model Retraining")

    if st.session_state.get("drift_done", False):

        if st.button("⚡ Retrain Model"):

            with st.spinner("Retraining model..."):

                try:
                    retrain()
                    st.success("✅ Model retrained successfully!")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

    else:
        st.info("👉 Run drift detection first")