# 🛡️ Fraud Guard Intelligence: Adaptive MLOps System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon_Cloud-336791)

---

## 📌 Overview

This project is an **end-to-end, production-ready distributed MLOps system** that combines:

* Real-time Microservice Prediction
* Model Explainability (SHAP)
* Continuous Data Drift Monitoring (Evidently AI)
* Human-in-the-Loop (HITL) Retraining Pipeline

Most models are trained once and decay silently. This system simulates how modern financial architectures dynamically detect fraud, monitor their own health, and adapt to evolving hacker patterns over time.

---

## 🎯 Key Features

✅ **Decoupled Microservices:** Independent Cloud UI (Hugging Face) and API (Render)

✅ **Real-time fraud prediction:** Low-latency inference via FastAPI

✅ **Explainable AI:** SHAP integration for mathematical prediction transparency

✅ **Data drift detection:** Statistical distribution monitoring using Evidently AI

✅ **Automated retraining pipeline:** Triggered via UI, tracked via DagsHub/MLflow

✅ **Cloud Database Telemetry:** Live transaction logging to Neon PostgreSQL

✅ **Resilient Engineering:** Advanced connection pooling and rollback architecture


---

## 📊 Dataset

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> ⚠️ Due to size limits, download manually and place inside the data/ folder.

### Key Details

* 284,807 transactions
* 492 fraud cases
* Highly imbalanced (0.17% Fraud)

### Features

* Time
* Amount
* V1–V28 (PCA transformed)
* Class (0 = Normal, 1 = Fraud)

---

## 🧠 System Architecture

    User (Streamlit UI on Hugging Face)
            ↓
    FastAPI Backend (REST API on Render)
            ↓
    ML Model (XGBoost + RF Ensemble)
            ↓
    Neon PostgreSQL (Stores live cloud telemetry)
            ↓
    Monitoring Layer (Evidently AI Drift Detection)
            ↓
    Human-in-the-Loop (Verifies Actual_Class)
            ↓
    Retraining Pipeline (Logs to MLflow/DagsHub)

---

## ⚙️ Project Modules

### 🔹 1. Data Pipeline (train_pipeline.py)
* Data loading & preprocessing
* Feature scaling
* SMOTE for severe class imbalance handling
* Model training (Random Forest + XGBoost)
* **[UPGRADE]** MLflow integration for hyperparameter tracking and model versioning

### 🔹 2. Prediction Pipeline (predict_pipeline.py)
* Loads trained .pkl models
* Enforces strict feature-column ordering
* Performs weighted ensemble prediction
* Returns Prediction (0/1) and Risk Probability (%)

### 🔹 3. FastAPI Backend (app/main.py)
* High-concurrency REST API for predictions
* Stores incoming data securely into Neon PostgreSQL
* Hardened database connection pooling to prevent cloud lockouts

### 🔹 4. Streamlit UI (streamlit_app.py)
Two main dashboards:

**🟢 Prediction Dashboard**
* Interactive transaction input
* Dynamic "AI Sensitivity" Threshold Slider
* Real-time SHAP Explainability Waterfall chart

**📉 Drift Monitoring Dashboard**
* Inject synthetic drift data
* Generate full Evidently AI HTML reports
* Trigger MLflow retraining

### 🔹 5. Database Layer (db.py)
* **[UPGRADE]** Migrated from local SQLite to Neon PostgreSQL via SQLAlchemy
* Implements explicit try-except-rollback handling
* Stores input data, predictions, and human-verified Actual_Class labels

### 🔹 6. Drift Detection (drift.py)
* Compares Training baseline vs. Live Database telemetry
* Uses Evidently AI to generate comprehensive drift visuals

### 🔹 7. Retraining Pipeline (retrain_pipeline.py)
* **[UPGRADE]** Human-in-the-Loop workflow (waits for analyst verification)
* Pulls fresh, verified data from Neon DB
* Retrains model, updates DagsHub registry, and hot-swaps local artifacts

### 🔹 8. Explainability (shap_explainer.py)
* Uses TreeExplainer for lightning-fast interpretation
* Visualizes exact feature contributions pushing the model toward Safe/Fraud

---

## 🔄 End-to-End Flow

    User Input → Streamlit (HF Spaces)
            ↓
    FastAPI HTTP Call (Render)
            ↓
    Ensemble Prediction + SHAP Computation
            ↓
    Log to PostgreSQL Cloud DB
            ↓
    Human Analyst Updates Ground Truth
            ↓
    Drift Detection Alert
            ↓
    Trigger MLflow Retraining

---

## 📈 Evaluation Metrics

* **F1-Score** (Primary focus due to high class imbalance)
* **Precision** (Minimizing False Positives)
* **Recall** (Minimizing False Negatives)
> Accuracy is avoided as an evaluation metric due to the 99.8% normal class dominance.

---

## 🛠️ Tech Stack

* **Languages:** Python
* **ML Frameworks:** Scikit-learn, XGBoost, Imbalanced-learn
* **Backend:** FastAPI, Uvicorn, SQLAlchemy
* **Frontend:** Streamlit
* **MLOps & XAI:** MLflow, DagsHub, Evidently AI, SHAP
* **Database & Cloud:** PostgreSQL (Neon), Render, Hugging Face

---

## 🧪 How to Run Locally

### 1️⃣ Clone Repo
    git clone https://github.com/MohitParmar78/adaptive-fraud-detection-mlops.git
    cd adaptive-fraud-detection-mlops

### 2️⃣ Environment Setup
Create a .env file in the root directory:

    DATABASE_URL="postgresql://user:pass@your-neon-url/dbname?sslmode=require"
    API_URL="http://127.0.0.1:8000/predict"
    MLFLOW_TRACKING_URI="https://dagshub.com/YourUsername/repo.mlflow"
    DAGSHUB_USER_TOKEN="your_token"

### 3️⃣ Create Python Environment

    conda create -n fraud-env python=3.12
    conda activate fraud-env
    pip install -r requirements.txt

### 4️⃣ Train Initial Model

    python -m src.pipeline.train_pipeline

### 5️⃣ Run Microservices

Terminal 1 (Backend):

    uvicorn app.main:app --reload

Terminal 2 (Frontend):

    streamlit run app/streamlit_app.py

---

## 🌐 Deployment Guide

### 🚀 Deploy Backend (Render)

1. Go to https://render.com and create a Web Service.
2. Connect GitHub repo.
3. Add Environment Variables (Copy from your .env file).
4. Build Settings:
    Build Command: pip install -r requirements.txt
    Start Command: uvicorn app.main:app --host 0.0.0.0 --port 10000

### 🚀 Deploy UI (Hugging Face Spaces)

1. Go to https://huggingface.co/spaces and create a Streamlit Space.
2. Add all requirements.txt and project files.
3. Add Environment Secrets in the Space Settings.
4. IMPORTANT: Update API_URL secret to point to your new live Render URL:
    API_URL: https://your-render-url.onrender.com/predict

---

## 📊 Drift Monitoring

* Uses live cloud DB telemetry.
* Compares with the original training distribution.
* Detects hacker behavior shifts and generates a visual HTML report within Streamlit.

---

## 🔁 Retraining Strategy (Human-in-the-Loop)

* Triggered manually via the UI monitoring tab.
* Does not train on blind model guesses. Wait for Actual_Class DB column to be updated by a verified human analyst.
* Updates model artifacts locally and versions them securely in MLflow.

---

## 🚀 Future Improvements

* [x] Migrate to Cloud PostgreSQL
* [x] Implement MLflow Model Versioning
* [ ] Automated retraining CRON scheduler
* [ ] Full Docker Compose integration for local testing

---

## 🏆 Key Highlights

* True Decoupled Microservices Architecture
* Production-grade Database Engineering
* Explainable AI Integration (Business Logic alignment)
* Full Continuous Training (CT) MLOps loop

---

## 👨‍💻 Author

**Mohit Parmar** Data Scientist & MLOps Engineer

---

## ⭐ If you like this project
Give it a ⭐ on GitHub to support the development of open-source MLOps architecture!
