# 🚀 Adaptive Real-Time Fraud Detection System with MLOps

---

## 📌 Overview

This project is an **end-to-end production-ready fraud detection system** that combines:

* Real-time prediction
* Model explainability (SHAP)
* Data drift monitoring
* Automated retraining pipeline

It simulates how modern financial systems detect fraud dynamically and adapt to changing patterns.

---

## 🎯 Key Features

✅ Real-time fraud prediction via API
✅ Interactive UI using Streamlit
✅ Explainable AI using SHAP
✅ Data drift detection using Evidently AI
✅ Automated retraining pipeline
✅ Database-based data collection (SQLite)
✅ End-to-end MLOps workflow

---

## 📊 Dataset

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> ⚠️ Due to size limits, download manually and place inside `data/`

### Key Details

* 284,807 transactions
* 492 fraud cases
* Highly imbalanced

### Features

* Time
* Amount
* V1–V28 (PCA transformed)
* Class (0 = Normal, 1 = Fraud)

---

## 🧠 System Architecture

```
User (Streamlit UI)
        ↓
FastAPI Backend (REST API)
        ↓
ML Model (RF + XGBoost Ensemble)
        ↓
SQLite Database (stores live data)
        ↓
Monitoring Layer (Drift Detection)
        ↓
Retraining Pipeline
```

---

## ⚙️ Project Modules

### 🔹 1. Data Pipeline (`train_pipeline.py`)

* Data loading & preprocessing
* Feature scaling
* SMOTE for imbalance handling
* Model training (Random Forest + XGBoost)
* Hyperparameter tuning
* Saves models in `artifacts/`

---

### 🔹 2. Prediction Pipeline (`predict_pipeline.py`)

* Loads trained models
* Applies preprocessing
* Performs ensemble prediction
* Returns:

  * Prediction (0/1)
  * Probability

---

### 🔹 3. FastAPI Backend (`app/main.py`)

* REST API for predictions
* Stores incoming data into SQLite DB
* Acts as bridge between UI and model

---

### 🔹 4. Streamlit UI (`streamlit_app.py`)

Two main dashboards:

#### 🟢 Prediction Dashboard

* User input form
* Fraud prediction
* Risk visualization
* SHAP explainability

#### 📉 Drift Monitoring Dashboard

* Run drift detection
* View drift report
* Trigger model retraining

---

Note: Drift detection and retraining require persistent storage. On Streamlit Cloud the prediction and SHAP explainability features are fully functional; drift/retrain are demonstrated locally via SQLite.

### 🔹 5. Database Layer (`db.py`)

* SQLite-based storage
* Stores:

  * Input data
  * Predictions
  * Probabilities

---

### 🔹 6. Drift Detection (`drift.py`)

* Compares:

  * Training data vs Live DB data
* Uses Evidently AI
* Generates HTML report

---

### 🔹 7. Retraining Pipeline (`retrain_pipeline.py`)

* Loads DB data
* Converts predictions → labels
* Retrains model
* Updates artifacts

---

### 🔹 8. Explainability (`shap_explainer.py`)

* Uses SHAP
* Provides feature-level contribution
* Visualized in Streamlit

---

## 🔄 End-to-End Flow

```
User Input → Streamlit
        ↓
FastAPI API Call
        ↓
Prediction
        ↓
Store in DB
        ↓
Drift Detection
        ↓
Retraining Trigger
        ↓
Updated Model
```

---

## 📈 Evaluation Metrics

* Precision
* Recall
* F1-score

> Accuracy is avoided due to class imbalance.

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn, XGBoost
* FastAPI
* Streamlit
* SHAP
* Evidently AI
* SQLite

---

## 🧪 How to Run Locally

### 1️⃣ Clone Repo

```bash
git clone <your-repo-url>
cd fraud-detection-mlops
```

---

### 2️⃣ Create Environment

```bash
conda create -n fraud-env python=3.10
conda activate fraud-env
pip install -r requirements.txt
```

---

### 3️⃣ Train Model

```bash
python -m src.pipeline.train_pipeline
```

---

### 4️⃣ Run FastAPI

```bash
uvicorn app.main:app --reload
```

---

### 5️⃣ Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

---

## 🌐 Deployment Guide

---

# 🚀 Deploy Backend (Render)

### Steps:

1. Go to https://render.com
2. Create **Web Service**
3. Connect GitHub repo
4. Add settings:

```
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port 10000
```

---

# 🚀 Deploy UI (Streamlit Cloud)

### Steps:

1. Go to https://share.streamlit.io
2. Connect repo
3. Select:

```
app/streamlit_app.py
```

---

### ⚠️ IMPORTANT

Update API URL in Streamlit:

```python
response = requests.post(
    "https://your-render-url/predict",
    json=data
)
```

---

## 📊 Drift Monitoring

* Uses live DB data
* Compares with training dataset
* Detects distribution changes
* Generates visual report

---

## 🔁 Retraining Strategy

* Triggered manually via UI
* Uses new collected data
* Updates model artifacts

> ⚠️ Note: Uses predicted labels (demo purpose)

---

## 🚀 Future Improvements

* Real label feedback system
* Automated retraining scheduler
* Model versioning (MLflow)
* CI/CD pipeline
* Docker + Kubernetes

---

## 🏆 Key Highlights

* Full MLOps pipeline
* Real-time system design
* Explainable AI integration
* Monitoring + retraining loop

---

## 👨‍💻 Author

**Mohit Rajput**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
