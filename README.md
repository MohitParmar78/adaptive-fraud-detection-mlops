# 🚀 Adaptive Real-Time Fraud Detection System with MLOps

---

## 📌 Problem Statement

Fraudulent transactions are rare and constantly evolving, making traditional static models ineffective over time. This leads to significant financial losses in real-world systems.

---

## 🎯 Project Goal

To build an **end-to-end fraud detection system** that:

* Detects fraudulent transactions in real-time
* Handles imbalanced data effectively
* Monitors data drift
* Supports model retraining

---

## 📊 Dataset Description

This project uses the **Credit Card Fraud Detection dataset**, which contains real-world transaction data.

### 🔹 Key Details:

* ~284,000 transactions
* ~492 fraud cases
* Highly imbalanced dataset

### 🔹 Features:

* **Time** → Time elapsed between transactions
* **Amount** → Transaction value
* **V1–V28** → PCA-transformed features (privacy-preserved)
* **Class** → Target variable

  * 0 → Normal
  * 1 → Fraud

### ⚠️ Notes:

* Data is anonymized using PCA
* No missing values
* Suitable for real-world fraud detection scenarios

---

## ⚙️ Project Architecture

```
Streamlit UI → FastAPI Backend → ML Model → Monitoring
```

---

## 🧠 Approach

* Data preprocessing and scaling
* Handling imbalanced data
* Model training (Logistic Regression, Random Forest, XGBoost)
* Model evaluation using Precision, Recall, F1-score
* API deployment using FastAPI
* Monitoring using MLflow and drift detection

---

## 🛠️ Tech Stack

* Python, Pandas, NumPy
* Scikit-learn, XGBoost
* FastAPI
* Streamlit
* MLflow
* Evidently AI

---

## 📈 Evaluation Metrics

Due to class imbalance, the following metrics are used:

* Precision
* Recall
* F1-score

---

## 🚀 Future Improvements

* Automated retraining pipeline
* Advanced anomaly detection
* Improved explainability

---

## 👨‍💻 Author

Your Name
MOHIT"# adaptive-fraud-detection-mlops" 
