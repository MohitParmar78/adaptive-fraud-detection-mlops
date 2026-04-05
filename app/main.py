from fastapi import FastAPI
import pandas as pd
import sys
import os

# ✅ Fix path FIRST
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pipeline.predict_pipeline import PredictPipeline
from src.monitoring.db import save_to_db, init_db

# Initialize DB
init_db()

app = FastAPI()

# Load pipeline once (efficient)
pipeline = PredictPipeline()


@app.get("/")
def home():
    return {"message": "Fraud Detection API is running 🚀"}


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    pred, prob = pipeline.predict(df)

# ✅ Save to DB
    save_to_db(data, int(pred), float(prob))

    # Decision logic
    if prob > 0.8:
        action = "🚫 Block Transaction"
    elif prob > 0.4:
        action = "⚠️ Flag for Review"
    else:
        action = "✅ Allow Transaction"

    return {
        "fraud_prediction": int(pred),
        "fraud_probability": float(prob),
        "recommended_action": action
}    
    
@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/model-info")
def model_info():
    return {
        "models": ["Random Forest", "XGBoost"],
        "ensemble": "average probability",
        "status": "active"
    }