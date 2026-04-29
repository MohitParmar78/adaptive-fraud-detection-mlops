from fastapi import FastAPI
import pandas as pd
import sys
import os
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from src.pipeline.predict_pipeline import PredictPipeline
from src.monitoring.db import save_to_db, init_db

# Initialize Cloud DB connection on startup
init_db()

app = FastAPI(title="Fraud Detection API")

# Load pipeline into memory once (prevents reloading massive models on every API call)
pipeline = PredictPipeline()

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running 🚀"}

@app.post("/predict")
def predict(data: dict):
    # 1. Define the exact column order the model was trained on
    expected_columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    # 2. Convert the incoming JSON into a Pandas DataFrame and FORCE the column order
    df = pd.DataFrame([data], columns=expected_columns)

    # 3. Get predictions using the properly ordered data
    pred, prob = pipeline.predict(df)

    # 4. Save to Cloud DB for future drift detection and retraining
    save_to_db(data, int(pred), float(prob))

    # Business Logic Layer
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