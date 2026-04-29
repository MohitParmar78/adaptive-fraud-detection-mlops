import sys
import os
import pandas as pd

# Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.monitoring.db import load_data_from_db
from src.pipeline.train_pipeline import train_pipeline

def retrain():
    print("🔁 Retraining started...")
    df = load_data_from_db()

    # 🔥 FIX: Prevent the "Echo Chamber"
    # We only keep rows where a human has verified the result and filled in Actual_Class
    verified_data = df.dropna(subset=['Actual_Class'])

    if len(verified_data) < 50:
        print("❌ Not enough human-verified data to retrain yet.")
        raise ValueError("Need at least 50 verified records to retrain safely.")

    # Drop the machine's old predictions, we only want the human's truth
    verified_data = verified_data.drop(["id", "prediction", "probability"], axis=1)
    verified_data = verified_data.rename(columns={"Actual_Class": "Class"})

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATA_PATH = os.path.join(BASE_DIR, "data")
    os.makedirs(DATA_PATH, exist_ok=True)
    file_path = os.path.join(DATA_PATH, "retrain_data.csv")

    verified_data.to_csv(file_path, index=False)

    # Pass the verified data to the training loop
    train_pipeline(file_path)
    print("✅ Retraining completed")