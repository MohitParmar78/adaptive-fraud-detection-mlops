import sys
import os
import pandas as pd

# ✅ Fix path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.monitoring.db import load_data_from_db
from src.pipeline.train_pipeline import train_pipeline

def retrain():

    print("🔁 Retraining started...")

    df = load_data_from_db()

    df = df.drop(["id"], axis=1)

    if "prediction" not in df.columns:
        print("❌ No labels available")
        return

    df = df.rename(columns={"prediction": "Class"})

    # ✅ Correct path
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    DATA_PATH = os.path.join(BASE_DIR, "data")

    os.makedirs(DATA_PATH, exist_ok=True)

    file_path = os.path.join(DATA_PATH, "retrain_data.csv")

    df.to_csv(file_path, index=False)

    # retrain using correct path
    train_pipeline(file_path)

    print("✅ Retraining completed")