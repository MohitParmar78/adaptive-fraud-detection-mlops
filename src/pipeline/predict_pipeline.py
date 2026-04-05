"""
Prediction Pipeline

Loads trained models and scaler,
applies preprocessing,
and returns final prediction using ensemble.
"""

import joblib
import os
import numpy as np
import pandas as pd


class PredictPipeline:

    def __init__(self):
        
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")

        self.rf_model = joblib.load(os.path.join(ARTIFACTS_PATH, "rf_model.pkl"))
        self.xgb_model = joblib.load(os.path.join(ARTIFACTS_PATH, "xgb_model.pkl"))
        self.scaler = joblib.load(os.path.join(ARTIFACTS_PATH, "scaler.pkl"))
    

    def preprocess(self, data: pd.DataFrame):
        """
        Apply same preprocessing as training
        """
        data = data.copy()

        # Scale Amount column
        data["Amount"] = self.scaler.transform(data[["Amount"]])

        return data

    def predict(self, data: pd.DataFrame):
        """
        Make prediction using ensemble
        """

        data = self.preprocess(data)

        # Get probabilities
        rf_prob = self.rf_model.predict_proba(data)[:, 1]
        xgb_prob = self.xgb_model.predict_proba(data)[:, 1]

        # Ensemble (average)
        final_prob = (rf_prob + xgb_prob) / 2

        # Convert to class (threshold = 0.5)
        final_pred = (final_prob > 0.5).astype(int)

        return final_pred[0], final_prob[0]