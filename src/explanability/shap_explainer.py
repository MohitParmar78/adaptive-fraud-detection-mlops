import shap
import joblib
import os
import pandas as pd
import numpy as np


class ShapExplainer:

    def __init__(self):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")

        self.model = joblib.load(os.path.join(ARTIFACTS_PATH, "xgb_model.pkl"))
        
        '''
        #run this in local host
        
        # Background data
        background = pd.read_csv(os.path.join(BASE_DIR, "data/creditcard.csv")) \
                        .drop("Class", axis=1) \
                        .sample(100, random_state=42)
        '''

        # ✅ FIX: use known feature count (30)
        background = np.zeros((50, 30))

        # Wrap model
        def model_fn(X):
            return self.model.predict_proba(X)

        self.explainer = shap.KernelExplainer(model_fn, background)

    def explain(self, data: pd.DataFrame):
        shap_values = self.explainer.shap_values(data)
        return shap_values