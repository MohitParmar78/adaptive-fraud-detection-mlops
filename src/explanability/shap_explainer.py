import shap
import joblib
import os
import pandas as pd

class ShapExplainer:

    def __init__(self):
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")

        self.model = joblib.load(os.path.join(ARTIFACTS_PATH, "xgb_model.pkl"))
        
        # 🔥 FIX: TreeExplainer is specifically built for tree-based models like XGBoost
        self.explainer = shap.TreeExplainer(self.model)

    def explain(self, data: pd.DataFrame):
        # Generate SHAP values for the given data
        shap_values = self.explainer(data)
        return shap_values