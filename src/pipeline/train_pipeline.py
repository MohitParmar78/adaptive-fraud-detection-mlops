"""
Training Pipeline with Hyperparameter Tuning and DagsHub MLflow Integration

Steps:
1. Load data
2. Split data
3. Scale features
4. Handle Imbalance (SMOTE)
5. Train Random Forest (baseline)
6. Tune XGBoost (GridSearchCV) with scale_pos_weight
7. Save models + scaler to MLflow and local artifacts
"""

import os
import joblib
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv

load_dotenv()

def train_pipeline(data_path: str):
    
    # --- NEW: Initialize DagsHub MLflow Tracking ---
    dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
    if dagshub_uri:
        import urllib.parse
        repo_owner, repo_name = urllib.parse.urlparse(dagshub_uri).path.strip('/').split('/')[:2]
        repo_name = repo_name.replace(".mlflow", "")
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    mlflow.set_experiment("Adaptive_Fraud_Detection")
    
    dagshub.init(repo_owner='MohitParmar78', repo_name='adaptive-fraud-detection-mlops', mlflow=True)

    with mlflow.start_run(run_name="Retrain_Ensemble_Weighted"):
        print("🔹 Step 1: Loading dataset...")
        df = pd.read_csv(data_path)
        X = df.drop("Class", axis=1)
        y = df["Class"]

        print("🔹 Step 2: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        print("🔹 Step 3: Scaling 'Amount' feature...")
        scaler = StandardScaler()
        X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
        X_test["Amount"] = scaler.transform(X_test[["Amount"]])
        
        print("🔹 Step 4: Applying SMOTE (handle imbalance)...")
        smote = SMOTE(random_state=42)
        if len(set(y_train)) > 1:
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            print("⚠️ Only one class present → skipping SMOTE")
            X_train_res, y_train_res = X_train, y_train

        print("🔹 Step 5: Training Random Forest (baseline)...")
        rf_model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_res, y_train_res)
        
        # 🔥 FIX: Extreme Class Imbalance Weighting
        # 284315 Normal / 492 Fraud ≈ 578. Forces XGBoost to prioritize catching fraud.
        print("🔹 Step 6: Hyperparameter tuning XGBoost...")
        
        xgb = XGBClassifier(eval_metric="logloss", scale_pos_weight=578, n_jobs=-1)
        
        # 🔥 PRODUCTION GRID: Now it will test 18 different combinations to find the best F1 score
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.05, 0.1]
        } 
        grid = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring="f1", cv=3, verbose=1, n_jobs=-1)
        grid.fit(X_train_res, y_train_res)
        best_xgb = grid.best_estimator_

        # Log to DagsHub
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_f1_score", grid.best_score_)

        print("🔹 Step 7: Saving models and scaler...")
        BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")
        os.makedirs(ARTIFACTS_PATH, exist_ok=True)

        joblib.dump(rf_model, os.path.join(ARTIFACTS_PATH, "rf_model.pkl"))
        joblib.dump(best_xgb, os.path.join(ARTIFACTS_PATH, "xgb_model.pkl"))
        joblib.dump(scaler, os.path.join(ARTIFACTS_PATH, "scaler.pkl"))

        # Save to remote registry
        mlflow.sklearn.log_model(best_xgb, "xgboost_fraud_model")
        mlflow.sklearn.log_model(rf_model, "rf_fraud_model")

if __name__ == "__main__":
    train_pipeline("data/creditcard.csv")