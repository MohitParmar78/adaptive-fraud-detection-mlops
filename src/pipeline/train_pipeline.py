"""
Training Pipeline with Hyperparameter Tuning

Steps:
1. Load data
2. Split data
3. Scale features
4. Train Random Forest (baseline)
5. Tune XGBoost (GridSearchCV)
6. Save models + scaler
"""

import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


def train_pipeline(data_path: str):

    print("🔹 Step 1: Loading dataset...")
    df = pd.read_csv(data_path)

    # Split features and target
    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    print("🔹 Step 2: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("🔹 Step 3: Scaling 'Amount' feature...")
    scaler = StandardScaler()

    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])
    
    print("🔹 Step 4: Applying SMOTE (handle imbalance)...")
    smote = SMOTE(random_state=42)

    # ✅ Check class diversity
    if len(set(y_train)) > 1:
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    else:
        print("⚠️ Only one class present → skipping SMOTE")
        X_train_res, y_train_res = X_train, y_train

    print("🔹 Step 5: Training Random Forest (baseline)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_res, y_train_res)

    print("🔹 Step 6: Hyperparameter tuning XGBoost...")

    xgb = XGBClassifier(eval_metric="logloss", n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1]
    }

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="f1",   # important for fraud detection
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid.fit(X_train_res, y_train_res)

    best_xgb = grid.best_estimator_

    print("✅ Best XGBoost Parameters:", grid.best_params_)

    print("🔹 Step 7: Saving models and scaler...")

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    ARTIFACTS_PATH = os.path.join(BASE_DIR, "artifacts")

    os.makedirs(ARTIFACTS_PATH, exist_ok=True)

    joblib.dump(rf_model, os.path.join(ARTIFACTS_PATH, "rf_model.pkl"))
    joblib.dump(best_xgb, os.path.join(ARTIFACTS_PATH, "xgb_model.pkl"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_PATH, "scaler.pkl"))

    print("✅ Training completed successfully!")


if __name__ == "__main__":
    train_pipeline("data/creditcard.csv")