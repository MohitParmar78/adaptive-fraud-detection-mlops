import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
import  sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.monitoring.db import load_data_from_db


def detect_drift(reference_path):

    # 🔹 Load reference (training data)
    reference_data = pd.read_csv(reference_path).drop("Class", axis=1)

    # 🔹 Load current data from DB
    current_data = load_data_from_db()

    # Remove non-feature columns
    current_data = current_data.drop(["id", "prediction", "probability"], axis=1)

    if len(current_data) < 50:
        print("⚠️ Not enough data for drift detection")
        return None

    # 🔹 Create report
    report = Report(metrics=[DataDriftPreset()])

    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    REPORT_PATH = os.path.join(BASE_DIR, "reports")

    os.makedirs(REPORT_PATH, exist_ok=True)

    path = os.path.join(REPORT_PATH, "drift_report.html")
    report.save_html(path)

    return path