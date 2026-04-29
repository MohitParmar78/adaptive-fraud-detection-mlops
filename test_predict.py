import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline

def test_local_prediction():
    pipeline = PredictPipeline()

    # Define the EXACT order the model was trained on
    # Time, V1...V28, Amount
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    # Create data dictionary
    data = {
        "Time": 1000.0,
        "Amount": 50.0
    }
    for i in range(1, 29):
        data[f"V{i}"] = 0.01 

    # Create DataFrame and REORDER columns immediately
    test_df = pd.DataFrame([data])[cols]

    # Run prediction
    prediction, probability = pipeline.predict(test_df)

    print(f"--- Local Test Results ---")
    print(f"Result: {'Fraud' if prediction == 1 else 'Safe'}")
    print(f"Probability: {probability:.4f}")

if __name__ == "__main__":
    test_local_prediction()