import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.pipeline.predict_pipeline import PredictPipeline
from src.explanability.shap_explainer import ShapExplainer

def test_shap_explainer():
    print("🔹 Step 1: Initializing models and explainer...")
    pipeline = PredictPipeline()
    explainer = ShapExplainer()

    # Create a dummy transaction (same as our predict test)
    data = {"Time": 10000.0, "Amount": 150.0}
    for i in range(1, 29):
        data[f"V{i}"] = 0.05 
    
    input_df = pd.DataFrame([data])

    print("🔹 Step 2: Preprocessing data (testing column order & scaler)...")
    # We use the pipeline to ensure 'Amount' gets flattened and columns are ordered
    processed_df = pipeline.preprocess(input_df)

    print("🔹 Step 3: Generating SHAP values (testing TreeExplainer)...")
    shap_values = explainer.explain(processed_df)
    
    print(f"✅ Success! SHAP values generated.")
    print(f"   - Expected Shape: (1, 30)")
    print(f"   - Actual Shape: {shap_values.shape}")
    print(f"   - Base Value: {shap_values.base_values[0]:.4f}")

    print("🔹 Step 4: Rendering waterfall chart...")
    # We save it to a file instead of showing it so the terminal doesn't freeze
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    
    # Save the plot to your main folder
    plt.tight_layout()
    plt.savefig("shap_test_output.png", dpi=300)
    print("✅ Chart saved successfully as 'shap_test_output.png'!")

if __name__ == "__main__":
    test_shap_explainer()