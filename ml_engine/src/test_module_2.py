import torch
from data_loader import ElectricityDataset
from theft_injector import TheftInjector
import os

def test_loading_and_injection():
    csv_path = "c:/Users/eric.domah/.gemini/antigravity/scratch/data/datasetsmall.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return

    # 1. Load Dataset
    print("--- Testing Data Loading ---")
    dataset = ElectricityDataset(csv_path, window_size=20)
    x, y = dataset[0]
    print(f"Loaded Sample - Shape: {x.shape}, Label: {y.item()}")
    print(f"Original Data (first 5 readings):\n{x[:5].flatten()}")

    # 2. Test Injection
    print("\n--- Testing Theft Injection (Pattern 1: Constant Reduction) ---")
    injector = TheftInjector()
    x_theft = injector.inject_constant_reduction(x, alpha=0.1)
    print(f"Injected Data (first 5 readings, should be 10% of original):\n{x_theft[:5].flatten()}")

    # 3. Test Partial Bypass
    print("\n--- Testing Theft Injection (Pattern 2: Partial Bypass) ---")
    x_partial = injector.inject_partial_bypass(x, start_idx=5, end_idx=15, alpha=0.0)
    print(f"Partial Bypass (readings 5 to 15 should be 0):")
    print(x_partial[4:16].flatten())

    print("\n✅ Module 2 Verification Complete!")

if __name__ == "__main__":
    test_loading_and_injection()
