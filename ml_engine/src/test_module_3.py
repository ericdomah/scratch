import torch
from data_loader import ElectricityDataset
import os

def test_preprocessing():
    csv_path = "c:/Users/eric.domah/.gemini/antigravity/scratch/data/datasetsmall.csv"
    
    # 1. Load with Preprocessing
    print("--- Testing Module 3: Preprocessing Pipeline ---")
    dataset = ElectricityDataset(csv_path, window_size=20, transform=True)
    x, y = dataset[0]
    
    print(f"Sample Shape: {x.shape}")
    print(f"Label: {y.item()}")
    print(f"Min Value: {x.min().item():.4f}")
    print(f"Max Value: {x.max().item():.4f}")
    
    # Check if values are in [0, 1] range due to Min-Max scaling
    assert 0.0 <= x.min().item() <= 1.0, "Scaling failed (min)"
    assert 0.0 <= x.max().item() <= 1.0, "Scaling failed (max)"
    
    print("\n✅ Module 3 Verification Complete! Data is cleaned, clipped, and normalized.")

if __name__ == "__main__":
    test_preprocessing()
