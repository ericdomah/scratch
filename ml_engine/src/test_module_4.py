import torch
from model import HybridLSTMTransformer
from data_loader import ElectricityDataset
from torch.utils.data import DataLoader

def test_model_forward():
    print("--- Testing Module 4: ML Model Architecture ---")
    
    # 1. Initialize Model
    model = HybridLSTMTransformer(
        input_dim=1, 
        hidden_dim=64, 
        num_lstm_layers=1, 
        num_heads=4, 
        num_transformer_layers=1
    )
    print("[OK] Model initialized successfully.")

    # 2. Test with Dummy Batch
    dummy_input = torch.randn(16, 20, 1) # Batch of 16, Seq len 20, 1 feature
    output = model(dummy_input)
    print(f"[OK] Dummy batch forward pass shape: {output.shape}") # Should be (16, 1)
    
    assert output.shape == (16, 1), f"Unexpected output shape: {output.shape}"

    # 3. Test with Real Data Batch
    csv_path = "c:/Users/User/Downloads/scratch-main/data/datasetsmall.csv"
    dataset = ElectricityDataset(csv_path, window_size=20)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    x_real, y_real = next(iter(dataloader))
    print(f"[OK] Real data batch shape: {x_real.shape}")
    
    logits = model(x_real)
    print(f"[OK] Real data forward pass shape: {logits.shape}")
    
    assert logits.shape == (8, 1), f"Unexpected real data output shape: {logits.shape}"

    print("\n[SUCCESS] Module 4 Verification Complete! Model architecture is sound.")

if __name__ == "__main__":
    test_model_forward()
