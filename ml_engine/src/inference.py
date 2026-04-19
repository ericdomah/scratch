import torch
import numpy as np
from model import HybridLSTMTransformer
from preprocessing import DataPreprocessor
import os

class InferenceEngine:
    def __init__(self, model_path='best_model.pth', device='cpu'):
        self.device = device
        self.preprocessor = DataPreprocessor()
        
        # Initialize model with same hyperparameters
        self.model = HybridLSTMTransformer()
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠️ Warning: {model_path} not found. Using uninitialized model.")
            
        self.model.to(device)
        self.model.eval()

    def predict(self, raw_consumption):
        """
        Takes raw sequential consumption, preprocesses it, and returns theft probability.
        """
        # 1. Preprocess
        processed_data = self.preprocessor.process_user_data(raw_consumption)
        
        # 2. Preparation (Add batch and channel dimensions)
        # Expected shape: (1, seq_len, 1)
        input_tensor = torch.tensor(processed_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probability = torch.sigmoid(logits).item()
            
        prediction = 1 if probability > 0.5 else 0
        return {
            "is_theft": bool(prediction),
            "confidence": float(probability),
            "raw_value_count": len(raw_consumption)
        }

if __name__ == "__main__":
    # Test Inference
    engine = InferenceEngine(model_path='best_model.pth')
    
    # Simulate some raw data
    sample_data = np.random.rand(20) * 10
    result = engine.predict(sample_data)
    print(f"Inference Result: {result}")
    
    # Test with a "theft" pattern (constant low)
    theft_sample = np.ones(20) * 0.1
    result_theft = engine.predict(theft_sample)
    print(f"Theft Sample Result: {result_theft}")
