import torch
import numpy as np
from ensemble_model import GridGuardUniversalHybrid
from xgboost_model import XGBoostBaseline
from preprocessing import DataPreprocessor
import os

class InferenceEngine:
    def __init__(self, dl_model_path='best_model_balanced.pth', xgb_model_path='best_xgb_augmented.pkl', device='cpu'):
        self.device = device
        self.preprocessor = DataPreprocessor()
        
        # 1. Initialize Universal Hybrid (DL)
        self.model = GridGuardUniversalHybrid()
        if os.path.exists(dl_model_path):
            try:
                self.model.load_state_dict(torch.load(dl_model_path, map_location=device))
                print(f"[OK] Universal Hybrid Model loaded from {dl_model_path}")
            except Exception as e:
                print(f"[WARN] Failed to load weights for DL model: {e}")
        else:
            print(f"[WARN] {dl_model_path} not found. Running with uninitialized hybrid weights.")
            
        self.model.to(device)
        self.model.eval()

        # 2. Initialize XGBoost (Baseline Hybrid Component)
        self.xgb_model = XGBoostBaseline()
        self.has_xgb = False
        if os.path.exists(xgb_model_path):
            try:
                self.xgb_model.load_model(xgb_model_path)
                self.has_xgb = True
                print(f"[OK] XGBoost Baseline loaded from {xgb_model_path}")
            except Exception as e:
                print(f"[WARN] Failed to load XGBoost model: {e}")

    def predict(self, raw_consumption):
        """
        Calculates a hybrid probability across all model architectures.
        """
        # Preprocess
        processed_data = self.preprocessor.process_user_data(raw_consumption)
        
        # DL Inference (Universal Hybrid)
        input_tensor = torch.tensor(processed_data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            dl_logits = self.model(input_tensor)
            dl_prob = torch.sigmoid(dl_logits).item()
            
        # ML Inference (XGBoost)
        xgb_prob = 0.5 # Default if not loaded
        if self.has_xgb:
            # Reshape for XGBoost (expects 2D: batch, features)
            xgb_prob = self.xgb_model.predict_proba(input_tensor.cpu().numpy())[0]
            
        # Hybrid Fusion (Weighted average: 70% Deep Learning, 30% XGBoost)
        # Deep learning handles sequential/seasonal patterns, XGB boosts feature-level anomalies
        hybrid_prob = (0.7 * dl_prob) + (0.3 * xgb_prob)
            
        # Use the optimal Meta-Ensemble threshold found in the SOTA comparative study
        prediction = 1 if hybrid_prob > 0.5270 else 0
        return {
            "is_theft": bool(prediction),
            "confidence": float(hybrid_prob),
            "components": {
                "deep_learning": float(dl_prob),
                "gradient_boosting": float(xgb_prob)
            },
            "raw_value_count": len(raw_consumption)
        }

if __name__ == "__main__":
    # Test Ensemble Inference
    engine = InferenceEngine()
    sample_data = np.random.rand(30) * 10
    result = engine.predict(sample_data)
    print(f"Hybrid Ensemble Result: {result}")
