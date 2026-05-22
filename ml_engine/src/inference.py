import torch
import numpy as np
import os
import sys
import yaml
from ensemble_model import GridGuardUniversalHybrid
from xgboost_model import XGBoostBaseline
from preprocessing import DataPreprocessor

# Configure pathing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gridguard")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gridguard", "backend")))

# Load config
CONFIG_PATH = "c:/Users/User/Downloads/scratch-main/gridguard/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Import GLIManager and PredictionRequest
try:
    from backend.infrastructure.gli_manager import GLIManager, PredictionRequest
except ImportError:
    try:
        from gridguard.backend.infrastructure.gli_manager import GLIManager, PredictionRequest
    except ImportError:
        from gli_manager import GLIManager, PredictionRequest

class InferenceEngine:
    def __init__(self, dl_model_path='best_model_balanced.pth', xgb_model_path='best_xgb_augmented.pkl', device='cpu'):
        self.device = device
        self.preprocessor = DataPreprocessor()
        self.gli_manager = GLIManager()
        
        # 1. Initialize Universal Hybrid (DL) with input_dim=2 and window_size=26
        input_dim = config["model"]["input_dim"] # 2
        window_size = config["model"]["seq_len"] # 26
        hidden_dim = config["model"]["hidden_dim"] # 64
        
        self.model = GridGuardUniversalHybrid(window_size=window_size, input_dim=input_dim, hidden_dim=hidden_dim)
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

    def predict(self, raw_consumption, meter_id="MTR_UNKNOWN", live_gli=None, live_gli_timestamp=None, hour_of_day=12, day_of_week=0):
        """
        Calculates a context-aware hybrid probability across all model architectures.
        Incorporates 4-tier GLI Fallback logic and strict 26-timestep sequence length.
        """
        # Preprocess consumption data
        processed_consumption = self.preprocessor.process_user_data(raw_consumption)
        
        # Enforce strict 26-week sequence length (Fix 3: Sequence Window Contradiction)
        if len(processed_consumption) < 26:
            # Pad beginning with edge values
            pad_len = 26 - len(processed_consumption)
            processed_consumption = np.pad(processed_consumption, (pad_len, 0), mode='edge')
        elif len(processed_consumption) > 26:
            # Slice to most recent 26 timesteps
            processed_consumption = processed_consumption[-26:]
            
        # Retrieve the context-aware GLI value and degradation status from GLIManager
        req = PredictionRequest(
            meter_id=meter_id,
            kwh_sequence=processed_consumption.tolist(),
            live_gli=live_gli,
            live_gli_timestamp=live_gli_timestamp,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week
        )
        gli_val, gli_status = self.gli_manager.process_gli(req)
        
        # Synthesize GLI sequence with correct phase-alignment to summer cooling peaks
        trnc_mode = config.get("data", {}).get("trnc_mode", True)
        if trnc_mode:
            gli_base = 0.5 + 0.12 * np.sin(np.linspace(np.pi, 5 * np.pi, 26))
        else:
            gli_base = 0.5 + 0.12 * np.sin(np.linspace(0, 4 * np.pi, 26))
            
        # Shift baseline GLI curve such that the final timestep aligns with the live/estimated GLI value
        shift = gli_val - gli_base[-1]
        gli_seq = np.clip(gli_base + shift, 0.0, 1.0)
        
        # Stack to form a (26, 2) multi-channel tensor [Feature 0: kWh, Feature 1: GLI]
        seq_2d = np.stack([processed_consumption, gli_seq], axis=1) # (26, 2)
        
        # DL Inference (Universal Hybrid)
        input_tensor = torch.tensor(seq_2d, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, 26, 2)
        with torch.no_grad():
            dl_logits = self.model(input_tensor)
            dl_prob = torch.sigmoid(dl_logits).item()
            
        # ML Inference (XGBoost)
        xgb_prob = 0.5 # Default if not loaded
        if self.has_xgb:
            xgb_prob = self.xgb_model.predict_proba(input_tensor.cpu().numpy())[0]
            
        # Hybrid Fusion (Weighted average: 70% Deep Learning, 30% XGBoost)
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
            "gli_status": gli_status,
            "gli_value_used": float(gli_val),
            "raw_value_count": len(raw_consumption)
        }

if __name__ == "__main__":
    # Test Ensemble Inference
    engine = InferenceEngine()
    sample_data = np.random.rand(30) * 10
    result = engine.predict(sample_data, meter_id="MTR_TEST_CLI")
    print(f"Hybrid Ensemble Result: {result}")
