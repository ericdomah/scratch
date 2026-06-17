import torch
import numpy as np
import os
import sys
import yaml
from ensemble_model import GridGuardUniversalHybrid
from xgboost_model import XGBoostBaseline
from preprocessing import DataPreprocessor

# Configure pathing
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "gridguard"))
sys.path.append(os.path.join(ROOT_DIR, "gridguard", "backend"))
sys.path.append(os.path.join(ROOT_DIR, "gridguard_real_data"))

from scipy.stats import skew as scipy_skew

def compute_tabular_features(X_np: np.ndarray) -> np.ndarray:
    kwh = X_np[:, :, 0]
    gli = X_np[:, :, 1]
    var_kwh  = kwh.var(axis=1)
    skw_kwh  = np.apply_along_axis(scipy_skew, 1, kwh)
    mean_kwh = np.where(kwh.mean(axis=1) > 0, kwh.mean(axis=1), 1e-8)
    par_kwh  = kwh.max(axis=1) / mean_kwh
    mean_gli = gli.mean(axis=1)
    std_gli  = gli.std(axis=1)
    return np.column_stack([var_kwh, skw_kwh, par_kwh, mean_gli, std_gli]).astype(np.float32)

# Load config
CONFIG_PATH = os.path.join(ROOT_DIR, "gridguard", "config.yaml")
if not os.path.exists(CONFIG_PATH):
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
    def __init__(self, 
                 syn_dl_path='best_model_balanced.pth', 
                 syn_xgb_path='best_xgb_augmented.pkl',
                 real_dl_path='../../gridguard_real_data/models/gridguard_sgcc_best.pth',
                 real_xgb_path='../../gridguard_real_data/models/xgboost_sgcc_edge.pkl',
                 device='cpu'):
        self.device = device
        self.preprocessor = DataPreprocessor()
        self.gli_manager = GLIManager()
        
        input_dim = config["model"]["input_dim"] # 2
        window_size = config["model"]["seq_len"] # 26
        hidden_dim = config["model"]["hidden_dim"] # 64
        
        # 1. Initialize Universal Hybrid (DL) models
        from legacy_model import GridGuardUniversalHybridLegacy
        
        self.model_syn = GridGuardUniversalHybrid(window_size=window_size, input_dim=input_dim, hidden_dim=hidden_dim).to(device)
        self.model_real = GridGuardUniversalHybridLegacy(input_dim=input_dim, hidden_dim=64).to(device)
        
        # Load Synthetic DL
        if os.path.exists(syn_dl_path):
            try:
                self.model_syn.load_state_dict(torch.load(syn_dl_path, map_location=device))
                print(f"[OK] Synthetic DL Model loaded from {syn_dl_path}")
            except Exception as e:
                print(f"[WARN] Failed to load Synthetic DL model: {e}")
        self.model_syn.eval()

        # Load Real-World DL
        if os.path.exists(real_dl_path):
            try:
                self.model_real.load_state_dict(torch.load(real_dl_path, map_location=device))
                print(f"[OK] Real-World DL Model loaded from {real_dl_path}")
            except Exception as e:
                print(f"[WARN] Failed to load Real-World DL model: {e}")
        self.model_real.eval()

        # 2. Initialize XGBoost (Baseline Hybrid Component)
        self.xgb_syn = XGBoostBaseline()
        self.xgb_real = XGBoostBaseline()
        self.has_xgb_syn = False
        self.has_xgb_real = False
        
        if os.path.exists(syn_xgb_path):
            try:
                self.xgb_syn.load_model(syn_xgb_path)
                self.has_xgb_syn = True
                print(f"[OK] Synthetic XGBoost loaded from {syn_xgb_path}")
            except Exception as e:
                print(f"[WARN] Failed to load Synthetic XGBoost model: {e}")
                
        if os.path.exists(real_xgb_path):
            try:
                self.xgb_real.load_model(real_xgb_path)
                self.has_xgb_real = True
                print(f"[OK] Real-World XGBoost loaded from {real_xgb_path}")
            except Exception as e:
                print(f"[WARN] Failed to load Real-World XGBoost model: {e}")

    def predict(self, raw_consumption, meter_id="MTR_UNKNOWN", live_gli=None, live_gli_timestamp=None, hour_of_day=12, day_of_week=0, model_type="real_world"):
        """
        Calculates a context-aware hybrid probability across selected model architecture.
        """
        # Select active models
        dl_model = self.model_syn if model_type == "synthetic" else self.model_real
        xgb_model = self.xgb_syn if model_type == "synthetic" else self.xgb_real
        has_xgb = self.has_xgb_syn if model_type == "synthetic" else self.has_xgb_real
        
        # Preprocess consumption data
        processed_consumption = self.preprocessor.process_user_data(raw_consumption)
        
        # Enforce strict 26-week sequence length
        if len(processed_consumption) < 26:
            pad_len = 26 - len(processed_consumption)
            processed_consumption = np.pad(processed_consumption, (pad_len, 0), mode='edge')
        elif len(processed_consumption) > 26:
            processed_consumption = processed_consumption[-26:]
            
        req = PredictionRequest(
            meter_id=meter_id,
            kwh_sequence=processed_consumption.tolist(),
            live_gli=live_gli,
            live_gli_timestamp=live_gli_timestamp,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week
        )
        gli_val, gli_status = self.gli_manager.process_gli(req)
        
        trnc_mode = config.get("data", {}).get("trnc_mode", True)
        if trnc_mode:
            gli_base = 0.5 + 0.12 * np.sin(np.linspace(np.pi, 5 * np.pi, 26))
        else:
            gli_base = 0.5 + 0.12 * np.sin(np.linspace(0, 4 * np.pi, 26))
            
        shift = gli_val - gli_base[-1]
        gli_seq = np.clip(gli_base + shift, 0.0, 1.0)
        
        seq_2d = np.stack([processed_consumption, gli_seq], axis=1) # (26, 2)
        
        # DL Inference
        input_tensor = torch.tensor(seq_2d, dtype=torch.float32).unsqueeze(0).to(self.device) # (1, 26, 2)
        with torch.no_grad():
            dl_logits = dl_model(input_tensor)
            dl_prob = torch.sigmoid(dl_logits).item()
            
        # ML Inference (XGBoost)
        xgb_prob = 0.5 # Default if not loaded
        if has_xgb:
            if model_type == "real_world":
                # Real-world model requires tabular extraction
                feats = compute_tabular_features(input_tensor.cpu().numpy())
                # xgb_model.model is raw sklearn wrapper since we loaded dict payload
                # we just use predict_proba directly on the raw model
                if hasattr(xgb_model, 'model'):
                    p = xgb_model.model.predict_proba(feats)
                    xgb_prob = p[0, 1] if p.shape[1] > 1 else p[0, 0]
            else:
                # Synthetic model expects flattened tensor
                xgb_prob = xgb_model.predict_proba(input_tensor.cpu().numpy())[0]
            
        # Hybrid Fusion (Weighted average: 70% Deep Learning, 30% XGBoost)
        hybrid_prob = (0.7 * dl_prob) + (0.3 * xgb_prob)
            
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
    engine = InferenceEngine()
    sample_data = np.random.rand(30) * 10
    result = engine.predict(sample_data, meter_id="MTR_TEST_CLI", model_type="real_world")
    print(f"Hybrid Ensemble Result (Real World): {result}")
