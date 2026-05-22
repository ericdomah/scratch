import torch
import time
import numpy as np
from ensemble_model import GridGuardUniversalHybrid
from xgboost_model import XGBoostBaseline
import os

def measure_latency():
    device = torch.device('cpu') # Measuring on CPU for realistic edge/utility server baseline
    print(f"Benchmarking Inference Latency on {device}...")
    
    # 1. Deep Learning Model (Ensemble)
    model = GridGuardUniversalHybrid(window_size=26, input_dim=2, hidden_dim=64).to(device)
    model_path = "best_model_balanced.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[+] Loaded production weights: {model_path}")
    else:
        print("[!] Production weights not found! Running randomly initialized model.")
    model.eval()
    
    dummy_input = torch.randn(1, 26, 2).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(dummy_input)
    
    # Timing
    start_time = time.perf_counter()
    iterations = 500
    for _ in range(iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    end_time = time.perf_counter()
    
    avg_latency_dl = (end_time - start_time) / iterations * 1000 # ms
    print(f"GridGuard Meta-Ensemble Latency: {avg_latency_dl:.4f} ms")

    # 2. XGBoost Baseline
    xgb = XGBoostBaseline()
    xgb_path = "best_xgb_augmented.pkl"
    if os.path.exists(xgb_path):
        xgb.load_model(xgb_path)
        print(f"[+] Loaded production XGBoost model: {xgb_path}")
    else:
        # Mock data for XGBoost (usually expects flattened features) if not found
        print("[!] Production XGBoost not found, fitting dummy XGBoost baseline...")
        xgb_input = np.random.randn(100, 26)
        xgb_labels = np.random.randint(0, 2, 100)
        xgb.train(torch.tensor(xgb_input), torch.tensor(xgb_labels))
    
    # For XGBoost, let's use the real 2D input flattened if needed, or shape expected by predict_proba
    xgb_dummy = np.random.randn(1, 26, 2)
    
    # Warmup
    for _ in range(10):
        _ = xgb.predict_proba(xgb_dummy)
        
    start_time = time.perf_counter()
    iterations = 1000 # Increase iterations for more precision
    for _ in range(iterations):
        _ = xgb.predict_proba(xgb_dummy)
    end_time = time.perf_counter()
    
    avg_latency_xgb = (end_time - start_time) / iterations * 1000 # ms
    print(f"XGBoost Baseline Latency: {avg_latency_xgb:.4f} ms")

    print("-" * 40)
    print(f"Correction Report: XGBoost is actually ~{avg_latency_xgb:.4f} ms, not 0.003 ms.")

if __name__ == "__main__":
    measure_latency()
