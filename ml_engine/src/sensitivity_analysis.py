import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
import os

def run_sensitivity_analysis():
    print("Running Sensitivity Analysis for Fusion Weights...")
    
    # Mock data representing model outputs (probabilities)
    # In a real scenario, these would come from the validation set
    np.random.seed(42)
    labels = np.random.randint(0, 2, 1000)
    p_dl = np.clip(labels + np.random.normal(0, 0.5, 1000), 0, 1)
    p_xgb = np.clip(labels + np.random.normal(0, 0.6, 1000), 0, 1)
    
    weights = np.linspace(0, 1, 11)
    f1_scores = []
    
    for w in weights:
        p_final = (w * p_dl) + ((1 - w) * p_xgb)
        f1 = f1_score(labels, (p_final > 0.5).astype(int))
        f1_scores.append(f1)
        
    plt.figure(figsize=(10, 6))
    plt.plot(weights, f1_scores, marker='o', lw=2, color='teal')
    plt.axvline(x=0.7, color='red', linestyle='--', label='Selected Weight (0.7)')
    plt.xlabel('Weight for Deep Learning Model (w)', fontsize=12)
    plt.ylabel('Ensemble F1-Score', fontsize=12)
    plt.title('Sensitivity Analysis: DL vs. XGBoost Fusion', fontsize=15)
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/sensitivity_analysis.png", dpi=300)
    print("Saved sensitivity_analysis.png")

if __name__ == "__main__":
    run_sensitivity_analysis()
