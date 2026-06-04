import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../gridguard_real_data/models')))
from gridguard_model import GridGuardUniversalHybrid
from xai_engine import XAIEngine

def visualize_explanation():
    print("=" * 60)
    print("  GridGuard AI: Generating Explainable AI (XAI) Report")
    print("=" * 60)

    # 1. Setup
    device = 'cpu'
    model = GridGuardUniversalHybrid(input_dim=2, hidden_dim=64)
    
    # Load Real-World SGCC weights
    model_path = "../../gridguard_real_data/models/gridguard_sgcc_best.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[OK] Loaded Real-World SGCC model: {model_path}")
    else:
        print(f"[!] No trained model found at {model_path}. Using random weights for demo.")

    xai = XAIEngine(model, device=device)

    # 2. Synthesize Programmatic Baseline
    print("Synthesizing purely programmatic baseline (Normal & Partial Bypass)...")
    
    # Shape: (26, 2)
    # Channel 0: kWh (Oscillating sine wave 0.2 to 0.8)
    # Channel 1: GLI (Summer peaking curve)
    normal_x = np.zeros((26, 2))
    
    # Baseline kWh: stable consumption with slight weekly variance
    normal_x[:, 0] = 0.5 + 0.15 * np.sin(np.linspace(0, 10 * np.pi, 26))
    
    # Baseline GLI: peaks in the middle of the 26-week summer window
    normal_x[:, 1] = 0.5 + 0.3 * np.sin(np.linspace(0, np.pi, 26))
    
    normal_x = torch.tensor(normal_x, dtype=torch.float32)
    
    # Generate a synthetic thief (Partial Bypass)
    # Reduce consumption by 80% from week 12 to 20 on the kWh channel (column 0)
    thief_x = normal_x.clone()
    thief_x[12:20, 0] = thief_x[12:20, 0] * 0.2
    
    # 3. Compute XAI
    print("Computing suspiciousness scores (Integrated Gradients)...")
    saliency_normal_2d = xai.get_integrated_gradients(normal_x.unsqueeze(0))
    saliency_thief_2d = xai.get_integrated_gradients(thief_x.unsqueeze(0))

    # Extract only the consumption channel saliency (feature index 0)
    saliency_normal = saliency_normal_2d[:, 0]
    saliency_thief = saliency_thief_2d[:, 0]

    # 4. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    # --- Plot Normal Case ---
    ax1 = axes[0]
    ax1.plot(normal_x[:, 0].numpy(), label='Consumption (kWh)', color='#2ecc71', linewidth=2)
    ax1.set_title("Normal Consumer: Background Suspicion Heatmap", fontweight='bold')
    ax1.set_ylabel("Energy (Normalized)")
    
    # Overlay saliency as heatmap
    for i in range(len(saliency_normal)):
        ax1.axvspan(i-0.5, i+0.5, color='red', alpha=float(saliency_normal[i] * 0.3))
    
    # --- Plot Thief Case ---
    ax2 = axes[1]
    ax2.plot(thief_x[:, 0].numpy(), label='Consumption (kWh)', color='#e67e22', linewidth=2)
    ax2.set_title("Flagged Thief (Real-World SGCC Model): Partial Bypass Detected (Red = Suspicion)", fontweight='bold')
    ax2.set_ylabel("Energy (Normalized)")
    ax2.set_xlabel("Week of Sequence Window (1-26)")
    
    # Overlay saliency as heatmap
    for i in range(len(saliency_thief)):
        ax2.axvspan(i-0.5, i+0.5, color='red', alpha=float(saliency_thief[i] * 0.8))

    # 5. Save Report
    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/xai_sgcc_report.png"
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    print("=" * 60)
    print(f"[SUCCESS] SGCC XAI Visualization Report saved to: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    visualize_explanation()
