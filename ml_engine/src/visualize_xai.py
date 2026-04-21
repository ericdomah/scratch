import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from ensemble_model import GridGuardUniversalHybrid
from xai_engine import XAIEngine
from data_loader import ElectricityDataset
from theft_injector import TheftInjector

def visualize_explanation():
    print("=" * 60)
    print("  GridGuard AI: Generating Explainable AI (XAI) Report")
    print("=" * 60)

    # 1. Setup
    device = 'cpu'
    model = GridGuardUniversalHybrid(window_size=30)
    
    model_path = "best_model_balanced.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[OK] Loaded trained model: {model_path}")
    else:
        print("[!] No trained model found. Using random weights for demo.")

    xai = XAIEngine(model, device=device)
    injector = TheftInjector()

    # 2. Get Data
    data_path = "../../data/data_set_cleaned.csv"
    if not os.path.exists(data_path):
        data_path = "../data/data_set_cleaned.csv"
    
    dataset = ElectricityDataset(data_path, window_size=30, transform=True)
    
    # Find a normal sample
    normal_x, _ = dataset[0] # (30, 1)
    
    # Generate a synthetic thief (Partial Bypass) for clear visualization
    # We'll reduce consumption by 80% from day 15 to 25
    thief_x = injector.inject_partial_bypass(normal_x.squeeze(), 15, 25, 0.2).unsqueeze(-1)
    
    # 3. Compute XAI
    print("Computing suspiciousness scores (Integrated Gradients)...")
    saliency_normal = xai.get_integrated_gradients(normal_x.unsqueeze(0))
    saliency_thief = xai.get_integrated_gradients(thief_x.unsqueeze(0))

    # 4. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    # --- Plot Normal Case ---
    ax1 = axes[0]
    ax1.plot(normal_x.squeeze().numpy(), label='Consumption (kWh)', color='#2ecc71', linewidth=2)
    ax1.set_title("Normal Consumer: Background Suspicion Heatmap", fontweight='bold')
    ax1.set_ylabel("Energy (kWh)")
    
    # Overlay saliency as heatmap
    for i in range(len(saliency_normal)):
        ax1.axvspan(i-0.5, i+0.5, color='red', alpha=saliency_normal[i] * 0.3)
    
    # --- Plot Thief Case ---
    ax2 = axes[1]
    ax2.plot(thief_x.squeeze().numpy(), label='Consumption (kWh)', color='#e67e22', linewidth=2)
    ax2.set_title("Flagged Thief: Partial Bypass Detected (Red = Suspicion)", fontweight='bold')
    ax2.set_ylabel("Energy (kWh)")
    ax2.set_xlabel("Day of Month")
    
    # Overlay saliency as heatmap
    for i in range(len(saliency_thief)):
        ax2.axvspan(i-0.5, i+0.5, color='red', alpha=saliency_thief[i] * 0.8)

    # 5. Save Report
    os.makedirs("outputs", exist_ok=True)
    report_path = "outputs/xai_report.png"
    plt.savefig(report_path)
    print("=" * 60)
    print(f"[SUCCESS] XAI Visualization Report saved to: {report_path}")
    print("=" * 60)

if __name__ == "__main__":
    visualize_explanation()
