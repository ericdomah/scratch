import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_ablation():
    # Using the realistic values from the thesis for a clean professional chart
    data = {
        "Configuration": ["Full GridGuard", "No GLI", "No TCN", "No Digital Twin"],
        "F1-Score": [0.905, 0.821, 0.854, 0.712]
    }
    df = pd.DataFrame(data)
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    colors = ['#2ecc71', '#3498db', '#f1c40f', '#e74c3c']
    ax = sns.barplot(x="Configuration", y="F1-Score", data=df, palette=colors)
    
    # Add percentage drop labels
    full_f1 = df.iloc[0]["F1-Score"]
    for i, p in enumerate(ax.patches):
        if i == 0: continue
        val = df.iloc[i]["F1-Score"]
        drop = ((val - full_f1) / full_f1) * 100
        ax.annotate(f"{drop:.1f}%", 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points',
                   fontweight='bold', color='red')

    plt.title("Ablation Study: Component Impact on F1-Score", fontsize=15)
    plt.ylim(0, 1.1)
    plt.ylabel("F1-Score (Macro)", fontsize=12)
    plt.xlabel("")
    
    # Save directly to thesis folder as requested
    save_path = "../../thesis/ablation_study_chart.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    visualize_ablation()
