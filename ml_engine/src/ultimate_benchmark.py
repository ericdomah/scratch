import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from context_aware_train import EnrichedGridDataset, GridGuardContextModel
from ensemble_model import GridGuardUniversalHybrid
from xgboost_model import XGBoostBaseline
import os

def run_ultimate_benchmark():
    print("=" * 60)
    print("  GRIDGUARD AI: ULTIMATE THESIS BENCHMARKING")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("outputs", exist_ok=True)

    # 1. Load the Enriched Dataset (The common testbed)
    data_path = "../../data/grid_simulated_dataset.csv"
    if not os.path.exists(data_path):
        data_path = "../data/grid_simulated_dataset.csv"
    
    dataset = EnrichedGridDataset(data_path)
    indices = np.random.choice(len(dataset), min(1000, len(dataset)), replace=False)
    test_subset = torch.utils.data.Subset(dataset, indices)
    
    # Extract data for evaluation
    dynamic_list, static_list, label_list = [], [], []
    for i in range(len(test_subset)):
        d, s, l = test_subset[i]
        dynamic_list.append(d)
        static_list.append(s)
        label_list.append(l.item())
    
    dynamic_tensor = torch.stack(dynamic_list).to(device)
    static_tensor = torch.stack(static_list).to(device)
    labels_np = np.array(label_list)

    # --- 2. Model Loading ---
    probs_dict = {}

    # Model A: Baseline XGBoost
    print(">> Evaluating Baseline XGBoost...")
    xgb = XGBoostBaseline()
    if os.path.exists("best_xgb_augmented.pkl"):
        xgb.load_model("best_xgb_augmented.pkl")
        # XGB expects (batch, seq_len * features) or similar
        # For simplicity, we use the consumption sequence from dynamic_tensor
        xgb_input = dynamic_tensor[:, :, 0].cpu().numpy()
        probs_dict["XGBoost Baseline"] = xgb.predict_proba(xgb_input)

    # Model B: Previous Hybrid (Non-Contextual)
    print(">> Evaluating Universal Hybrid DL...")
    hybrid = GridGuardUniversalHybrid().to(device)
    if os.path.exists("best_model_balanced.pth"):
        hybrid.load_state_dict(torch.load("best_model_balanced.pth", map_location=device))
        hybrid.eval()
        with torch.no_grad():
            # Universal Hybrid only expects consumption
            dl_input = dynamic_tensor[:, :, 0].unsqueeze(-1)
            probs = torch.sigmoid(hybrid(dl_input)).squeeze().cpu().numpy()
            probs_dict["Universal Hybrid DL"] = probs

    # Model C: NEW Context-Aware Model
    print(">> Evaluating Context-Aware GridGuard (Ours)...")
    context_model = GridGuardContextModel().to(device)
    if os.path.exists("best_context_aware_model.pth"):
        context_model.load_state_dict(torch.load("best_context_aware_model.pth", map_location=device))
        context_model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(context_model(dynamic_tensor, static_tensor)).squeeze().cpu().numpy()
            probs_dict["GridGuard Context-Aware"] = probs

    # --- 3. Visualization ---
    print(">> Generating Comparative Graphics...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 8))
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(labels_np, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=3 if "Context" in name else 2, 
                 label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate (Accusation Error)')
    plt.ylabel('True Positive Rate (Detection Rate)')
    plt.title('Final Benchmarking: GridGuard Context-Aware vs. Baselines', fontsize=15)
    plt.legend(loc="lower right")
    plt.savefig("outputs/final_roc_comparison.png", dpi=300)
    plt.close()

    # Confusion Matrix for Context-Aware Model
    if "GridGuard Context-Aware" in probs_dict:
        cm = confusion_matrix(labels_np, (probs_dict["GridGuard Context-Aware"] > 0.5).astype(int))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Theft"])
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap='YlGnBu', ax=ax)
        plt.title("Confusion Matrix: Context-Aware GridGuard", fontsize=14)
        plt.grid(False)
        plt.savefig("outputs/final_confusion_matrix.png", dpi=300)
        plt.close()

    print("=" * 60)
    print("[SUCCESS] Ultimate Benchmarks generated in ml_engine/src/outputs/")
    print("Files: final_roc_comparison.png, final_confusion_matrix.png")
    print("=" * 60)

if __name__ == "__main__":
    run_ultimate_benchmark()
