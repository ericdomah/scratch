import torch
from torch.utils.data import DataLoader
from data_loader import ElectricityDataset
from ensemble_model import GridGuardUniversalHybrid
from xgboost_model import XGBoostBaseline
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

def evaluate_pytorch_model(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).float()
            logits = model(batch_x)
            probs = torch.sigmoid(logits).view(-1)
            preds = (probs > 0.5).int().cpu().numpy()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
    return np.array(all_probs), np.array(all_preds), np.array(all_labels)

def run_comparative_study():
    print("--- GridGuard AI: Advanced Thesis Benchmarking ---")
    
    # 1. Load Data
    print("Loading Evaluation Dataset...")
    DATA_PATH = "../../data/data set.csv"
    if not os.path.exists(DATA_PATH):
        DATA_PATH = "../data/data set.csv"
        
    dataset = ElectricityDataset(DATA_PATH, window_size=30, transform=True)
    
    # Use a larger subset for evaluation (5000 samples) to get stable curves
    eval_size = min(5000, len(dataset))
    indices = np.random.choice(len(dataset), eval_size, replace=False)
    eval_ds = torch.utils.data.Subset(dataset, indices)
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False)
    
    # Extract data for XGBoost evaluation
    X_eval_list, y_eval_list = zip(*[eval_ds[i] for i in range(len(eval_ds))])
    X_eval = torch.stack(X_eval_list)
    y_eval = torch.stack(y_eval_list)
    y_eval_np = y_eval.cpu().numpy()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    probs_dict = {}
    preds_dict = {}

    # --- 1. Load XGBoost Vigorous ---
    print(">> Evaluating XGBoost Baseline...")
    xgb_model = XGBoostBaseline()
    xgb_path = "best_xgb_vigorous.pkl"
    if not os.path.exists(xgb_path):
        xgb_path = "best_xgb.pkl"
        
    if os.path.exists(xgb_path):
        xgb_model.load_model(xgb_path)
        xgb_probs = xgb_model.predict_proba(X_eval)
        xgb_preds = (xgb_probs > 0.5).astype(int)
        probs_dict["XGBoost Baseline"] = xgb_probs
        preds_dict["XGBoost Baseline"] = xgb_preds
    else:
        print("[SKIP] XGBoost weights not found.")

    # --- 2. Load Super-Hybrid (Balanced Senior) ---
    print(">> Evaluating Balanced Senior Engineer Model...")
    hybrid_model = GridGuardUniversalHybrid().to(device)
    # Priority order: Balanced > Vigorous > Baseline
    weight_paths = ["best_model_balanced.pth", "best_model_vigorous.pth", "best_model.pth"]
    loaded = False
    for wp in weight_paths:
        if os.path.exists(wp):
            print(f"Loading weights from: {wp}")
            hybrid_model.load_state_dict(torch.load(wp, map_location=device))
            hybrid_probs, hybrid_preds, _ = evaluate_pytorch_model(hybrid_model, eval_loader, device)
            probs_dict["GridGuard Super-Hybrid"] = hybrid_probs
            preds_dict["GridGuard Super-Hybrid"] = hybrid_preds
            loaded = True
            break
    
    if not loaded:
        print("[SKIP] No Super-Hybrid weights found.")

    # --- 3. Meta-Ensemble Fusion ---
    if "XGBoost Baseline" in probs_dict and "GridGuard Super-Hybrid" in probs_dict:
        print(">> Generating Meta-Ensemble (Hybrid Fusion)...")
        ensemble_probs = (probs_dict["XGBoost Baseline"] * 0.3) + (probs_dict["GridGuard Super-Hybrid"] * 0.7)
        probs_dict["Meta-Ensemble (Final)"] = ensemble_probs
        
        # Apply Threshold Search for Meta-Ensemble
        from balanced_senior_train import find_optimal_threshold
        t, f1, p, r = find_optimal_threshold(ensemble_probs, y_eval_np, min_recall=0.60)
        print(f"Optimal Ensemble Threshold: {t:.4f} (F1: {f1:.4f}, Prec: {p:.4f}, Rec: {r:.4f})")
        preds_dict["Meta-Ensemble (Final)"] = (ensemble_probs > t).astype(int)

    if not probs_dict:
        print("!!! ERROR: No trained models found. Please run 'python balanced_senior_train.py' first.")
        return

    print("\nGenerating Thesis Graphic Suite...")
    sns.set_theme(style="whitegrid")
    
    # 1. Confusion Matrices
    fig, axes = plt.subplots(1, len(preds_dict), figsize=(6 * len(preds_dict), 5))
    if len(preds_dict) == 1: axes = [axes]
    for i, (name, preds) in enumerate(preds_dict.items()):
        cm = confusion_matrix(y_eval_np, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Theft"])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(f"{name}\n(Balanced Strategy)", fontsize=12)
        axes[i].grid(False)
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrices.png", dpi=300)
    plt.close()

    # 2. Precision-Recall Curves with Optimal Point
    plt.figure(figsize=(10, 8))
    for name, probs in probs_dict.items():
        precision, recall, _ = precision_recall_curve(y_eval_np, probs)
        ap = average_precision_score(y_eval_np, probs)
        plt.plot(recall, precision, lw=3 if "Ensemble" in name else 2, label=f'{name} (AP = {ap:.3f})')
        
        if "Ensemble" in name:
            # Highlight the Optimal Threshold point from the Senior Engineer Search
            from balanced_senior_train import find_optimal_threshold
            _, _, opt_p, opt_r = find_optimal_threshold(probs, y_eval_np, min_recall=0.60)
            plt.plot(opt_r, opt_p, 'ro', markersize=10, label=f'Optimal Deployment Point\n(Prec: {opt_p:.2f}, Rec: {opt_r:.2f})')

    plt.xlabel('Recall (Detection Rate)', fontsize=12)
    plt.ylabel('Precision (Inspection Accuracy)', fontsize=12)
    plt.title('Precision-Recall Frontier: Senior Engineer Balanced Mode', fontsize=15)
    plt.legend(loc="upper right", fontsize=10)
    plt.savefig("outputs/pr_curve_comparison.png", dpi=300)
    plt.close()

    # 3. ROC Curves
    plt.figure(figsize=(10, 8))
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(y_eval_np, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Benchmarking')
    plt.legend(loc="lower right")
    plt.savefig("outputs/roc_curve_comparison.png", dpi=300)
    plt.close()

    print(">> Script Complete! All PNG graphics exported to 'ml_engine/src/outputs/'.")

if __name__ == "__main__":
    run_comparative_study()
