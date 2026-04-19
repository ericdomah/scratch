import torch
from torch.utils.data import DataLoader
from data_loader import ElectricityDataset
from model import HybridLSTMTransformer
from xgboost_model import XGBoostBaseline
from tft_model import SimplifiedTFT
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
import time
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
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds)
            all_labels.extend(batch_y.cpu().numpy())
    return np.array(all_probs).flatten(), np.array(all_preds).flatten(), np.array(all_labels).flatten()

def run_comparative_study():
    print("--- 🚀 GridGuard AI: Comprehensive Thesis Evaluation ---")
    
    print("Loading SGCC Dataset...")
    dataset = ElectricityDataset("../../data/datasetsmall.csv", window_size=30, transform=True)
    
    subset_size = min(1000, len(dataset))
    subset_ds = torch.utils.data.Subset(dataset, list(range(subset_size)))
    
    split_idx = int(0.8 * subset_size)
    train_dataset, test_dataset = torch.utils.data.random_split(subset_ds, [split_idx, subset_size - split_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    X_train_list, y_train_list = zip(*[subset_ds[i] for i in train_dataset.indices])
    X_train = torch.stack(X_train_list)
    y_train = torch.stack(y_train_list)
    
    X_test_list, y_test_list = zip(*[subset_ds[i] for i in test_dataset.indices])
    X_test = torch.stack(X_test_list)
    y_test = torch.stack(y_test_list)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    probs_dict = {}
    preds_dict = {}

    # --- 1. XGBoost ---
    print(">> Training XGBoost (Baseline)")
    xgb_model = XGBoostBaseline(n_estimators=100, max_depth=5)
    xgb_model.train(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)
    
    y_test_np = y_test.cpu().numpy()
    results["XGBoost"] = {"F1": f1_score(y_test_np, xgb_preds, zero_division=0)}
    probs_dict["XGBoost"] = xgb_probs
    preds_dict["XGBoost"] = xgb_preds

    # --- 2. Hybrid LSTM-Transformer ---
    print(">> Training Hybrid LSTM-Transformer (3 Epochs)")
    hybrid_model = HybridLSTMTransformer(input_dim=1, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    hybrid_loss_history = []
    hybrid_model.train()
    for _ in range(3):  
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = hybrid_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        hybrid_loss_history.append(epoch_loss / len(train_loader))
            
    hybrid_probs, hybrid_preds, hybrid_labels = evaluate_pytorch_model(hybrid_model, test_loader, device)
    results["Hybrid LSTM-Transformer"] = {"F1": f1_score(hybrid_labels, hybrid_preds, zero_division=0)}
    probs_dict["Hybrid LSTM-Transformer"] = hybrid_probs
    preds_dict["Hybrid LSTM-Transformer"] = hybrid_preds

    # --- 3. Simplified TFT ---
    print(">> Training Simplified TFT (3 Epochs)")
    tft_model = SimplifiedTFT(input_dim=1, hidden_dim=64).to(device)
    optimizer_tft = torch.optim.Adam(tft_model.parameters(), lr=1e-3)
    
    tft_loss_history = []
    tft_model.train()
    for _ in range(3): 
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
            optimizer_tft.zero_grad()
            out = tft_model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer_tft.step()
            epoch_loss += loss.item()
        tft_loss_history.append(epoch_loss / len(train_loader))
            
    tft_probs, tft_preds, tft_labels = evaluate_pytorch_model(tft_model, test_loader, device)
    results["TFT Model"] = {"F1": f1_score(tft_labels, tft_preds, zero_division=0)}
    probs_dict["TFT Model"] = tft_probs
    preds_dict["TFT Model"] = tft_preds

    # --- 4. Meta-Ensemble ---
    print(">> Generating Meta-Ensemble")
    ensemble_probs = (xgb_probs + hybrid_probs + tft_probs) / 3.0
    ensemble_preds = (ensemble_probs > 0.45).astype(int) 
    results["Meta-Ensemble"] = {"F1": f1_score(tft_labels, ensemble_preds, zero_division=0)}
    probs_dict["Meta-Ensemble"] = ensemble_probs
    preds_dict["Meta-Ensemble"] = ensemble_preds

    print("\nGenerating Full Thesis Graphic Suite (Confusion Matrices, PR Curves, ROC, F1)...")
    sns.set_theme(style="whitegrid")
    
    # 1. Training Convergence Loss Curve
    plt.figure(figsize=(10, 6))
    epochs = [1, 2, 3]
    plt.plot(epochs, hybrid_loss_history, marker='o', lw=2, label="Hybrid LSTM-Transformer")
    plt.plot(epochs, tft_loss_history, marker='s', lw=2, label="TFT Model")
    plt.title("Deep Learning Model Convergence (BCE Loss)", fontsize=16, pad=15)
    plt.xlabel("Training Epoch", fontsize=12)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
    plt.xticks(epochs)
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/training_loss_curve.png", dpi=300)
    plt.close()

    # 2. Confusion Matrices (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (name, preds) in enumerate(preds_dict.items()):
        cm = confusion_matrix(tft_labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Theft"])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False, values_format='d')
        axes[i].set_title(f"{name}", fontsize=14)
        axes[i].grid(False)
    plt.suptitle("Confusion Matrices Across Model Architectures", fontsize=18)
    plt.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.savefig("outputs/confusion_matrices.png", dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    for name, probs in probs_dict.items():
        try:
            precision, recall, _ = precision_recall_curve(tft_labels, probs)
            ap = average_precision_score(tft_labels, probs)
            lw = 3 if "Ensemble" in name else 2
            style = '-' if "Ensemble" in name else '--'
            plt.plot(recall, precision, style, lw=lw, label=f'{name} (AP = {ap:.3f})')
        except Exception:
            pass
    plt.xlabel('Recall (True Positive Rate)', fontsize=14)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=14)
    plt.title('Precision-Recall Curves for Highly Imbalanced Theft Detection', fontsize=16, pad=15)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/pr_curve_comparison.png", dpi=300)
    plt.close()

    # 4. ROC Curve 
    plt.figure(figsize=(10, 8))
    for name, probs in probs_dict.items():
        try:
            fpr, tpr, _ = roc_curve(tft_labels, probs)
            roc_auc = auc(fpr, tpr)
            lw = 3 if "Ensemble" in name else 2
            style = '-' if "Ensemble" in name else '--'
            plt.plot(fpr, tpr, style, lw=lw, label=f'{name} (AUC = {roc_auc:.3f})')
        except Exception:
            pass
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Topologies', fontsize=16, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig("outputs/roc_curve_comparison.png", dpi=300)
    plt.close()
    
    print(">> Script Complete! All PNG graphics exported to 'ml_engine/src/outputs/'.")

if __name__ == "__main__":
    run_comparative_study()
