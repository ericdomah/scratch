import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from trainer import GridGuardTrainer
from ensemble_model import GridGuardUniversalHybrid
from data_loader import ElectricityDataset
from xgboost_model import XGBoostBaseline

# 1. Advanced Configuration
DATA_PATH = "../../data/data set.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/data set.csv"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50  # Significantly more epochs for deep refinement
BATCH_SIZE = 128 # Larger batch size for more stable gradients
WINDOW_SIZE = 30

def train_vigorous():
    print(f"INITIALIZING VIGOROUS HARDENING PIPELINE")
    print(f"Source: {DATA_PATH}")
    print(f"Device: {DEVICE}")
    print("-" * 40)

    # 2. Load Dataset with Full Context
    print("[1/3] Preparing High-Volume Dataset...")
    full_dataset = ElectricityDataset(DATA_PATH, window_size=WINDOW_SIZE, transform=True)
    
    # Stratified-style split
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0)

    # 3. Train Universal Hybrid with Improved Trainer
    print("\n[2/3] Executing Deep Learning Hardening (30-50 Epochs)...")
    dl_model = GridGuardUniversalHybrid()
    
    # We use a slightly lower pos_weight (8.0 instead of 11.0) to force higher Precision
    trainer = GridGuardTrainer(dl_model, train_loader, val_loader, device=DEVICE, pos_weight=8.0)
    
    history = trainer.train(epochs=EPOCHS, save_path='best_model_vigorous.pth')

    # 4. Train XGBoost with Expanded Feature Extraction
    print("\n[3/3] Boosting Statistical Baseline (XGBoost)...")
    xgb_model = XGBoostBaseline(n_estimators=300, max_depth=7, learning_rate=0.05)
    
    # Collect 30,000 samples for the statistical layer
    X_list, y_list = [], []
    limit = min(30000, len(full_dataset))
    indices = np.random.choice(len(full_dataset), limit, replace=False)
    
    print(f"Extracting {limit} high-fidelity feature vectors...")
    for idx in indices:
        x, y = full_dataset[idx]
        X_list.append(x)
        y_list.append(y)
    
    X_xgb = torch.stack(X_list)
    y_xgb = torch.stack(y_list)
    
    xgb_model.train(X_xgb, y_xgb)
    xgb_model.save_model("best_xgb_vigorous.pkl")

    print("\n" + "=" * 40)
    print("VIGOROUS HARDENING COMPLETE")
    print(f"Final Best F1: {max(history['val_f1']):.4f}")
    print(f"Final Best AUC: {max(history['val_auc']):.4f}")
    print("Weights saved to: best_model_vigorous.pth and best_xgb_vigorous.pkl")
    print("=" * 40)

if __name__ == "__main__":
    train_vigorous()
