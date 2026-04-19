import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from trainer import GridGuardTrainer
from ensemble_model import GridGuardUniversalHybrid
from data_loader import ElectricityDataset
from xgboost_model import XGBoostBaseline

# 1. Configuration
DATA_PATH = "../../data/data set.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/data set.csv"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 10
BATCH_SIZE = 64
WINDOW_SIZE = 30

def train_full_system():
    print(f"🚀 INITIALIZING FULL SYSTEM TRAINING")
    print(f"Source: {DATA_PATH}")
    print(f"Device: {DEVICE}")
    print("-" * 40)

    # 2. Load Full Dataset
    print("[1/3] Loading Full Dataset into Memory...")
    full_dataset = ElectricityDataset(DATA_PATH, window_size=WINDOW_SIZE, transform=True)
    print(f"Total Samples: {len(full_dataset)}")

    # Split into train/val
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 3. Train Universal Hybrid (Deep Learning)
    print("\n[2/3] Training Universal Hybrid (Bi-LSTM + Transformer + TFT)...")
    dl_model = GridGuardUniversalHybrid()
    trainer = GridGuardTrainer(dl_model, train_loader, val_loader, device=DEVICE)
    trainer.train(epochs=EPOCHS, save_path='best_model.pth')

    # 4. Train XGBoost Baseline (Statistical)
    print("\n[3/3] Training XGBoost Baseline on Full Distribution...")
    xgb_model = XGBoostBaseline()
    
    # We collect a subset for XGBoost to avoid RAM overflow on standard machines, 
    # but still use 5x more data than the 'small' set.
    X_list, y_list = [], []
    print("Collecting high-fidelity feature vectors for XGBoost...")
    limit = min(20000, len(full_dataset))
    for i in range(limit):
        x, y = full_dataset[i]
        X_list.append(x)
        y_list.append(y)
    
    X_xgb = torch.stack(X_list)
    y_xgb = torch.stack(y_list)
    
    xgb_model.train(X_xgb, y_xgb)
    xgb_model.save_model("best_xgb.pkl")

    print("\n" + "=" * 40)
    print("✅ FULL SYSTEM TRAINING COMPLETE")
    print("Weights saved to: best_model.pth and best_xgb.pkl")
    print("=" * 40)

if __name__ == "__main__":
    train_full_system()
