import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from trainer import GridGuardTrainer
from ensemble_model import GridGuardUniversalHybrid
from data_loader import ElectricityDataset

# 1. Ultra-Conservative Configuration
DATA_PATH = "../../data/data set.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/data set.csv"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 40
BATCH_SIZE = 64
WINDOW_SIZE = 30
TARGET_THRESHOLD = 0.85 # High threshold for 3-in-4 precision

def train_precision_max():
    print("INITIALIZING PRECISION-MAX (3-IN-4) PIPELINE")
    print(f"Source: {DATA_PATH}")
    print(f"Inference Threshold Target: {TARGET_THRESHOLD}")
    print("-" * 40)

    # 2. Load Dataset
    full_dataset = ElectricityDataset(DATA_PATH, window_size=WINDOW_SIZE, transform=True)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # 3. Train with Balanced Weights (pos_weight=1.0)
    # This forces the model to only flag very obvious theft
    dl_model = GridGuardUniversalHybrid()
    trainer = GridGuardTrainer(dl_model, train_loader, val_loader, device=DEVICE, pos_weight=1.0)
    
    print(f"Phase 1: Training for high-confidence signatures...")
    history = trainer.train(
        epochs=EPOCHS, 
        save_path='best_model_precision.pth', 
        eval_threshold=TARGET_THRESHOLD
    )

    print("\n" + "=" * 40)
    print("PRECISION-MAX HARDENING COMPLETE")
    print(f"Target Threshold: {TARGET_THRESHOLD}")
    print(f"Peak Precision Reached: {max(history['val_prec']):.4f}")
    print("Weights saved to: best_model_precision.pth and best_model_precision_prec.pth")
    print("=" * 40)

if __name__ == "__main__":
    train_precision_max()
