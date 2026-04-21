import os
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from data_loader import ElectricityDataset
from xgboost_model import XGBoostBaseline
from theft_injector import TheftInjector

# Configuration
CFG = {
    "data_path": "../../data/data_set_cleaned.csv",
    "fallback_path": "../data/data_set_cleaned.csv",
    "window_size": 30,
    "test_size": 0.20,
    "random_seed": 42,
    "xgb_estimators": 200,
    "xgb_max_depth": 6,
    "xgb_lr": 0.1,
    "model_save_path": "best_xgb_augmented.pkl"
}

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def apply_random_theft_pattern(window):
    """Applies a random academic theft pattern to a normal consumption window."""
    pattern_choice = random.randint(1, 4)
    injector = TheftInjector()
    
    # window shape is (seq_len, 1)
    seq_len = window.shape[0]
    
    if pattern_choice == 1:
        # Constant reduction
        alpha = random.uniform(0.1, 0.8)
        return injector.inject_constant_reduction(window, alpha)
    elif pattern_choice == 2:
        # Partial bypass
        start = random.randint(0, seq_len // 2)
        end = random.randint(start + 1, seq_len)
        alpha = random.uniform(0.1, 0.8)
        return injector.inject_partial_bypass(window, start, end, alpha)
    elif pattern_choice == 3:
        # On/Off bypass
        prob = random.uniform(0.3, 0.7)
        alpha = random.uniform(0.1, 0.5)
        return injector.inject_on_off_bypass(window, prob, alpha)
    else:
        # Constant low value (Flatline)
        val = random.uniform(0.0, 0.2)
        return injector.inject_constant_value(window, val)

def train_advanced_xgboost():
    print("=" * 60)
    print("  Advanced XGBoost Training with Synthetic Augmentation")
    print("=" * 60)
    set_seed(CFG["random_seed"])
    
    # 1. Load Data
    data_path = CFG["data_path"] if os.path.exists(CFG["data_path"]) else CFG["fallback_path"]
    print(f"Loading cleaned dataset from {data_path}...")
    dataset = ElectricityDataset(data_path, window_size=CFG["window_size"], transform=True)
    
    X_list, y_list = [], []
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_list.append(x)
        y_list.append(y)
        
    X_all = torch.stack(X_list)
    y_all = torch.stack(y_list).numpy()
    
    # 2. Stratified Split (Pristine Validation Set)
    print("\nSplitting into Train / Validation (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=CFG["test_size"], stratify=y_all, random_state=CFG["random_seed"]
    )
    
    num_train = len(y_train)
    num_train_0 = (y_train == 0).sum()
    num_train_1 = (y_train == 1).sum()
    print(f"Original Train Set : {num_train} samples (Normal: {num_train_0}, Theft: {num_train_1})")
    
    # 3. Synthetic Augmentation on Train Set ONLY
    print("\nPerforming Synthetic Data Augmentation...")
    num_to_augment = num_train_0 - num_train_1
    print(f"Targeting a 50/50 balance -> Generating {num_to_augment} synthetic theft cases.")
    
    # Isolate normal samples to augment
    normal_indices = np.where(y_train == 0)[0]
    
    augmented_x = []
    augmented_y = []
    
    for i in range(num_to_augment):
        # Pick a random normal sample
        idx = random.choice(normal_indices)
        normal_window = X_train[idx]
        
        # Turn it into a theft case
        theft_window = apply_random_theft_pattern(normal_window)
        augmented_x.append(theft_window)
        augmented_y.append(1.0)  # Label as theft
        
    # Append to training set
    if augmented_x:
        X_train_aug = torch.cat([X_train, torch.stack(augmented_x)])
        y_train_aug = np.concatenate([y_train, np.array(augmented_y)])
    else:
        X_train_aug = X_train
        y_train_aug = y_train
        
    print(f"Augmented Train Set: {len(y_train_aug)} samples (Normal: {(y_train_aug==0).sum()}, Theft: {(y_train_aug==1).sum()})")
    
    # 4. Train XGBoost
    print("\nTraining XGBoost model...")
    model = XGBoostBaseline(
        n_estimators=CFG["xgb_estimators"], 
        max_depth=CFG["xgb_max_depth"], 
        learning_rate=CFG["xgb_lr"]
    )
    
    # Train using the augmented set
    model.train(X_train_aug, y_train_aug)
    
    # 5. Evaluate on PRISTINE validation set
    from balanced_senior_train import find_optimal_threshold
    
    print("\nEvaluating on PRISTINE Validation Set...")
    val_probs = model.predict_proba(X_val)
    
    # Use the same threshold search as the DL model
    best_t, f1, precision, recall = find_optimal_threshold(val_probs, y_val, min_recall=0.60)
    
    auroc = roc_auc_score(y_val, val_probs)
    auprc = average_precision_score(y_val, val_probs)
    
    print("=" * 60)
    print("  FINAL VALIDATION METRICS (XGBoost + Augmentation)")
    print("=" * 60)
    print(f"  Optimal Threshold : {best_t:.4f}")
    print(f"  Precision         : {precision:.4f}")
    print(f"  Recall            : {recall:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  AUROC             : {auroc:.4f}")
    print(f"  AUPRC             : {auprc:.4f}")
    print("=" * 60)
    
    # 6. Save
    model.save_model(CFG["model_save_path"])
    print(f"\nAdvanced XGBoost model saved to {CFG['model_save_path']}")

if __name__ == "__main__":
    train_advanced_xgboost()
