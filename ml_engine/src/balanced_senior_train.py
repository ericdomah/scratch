import os
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score
)

from ensemble_model import GridGuardUniversalHybrid
from data_loader import ElectricityDataset
import random
from theft_injector import TheftInjector
from torch.utils.data import TensorDataset

# ── Configuration (Optimized for CPU) ────────────────────────────────────────
CFG = {
    "data_path"       : "../../data/data_set_cleaned.csv",
    "fallback_path"   : "../data/data_set_cleaned.csv",
    "device"          : "cpu",
    "epochs"          : 25,
    "batch_size"      : 128,
    "window_size"     : 30,
    "n_folds"         : 3,
    "lr_max"          : 2e-3,
    "weight_decay"    : 1e-4,
    "focal_alpha"     : 0.80,
    "focal_gamma"     : 2.0,
    "min_recall"      : 0.60,
    "save_dir"        : "checkpoints",
}

DEVICE = torch.device(CFG["device"])
os.makedirs(CFG["save_dir"], exist_ok=True)


# ── Asymmetric Focal Loss ─────────────────────────────────────────────────────
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma_pos=2.0, gamma_neg=4.0):
        super().__init__()
        self.alpha     = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits, targets):
        probs    = torch.sigmoid(logits).view(-1)
        targets  = targets.view(-1).float()
        bce      = F.binary_cross_entropy_with_logits(logits.view(-1), targets, reduction='none')
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        gamma_t  = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** gamma_t * bce).mean()


# ── Threshold Search ──────────────────────────────────────────────────────────
def find_optimal_threshold(probs, labels, min_recall=0.60):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    mask = recalls[:-1] >= min_recall
    if mask.sum() == 0:
        print("  ⚠️  Recall constraint unachievable, relaxing to 0.50")
        mask = recalls[:-1] >= 0.50
    valid_t = thresholds[mask]
    valid_r = recalls[:-1][mask]
    valid_p = precisions[:-1][mask]
    f1      = (2 * valid_p * valid_r) / (valid_p + valid_r + 1e-10)
    best    = np.argmax(f1)
    return valid_t[best], f1[best], valid_p[best], valid_r[best]


# ── Single Fold Training ──────────────────────────────────────────────────────
def train_one_fold(fold, train_ds, val_ds, cfg):
    model     = GridGuardUniversalHybrid().to(DEVICE)
    criterion = AsymmetricFocalLoss(alpha=cfg["focal_alpha"], gamma_neg=cfg["focal_gamma"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=cfg["weight_decay"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = cfg["lr_max"],
        steps_per_epoch = len(train_loader),
        epochs          = cfg["epochs"],
    )

    ckpt_path = os.path.join(cfg["save_dir"], f"fold{fold}_best.pth")

    # FIX: track best epoch's probabilities, not last epoch's
    best_f1     = 0.0
    best_probs  = None
    best_labels = None

    for epoch in range(cfg["epochs"]):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x.to(DEVICE)), batch_y.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                val_probs.extend(
                    torch.sigmoid(model(batch_x.to(DEVICE))).view(-1).cpu().numpy()
                )
                val_labels.extend(batch_y.numpy())

        val_probs_arr  = np.array(val_probs)
        val_labels_arr = np.array(val_labels)

        _, f1, p, r = find_optimal_threshold(val_probs_arr, val_labels_arr, cfg["min_recall"])
        auroc        = roc_auc_score(val_labels_arr, val_probs_arr)
        avg_loss     = train_loss / len(train_loader)

        print(f"  Fold {fold} | Ep {epoch+1:>2}/{cfg['epochs']} | "
              f"Loss: {avg_loss:.4f} | F1: {f1:.4f} | "
              f"Prec: {p:.4f} | Rec: {r:.4f} | AUROC: {auroc:.4f}")

        # FIX: save checkpoint AND probabilities from the best epoch only
        if f1 > best_f1:
            best_f1     = f1
            best_probs  = val_probs_arr.copy()   # snapshot of best epoch probs
            best_labels = val_labels_arr.copy()
            torch.save(model.state_dict(), ckpt_path)

    return model, best_probs, best_labels, best_f1


# ── Synthetic Data Augmentation ───────────────────────────────────────────────
def get_augmented_train_ds(full_ds, train_idx):
    """Dynamically creates a perfectly balanced augmented dataset."""
    X_list, y_list = [], []
    for idx in train_idx:
        x, y = full_ds[idx]
        X_list.append(x)
        y_list.append(y)
        
    X_train = torch.stack(X_list)
    y_train = torch.stack(y_list)
    
    num_train_0 = (y_train == 0).sum().item()
    num_train_1 = (y_train == 1).sum().item()
    num_to_augment = num_train_0 - num_train_1
    
    if num_to_augment <= 0:
        return TensorDataset(X_train, y_train)
        
    normal_indices = (y_train == 0).nonzero(as_tuple=True)[0].numpy()
    
    injector = TheftInjector()
    augmented_x = []
    augmented_y = []
    
    for _ in range(num_to_augment):
        idx = random.choice(normal_indices)
        normal_window = X_train[idx]
        seq_len = normal_window.shape[0]
        
        pattern_choice = random.randint(1, 4)
        if pattern_choice == 1:
            alpha = random.uniform(0.1, 0.8)
            theft_window = injector.inject_constant_reduction(normal_window, alpha)
        elif pattern_choice == 2:
            start = random.randint(0, seq_len // 2)
            end = random.randint(start + 1, seq_len)
            alpha = random.uniform(0.1, 0.8)
            theft_window = injector.inject_partial_bypass(normal_window, start, end, alpha)
        elif pattern_choice == 3:
            prob = random.uniform(0.3, 0.7)
            alpha = random.uniform(0.1, 0.5)
            theft_window = injector.inject_on_off_bypass(normal_window, prob, alpha)
        else:
            val = random.uniform(0.0, 0.2)
            theft_window = injector.inject_constant_value(normal_window, val)
            
        augmented_x.append(theft_window)
        augmented_y.append(torch.tensor(1.0, dtype=torch.float32))
        
    X_train_aug = torch.cat([X_train, torch.stack(augmented_x)])
    y_train_aug = torch.cat([y_train, torch.stack(augmented_y)])
    
    return TensorDataset(X_train_aug, y_train_aug)


# ── Main ──────────────────────────────────────────────────────────────────────
def train_kfold():
    print("-" * 60)
    print("  GridGuard AI v2 Lite — Fixed CPU Pipeline")
    print("-" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  Folds   : {CFG['n_folds']}   Epochs/fold: {CFG['epochs']}")
    print(f"  Loss    : Asymmetric Focal (alpha={CFG['focal_alpha']}, gamma={CFG['focal_gamma']})")
    print("-" * 60)

    data_path = CFG["data_path"] if os.path.exists(CFG["data_path"]) else CFG["fallback_path"]
    full_ds   = ElectricityDataset(data_path, window_size=CFG["window_size"], transform=True)

    print("Extracting labels for Stratified K-Fold...")
    try:
        all_labels = np.array(full_ds.labels)
    except AttributeError:
        all_labels = np.array([full_ds[i][1] for i in range(len(full_ds))])

    skf          = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_ds)), all_labels)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold+1}/{CFG['n_folds']}  |  "
              f"train={len(train_idx):,}  val={len(val_idx):,}")
        print(f"{'='*60}")

        # Generate perfectly balanced augmented dataset for this fold
        augmented_train_ds = get_augmented_train_ds(full_ds, train_idx)
        
        model, best_probs, best_labels, _ = train_one_fold(
            fold + 1,
            augmented_train_ds,
            Subset(full_ds, val_idx),
            CFG,
        )

        # Final threshold search on best-epoch probabilities
        best_t, best_f1, best_p, best_r = find_optimal_threshold(
            best_probs, best_labels, CFG["min_recall"]
        )
        auroc  = roc_auc_score(best_labels, best_probs)
        auprc  = average_precision_score(best_labels, best_probs)

        fold_results.append({
            "fold"      : fold + 1,
            "f1"        : best_f1,
            "prec"      : best_p,
            "rec"       : best_r,
            "auroc"     : auroc,
            "auprc"     : auprc,
            "threshold" : float(best_t),
        })

        print(f"\n  Fold {fold+1} Summary:")
        print(f"  Threshold : {best_t:.4f}")
        print(f"  Precision : {best_p:.4f}   Recall : {best_r:.4f}")
        print(f"  F1        : {best_f1:.4f}   AUROC  : {auroc:.4f}   AUPRC: {auprc:.4f}")

    # ── Cross-validation summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  CROSS-VALIDATION RESULTS")
    print("=" * 60)
    for m in ["f1", "prec", "rec", "auroc", "auprc"]:
        vals = [r[m] for r in fold_results]
        print(f"  {m.upper():<10}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 60)

    # ── Save best fold weights as production checkpoint ───────────────────────
    best_fold_idx = int(np.argmax([r["f1"] for r in fold_results]))
    best_ckpt     = os.path.join(CFG["save_dir"], f"fold{best_fold_idx+1}_best.pth")
    prod_ckpt     = "best_model_balanced.pth"

    import shutil
    shutil.copy(best_ckpt, prod_ckpt)

    # Save metadata (threshold + all fold metrics)
    meta = {
        "optimal_threshold" : fold_results[best_fold_idx]["threshold"],
        "best_fold"         : best_fold_idx + 1,
        "fold_results"      : fold_results,
        "config"            : CFG,
    }
    with open(os.path.join(CFG["save_dir"], "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Production weights  -> {prod_ckpt}")
    print(f"  Optimal threshold   -> {fold_results[best_fold_idx]['threshold']:.4f}")
    print(f"  Metadata saved      -> {CFG['save_dir']}/model_meta.json")


if __name__ == "__main__":
    train_kfold()