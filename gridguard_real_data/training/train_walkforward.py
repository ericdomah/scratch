"""
Experiment 2 -- SGCC Walk-Forward Temporal Validation (Primary Protocol)
=========================================================================
5 expanding-window rounds sorted by CONS_NO as a temporal proxy.

Fixes applied (per thesis prompt):
  FIX 1 — Per-round threshold calibration on inner val split fused probs
  FIX 2 — Class-weighted loss (pos_weight = n_normal / n_theft per round)
  FIX 3 — AsymmetricFocalLoss(alpha=0.92, gamma_pos=2.0, gamma_neg=2.0)
  FIX 4 — epochs=30, max_lr=5e-4, early stopping (patience=8 on val AUROC)
           AMP enabled on CUDA for speed.

Unchanged from thesis specification:
  - GridGuardUniversalHybrid and BiGRUBiLSTMBaseline architectures (frozen)
  - Fusion: 0.70 x P_DL + 0.30 x P_XGB
  - batch_size=128, AdamW lr=1e-4, wd=1e-4, seed=42
  - 5 walk-forward rounds, retrained from scratch each round
  - gradient_clip=1.0

Round schedule (per thesis specification)
-----------------------------------------
  Round  Train end  Test window
    1      60 %    -> next 20 %
    2      70 %    -> next 10 %
    3      80 %    -> next 10 %
    4      85 %    -> next 10 %
    5      90 %    -> remaining 10 %

Outputs
-------
  results/exp2_walkforward.csv   -- per-round metrics + summary
"""

from __future__ import annotations

import os
import sys
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
)
from scipy.stats import t as t_dist
import xgboost as xgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.gridguard_model import (
    GridGuardUniversalHybrid, BiGRUBiLSTMBaseline,
    AsymmetricFocalLoss, build_model,
)
from preprocessing.sgcc_pipeline import compute_tabular_features

# =============================================================================
#  Constants  (DO NOT CHANGE — thesis specification)
# =============================================================================
SEED             = 42
BATCH_SIZE       = 128
EPOCHS           = 30           # FIX 4: was 15
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
MAX_LR           = 5e-4         # FIX 4: was 2e-3
GRAD_CLIP        = 1.0
FIXED_TAU        = 0.5270
PATIENCE         = 8            # FIX 4: early stopping patience
CHECKPOINT_EVERY = 5
MIN_THEFT_PER_TEST = 30

ROUNDS = [
    (0.00, 0.60, 0.80),
    (0.00, 0.70, 0.80),
    (0.00, 0.80, 0.90),
    (0.00, 0.85, 0.95),
    (0.00, 0.90, 1.00),
]

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=SEED,
)

# =============================================================================
#  Reproducibility
# =============================================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic    = True
    torch.backends.cudnn.benchmark        = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False

# =============================================================================
#  Threshold Calibration  (FIX 1)
# =============================================================================

def calibrate_threshold(fused_probs: np.ndarray, labels: np.ndarray) -> float:
    best_t, best_f1 = FIXED_TAU, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, (fused_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2)

# =============================================================================
#  Metrics
# =============================================================================

def compute_metrics(y_true, y_prob, threshold, prefix=""):
    y_pred = (y_prob >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = float("nan")
    return {
        f"{prefix}F1":        f1_score(y_true, y_pred,        zero_division=0),
        f"{prefix}Precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}Recall":    recall_score(y_true, y_pred,    zero_division=0),
        f"{prefix}AUROC":     auroc,
        f"{prefix}Brier":     brier_score_loss(y_true, y_prob),
        f"{prefix}TN": float(tn), f"{prefix}FP": float(fp),
        f"{prefix}FN": float(fn), f"{prefix}TP": float(tp),
    }


def ci_95(values):
    n  = len(values)
    sd = np.std(values, ddof=1)
    return t_dist.ppf(0.975, df=n - 1) * sd / np.sqrt(n)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / (pooled_std + 1e-12)

# =============================================================================
#  Training helpers  (FIX 2: class weighting, FIX 4: AMP + early stopping)
# =============================================================================

def train_one_epoch(
    model, loader, optimizer, scheduler, criterion,
    device, pos_weight: float, scaler: GradScaler,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp = device == "cuda"

    for X_b, y_b in loader:
        X_b = X_b.to(device, non_blocking=True)
        y_b = y_b.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            per_sample_loss = criterion(model(X_b), y_b)
            # FIX 2: class-weighted loss
            theft_mask  = (y_b == 1).float()
            normal_mask = (y_b == 0).float()
            loss = (per_sample_loss * theft_mask  * pos_weight +
                    per_sample_loss * normal_mask * 1.0).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * len(X_b)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_probabilities(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    use_amp = device == "cuda"
    for X_b, _ in loader:
        with autocast(enabled=use_amp):
            p = model(X_b.to(device, non_blocking=True)).squeeze().float()
        if p.ndim == 0:
            p = p.reshape(1)
        probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def fuse_predictions(p_dl: np.ndarray, p_xgb: np.ndarray) -> np.ndarray:
    return 0.70 * p_dl + 0.30 * p_xgb


def make_loader(X, y, shuffle: bool, device: str) -> DataLoader:
    use_gpu = device == "cuda"
    return DataLoader(
        TensorDataset(X, y),
        batch_size=BATCH_SIZE, shuffle=shuffle,
        pin_memory=use_gpu,
        num_workers=4 if use_gpu else 0,
        persistent_workers=(use_gpu),
    )

# =============================================================================
#  Train one model on a split with all fixes
# =============================================================================

def train_model_on_split(
    model_type: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val:   torch.Tensor,
    y_val:   torch.Tensor,
    X_test:  torch.Tensor,
    y_test:  torch.Tensor,
    feats_train: np.ndarray,
    feats_val:   np.ndarray,
    feats_test:  np.ndarray,
    y_train_np:  np.ndarray,
    y_val_np:    np.ndarray,
    device: str,
    round_id: int,
    ckpt_dir: str,
) -> Tuple[np.ndarray, float, float]:
    """
    Train a FRESH model + XGB on (X_train) with inner val for early stopping.
    Returns (fused_test_probs, calibrated_threshold, xgb_clf).
    No weight sharing between rounds.
    """
    train_ld = make_loader(X_train, y_train, shuffle=True,  device=device)
    val_ld   = make_loader(X_val,   y_val,   shuffle=False, device=device)
    test_ld  = make_loader(X_test,  y_test,  shuffle=False, device=device)

    # Class weighting (FIX 2)
    n_theft  = int((y_train_np == 1).sum())
    n_normal = int((y_train_np == 0).sum())
    pos_weight = n_normal / max(n_theft, 1)
    print(f"     [{model_type}] pos_weight={pos_weight:.2f} "
          f"(n_normal={n_normal:,}  n_theft={n_theft:,})")

    model = build_model(model_type, device=device)
    # FIX 3: alpha=0.92, gamma_neg=2.0 for real SGCC data
    criterion = AsymmetricFocalLoss(alpha=0.92, gamma_pos=2.0, gamma_neg=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps     = EPOCHS * len(train_ld)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=steps
    )
    amp_scaler = GradScaler(enabled=(device == "cuda"))

    # XGB trained on train split
    n_xgb_theft  = int((y_train_np == 1).sum())
    n_xgb_normal = int((y_train_np == 0).sum())
    clf = xgb.XGBClassifier(
        scale_pos_weight=n_xgb_normal / max(n_xgb_theft, 1),
        **XGB_PARAMS,
    )
    clf.fit(feats_train, y_train_np,
            eval_set=[(feats_val, y_val_np)],
            verbose=False)
    p_xgb_val = clf.predict_proba(feats_val)[:, 1]

    # Early stopping on val AUROC (FIX 4)
    best_val_auroc = -1.0
    patience_ctr   = 0
    best_state     = None
    best_epoch     = 1

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model, train_ld, optimizer, scheduler,
            criterion, device, pos_weight, amp_scaler,
        )

        p_dl_val  = get_probabilities(model, val_ld, device)
        va_fused  = fuse_predictions(p_dl_val, p_xgb_val)
        try:
            val_auroc = roc_auc_score(y_val_np, va_fused)
        except Exception:
            val_auroc = 0.0

        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"     [{model_type}] Round {round_id}  "
                  f"Epoch {epoch:2d}/{EPOCHS}  "
                  f"loss={loss:.4f}  val_AUROC={val_auroc:.4f}")

        if epoch % CHECKPOINT_EVERY == 0:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir,
                                    f"{model_type}_round{round_id}_epoch{epoch}.pth"))

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch     = epoch
            patience_ctr   = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"     [{model_type}] Early stopping at epoch {epoch}, "
                      f"best epoch={best_epoch} (AUROC={best_val_auroc:.4f})")
                break

    # Load best checkpoint
    model.load_state_dict(best_state)

    # Threshold calibration on val fused probs (FIX 1)
    p_dl_val_best  = get_probabilities(model, val_ld, device)
    va_fused_best  = fuse_predictions(p_dl_val_best, p_xgb_val)
    tau_cal        = calibrate_threshold(va_fused_best, y_val_np)
    print(f"     [{model_type}] Calibrated threshold = {tau_cal:.2f}")

    # Test inference
    p_dl_test  = get_probabilities(model, test_ld, device)
    p_xgb_test = clf.predict_proba(feats_test)[:, 1]
    p_fused    = fuse_predictions(p_dl_test, p_xgb_test)

    return p_fused, tau_cal, clf

# =============================================================================
#  Walk-Forward Split Builder
# =============================================================================

def build_wf_splits(n: int, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    for _, train_end_frac, test_end_frac in ROUNDS:
        train_end = int(n * train_end_frac)
        test_end  = min(int(n * test_end_frac), n)
        tr_idx    = np.arange(0, train_end)
        te_idx    = np.arange(train_end, test_end)

        if len(tr_idx) == 0 or len(te_idx) == 0:
            print(f"  [WF] Skipping empty split "
                  f"(train={train_end_frac:.0%}, test={test_end_frac:.0%})")
            continue

        n_theft_test = y[te_idx].sum()
        if n_theft_test < MIN_THEFT_PER_TEST:
            print(f"  [WF] Skipping (only {n_theft_test:.0f} theft in test)")
            continue

        splits.append((tr_idx, te_idx))
    return splits

# =============================================================================
#  Main: Walk-Forward Temporal Validation  (Experiment 2)
# =============================================================================

def run_walk_forward(
    X: torch.Tensor,
    y: torch.Tensor,
    metadata: dict,
    output_dir: str,
    verbose: bool = True,
) -> pd.DataFrame:
    set_seed(SEED)
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 2 -- Walk-Forward Temporal Validation")
    print(f"  Device : {device.upper()}  |  AMP: {'Enabled' if use_amp else 'Disabled'}")
    print(f"  Epochs : {EPOCHS}  |  max_lr={MAX_LR}  |  patience={PATIENCE}")
    print(f"{'='*60}")

    result_dir = os.path.join(output_dir, "results")
    ckpt_dir   = os.path.join(output_dir, "checkpoints", "exp2")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    # Sort by CONS_NO as temporal proxy
    sort_order = metadata["sort_order"]
    X_sorted   = X[sort_order]
    y_sorted   = y[sort_order]
    y_np       = y_sorted.numpy().astype(int)
    n          = len(y_np)

    feats_all  = compute_tabular_features(X_sorted.numpy())

    print(f"  Total samples (one per consumer): {n:,}")
    splits = build_wf_splits(n, y_np)
    print(f"  Valid walk-forward rounds: {len(splits)}")

    if len(splits) == 0:
        print("  [WF] No valid splits.")
        empty_df = pd.DataFrame()
        empty_df.to_csv(os.path.join(result_dir, "exp2_walkforward.csv"), index=False)
        return empty_df

    rows     = []
    gg_f1s   = []
    base_f1s = []

    for rnd, (tr_idx, te_idx) in enumerate(splits, 1):
        print(f"\n-- Round {rnd}/{len(splits)} " + "-"*40)
        print(f"   Train: {len(tr_idx):,}  |  Test: {len(te_idx):,}  |  "
              f"Theft(train): {y_np[tr_idx].mean():.2%}  |  "
              f"Theft(test): {y_np[te_idx].mean():.2%}")

        # Inner val split from training set (10%) for early stopping
        tr_inner, va_idx = train_test_split(
            tr_idx, test_size=0.10,
            stratify=y_np[tr_idx],
            random_state=SEED + rnd,
        )

        X_tr = X_sorted[tr_inner]; y_tr = y_sorted[tr_inner]
        X_va = X_sorted[va_idx];   y_va = y_sorted[va_idx]
        X_te = X_sorted[te_idx];   y_te = y_sorted[te_idx]

        y_tr_np = y_np[tr_inner]
        y_va_np = y_np[va_idx]
        y_te_np = y_np[te_idx]

        # -- GridGuardUniversalHybrid -----------------------------------------
        set_seed(SEED + rnd)
        print(f"  Training GridGuardUniversalHybrid (from scratch)...")
        p_gg, tau_gg, _ = train_model_on_split(
            "gridguard",
            X_tr, y_tr, X_va, y_va, X_te, y_te,
            feats_all[tr_inner], feats_all[va_idx], feats_all[te_idx],
            y_tr_np, y_va_np,
            device, rnd, ckpt_dir,
        )
        m_gg       = compute_metrics(y_te_np, p_gg, tau_gg,   prefix="GG_")
        m_gg_fixed = compute_metrics(y_te_np, p_gg, FIXED_TAU, prefix="GG_Fixed_")
        gg_f1s.append(m_gg["GG_F1"])

        # -- BiGRU-BiLSTM Baseline -------------------------------------------
        set_seed(SEED + rnd + 100)
        print(f"  Training BiGRU-BiLSTM Baseline (from scratch)...")
        p_base, tau_base, _ = train_model_on_split(
            "bigru_bilstm",
            X_tr, y_tr, X_va, y_va, X_te, y_te,
            feats_all[tr_inner], feats_all[va_idx], feats_all[te_idx],
            y_tr_np, y_va_np,
            device, rnd, ckpt_dir,
        )
        m_base = compute_metrics(y_te_np, p_base, tau_base, prefix="Base_")
        base_f1s.append(m_base["Base_F1"])

        gg_wins = m_gg["GG_F1"] > m_base["Base_F1"]
        print(f"  GG   [tau={tau_gg:.2f}] "
              f"F1={m_gg['GG_F1']:.4f}  AUROC={m_gg['GG_AUROC']:.4f}  "
              f"Prec={m_gg['GG_Precision']:.4f}  Rec={m_gg['GG_Recall']:.4f}")
        print(f"  Base [tau={tau_base:.2f}] "
              f"F1={m_base['Base_F1']:.4f}  AUROC={m_base['Base_AUROC']:.4f}")
        print(f"  GG_Fixed_tau F1={m_gg_fixed['GG_Fixed_F1']:.4f}")
        print(f"  GridGuard {'OUTPERFORMS' if gg_wins else 'does NOT outperform'} "
              f"baseline in Round {rnd}")

        row = {
            "Round":           rnd,
            "Train_samples":   len(tr_inner),
            "Test_samples":    len(te_idx),
            "Theft_frac_test": round(float(y_te_np.mean()), 4),
            "GG_threshold":    tau_gg,
            "Base_threshold":  tau_base,
            **m_gg, **m_base,
            "GG_F1_fixed_tau": m_gg_fixed["GG_Fixed_F1"],
        }
        rows.append(row)

    # =========================================================================
    #  Summary
    # =========================================================================
    results_df = pd.DataFrame(rows)
    gg_f1s     = np.array(gg_f1s)
    base_f1s   = np.array(base_f1s)
    d          = cohens_d(gg_f1s, base_f1s)

    print(f"\n{'='*60}")
    for metric in ["GG_F1", "GG_AUROC", "GG_Precision", "GG_Recall",
                   "Base_F1", "Base_AUROC"]:
        vals = results_df[metric].values.astype(float)
        mu   = vals.mean()
        sd   = vals.std(ddof=1)
        ci   = ci_95(vals)
        print(f"  {metric}: {mu:.4f} +/- {sd:.4f}  [95%CI +/-{ci:.4f}]")

    print(f"\n  Cohen's d (GridGuard vs Baseline, F1): {d:.4f}")
    print(f"  (|d|>0.8=large, 0.5-0.8=medium, 0.2-0.5=small)")

    summary_row = {
        "Round":     "mean +/- SD",
        "GG_F1":     f"{gg_f1s.mean():.4f} +/- {gg_f1s.std(ddof=1):.4f} "
                     f"[95%CI +/-{ci_95(gg_f1s):.4f}]",
        "Base_F1":   f"{base_f1s.mean():.4f} +/- {base_f1s.std(ddof=1):.4f} "
                     f"[95%CI +/-{ci_95(base_f1s):.4f}]",
        "Cohens_d":  round(d, 4),
    }
    final_df = pd.concat(
        [results_df, pd.DataFrame([summary_row])], ignore_index=True
    )
    csv_path = os.path.join(result_dir, "exp2_walkforward.csv")
    final_df.to_csv(csv_path, index=False)

    print(f"\n  Results -> {csv_path}")
    print(f"{'='*60}\n")

    return final_df


# =============================================================================
#  CLI entry-point
# =============================================================================

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgcc_path",  default="data/sgcc")
    ap.add_argument("--output_dir", default=".")
    args = ap.parse_args()

    from preprocessing.sgcc_pipeline import run_sgcc_pipeline
    X, y, meta = run_sgcc_pipeline(
        args.sgcc_path,
        cache_path=os.path.join(args.sgcc_path, "sgcc_processed.pt"),
    )
    run_walk_forward(X, y, meta, output_dir=args.output_dir)
