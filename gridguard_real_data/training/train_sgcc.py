"""
Experiment 1 -- SGCC Standard Cross-Validation Training (Protocol A*)
=====================================================================
10-fold StratifiedKFold on all SGCC windows.

Fixes applied (per thesis prompt):
  FIX 1 — Per-fold threshold calibration (sweep 0.05-0.95 on val fused probs)
  FIX 2 — Class-weighted loss (pos_weight = n_normal / n_theft per fold)
  FIX 3 — AsymmetricFocalLoss(alpha=0.92, gamma_pos=2.0, gamma_neg=2.0)
  FIX 4 — epochs=30, max_lr=5e-4, early stopping (patience=8 on val AUROC)
           Also uses AMP on CUDA for speed.

Unchanged from thesis specification:
  - GridGuardUniversalHybrid architecture (frozen)
  - Fusion: 0.70 x P_DL + 0.30 x P_XGB
  - batch_size=128, AdamW lr=1e-4, wd=1e-4, seed=42
  - 10 folds, gradient_clip=1.0
  - XGBoost parameters

Outputs
-------
  models/gridguard_sgcc_best.pth   -- best DL fold by fused F1
  models/xgboost_sgcc_edge.pkl     -- XGBoost edge filter from best fold
  results/exp1_standard_cv.csv     -- per-fold metrics + summary row
"""

from __future__ import annotations

import os
import sys
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
)
from scipy.stats import t as t_dist
import xgboost as xgb
import pandas as pd

# -- path setup ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.gridguard_model import GridGuardUniversalHybrid, AsymmetricFocalLoss
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
FIXED_TAU        = 0.5270       # kept for F1_fixed_tau column only
N_FOLDS          = 10
PATIENCE         = 8            # FIX 4: early stopping patience
CHECKPOINT_EVERY = 5

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
    """Sweep 0.05-0.95 in steps of 0.01; return threshold that maximises F1."""
    best_t, best_f1 = FIXED_TAU, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, (fused_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(float(best_t), 2)

# =============================================================================
#  Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    prefix: str = "",
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}F1":        f1_score(y_true, y_pred,        zero_division=0),
        f"{prefix}Precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}Recall":    recall_score(y_true, y_pred,    zero_division=0),
        f"{prefix}AUROC":     roc_auc_score(y_true, y_prob),
        f"{prefix}Brier":     brier_score_loss(y_true, y_prob),
        f"{prefix}TN": float(tn), f"{prefix}FP": float(fp),
        f"{prefix}FN": float(fn), f"{prefix}TP": float(tp),
    }

def ci_95(values: np.ndarray) -> float:
    n  = len(values)
    sd = values.std(ddof=1)
    return t_dist.ppf(0.975, df=n - 1) * sd / np.sqrt(n)

# =============================================================================
#  Training Loop  (FIX 2: class-weighted loss, FIX 3: focal params, FIX 4: AMP)
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: str,
    pos_weight: float,
    scaler: GradScaler,
) -> float:
    """
    One epoch with:
      - AMP (autocast) for GPU speed
      - Class-weighted loss: theft samples weighted by pos_weight
    """
    model.train()
    total_loss = 0.0
    use_amp = device == "cuda"

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            preds           = model(X_batch)
            # FIX 3: alpha=0.92, gamma_neg=2.0
            per_sample_loss = criterion(preds, y_batch)
            # FIX 2: class-weighted loss
            theft_mask  = (y_batch == 1).float()
            normal_mask = (y_batch == 0).float()
            loss = (per_sample_loss * theft_mask  * pos_weight +
                    per_sample_loss * normal_mask * 1.0).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> np.ndarray:
    model.eval()
    probs = []
    use_amp = device == "cuda"
    for X_batch, _ in loader:
        with autocast(enabled=use_amp):
            p = model(X_batch.to(device, non_blocking=True)).squeeze().float()
        if p.ndim == 0:
            p = p.reshape(1)
        probs.append(p.cpu().numpy())
    return np.concatenate(probs)

# =============================================================================
#  Fusion
# =============================================================================

def fuse_predictions(p_dl: np.ndarray, p_xgb: np.ndarray) -> np.ndarray:
    """Late fusion: 0.70 x P_DL + 0.30 x P_XGB"""
    return 0.70 * p_dl + 0.30 * p_xgb

# =============================================================================
#  XGBoost
# =============================================================================

def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray,   y_val: np.ndarray,
) -> tuple:
    n_normal = (y_train == 0).sum()
    n_theft  = (y_train == 1).sum()
    spw      = n_normal / max(n_theft, 1)

    clf = xgb.XGBClassifier(scale_pos_weight=spw, **XGB_PARAMS)
    clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False)

    return clf, clf.predict_proba(X_val)[:, 1]

# =============================================================================
#  Main: 10-fold Standard CV  (Experiment 1)
# =============================================================================

def run_standard_cv(
    X: torch.Tensor,
    y: torch.Tensor,
    output_dir: str,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run 10-fold StratifiedKFold CV with all 5 fixes applied.

    Each fold:
      - Inner 10% val split for early stopping and threshold calibration
      - Early stopping on val AUROC (patience=8)
      - Per-fold threshold calibration (F1-maximising sweep on val)
      - Results include both calibrated F1 and F1_fixed_tau

    Returns DataFrame with per-fold rows + summary row.
    """
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 1 -- Standard {n_folds}-Fold StratifiedKFold")
    print(f"  Device   : {device.upper()}")
    print(f"  AMP      : {'Enabled' if use_amp else 'Disabled'}")
    print(f"  Epochs   : {EPOCHS}  |  max_lr={MAX_LR}  |  patience={PATIENCE}")
    print(f"  Loss     : AsymmetricFocalLoss(alpha=0.92, gpos=2.0, gneg=2.0)")
    print(f"{'='*60}")

    model_dir  = os.path.join(output_dir, "models")
    result_dir = os.path.join(output_dir, "results")
    ckpt_dir   = os.path.join(output_dir, "checkpoints", "exp1")
    for d in (model_dir, result_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    X_np     = X.numpy()
    y_np     = y.numpy().astype(int)
    feats_np = compute_tabular_features(X_np)   # (N, 5)

    skf        = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_rows  = []
    best_f1    = -1.0
    cal_taus   = []              # collect per-fold calibrated thresholds
    best_weights_path = None

    # Resume from partial
    partial_csv     = os.path.join(result_dir, "exp1_standard_cv_partial.csv")
    completed_folds = set()
    if os.path.isfile(partial_csv):
        partial_df = pd.read_csv(partial_csv)
        fold_rows  = partial_df.to_dict("records")
        completed_folds = {int(r["Fold"]) for r in fold_rows
                           if str(r.get("Fold", "")).isdigit()}
        if fold_rows:
            best_f1  = max(float(r.get("Fused_F1", -1.0)) for r in fold_rows)
            cal_taus = [float(r.get("threshold", FIXED_TAU))
                        for r in fold_rows if str(r.get("Fold", "")).isdigit()]
        print(f"\n  [Resume] {len(completed_folds)} folds done. "
              f"Best F1 so far: {best_f1:.4f}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y_np), 1):
        print(f"\n-- Fold {fold}/{n_folds} " + "-"*40)
        if fold in completed_folds:
            print("   [SKIPPED] Already completed.")
            continue

        # -- Inner val split (10%) for early stopping + threshold calibration --
        tr_inner, inner_va_idx = train_test_split(
            tr_idx, test_size=0.10,
            stratify=y_np[tr_idx],
            random_state=SEED + fold,
        )

        X_tr = X[tr_inner];      y_tr = y[tr_inner]
        X_va = X[inner_va_idx];  y_va = y[inner_va_idx]
        X_te = X[va_idx];        y_te = y[va_idx]      # outer test fold

        # -- Class weighting (FIX 2) -------------------------------------------
        n_theft  = int((y_np[tr_inner] == 1).sum())
        n_normal = int((y_np[tr_inner] == 0).sum())
        pos_weight = n_normal / max(n_theft, 1)
        print(f"   pos_weight = {pos_weight:.2f}  "
              f"(n_normal={n_normal:,}  n_theft={n_theft:,})")

        # -- DataLoaders -------------------------------------------------------
        num_wk = 4 if use_amp else 0
        train_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                              shuffle=True,  pin_memory=use_amp,
                              num_workers=num_wk,
                              persistent_workers=(num_wk > 0))
        val_ld   = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE,
                              shuffle=False, pin_memory=use_amp,
                              num_workers=num_wk,
                              persistent_workers=(num_wk > 0))
        test_ld  = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH_SIZE,
                              shuffle=False)

        # -- Model + optimizer + scheduler ------------------------------------
        set_seed(SEED + fold)
        model     = GridGuardUniversalHybrid().to(device)
        # FIX 3: alpha=0.92, gamma_neg=2.0
        criterion = AsymmetricFocalLoss(alpha=0.92, gamma_pos=2.0, gamma_neg=2.0)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LR, weight_decay=WEIGHT_DECAY)
        steps     = EPOCHS * len(train_ld)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR, total_steps=steps
        )
        amp_scaler = GradScaler(enabled=use_amp)

        # -- XGB trained once on inner train split ----------------------------
        clf, p_xgb_va = train_xgboost(
            feats_np[tr_inner], y_np[tr_inner],
            feats_np[inner_va_idx], y_np[inner_va_idx],
        )

        # -- Early stopping state (FIX 4) -------------------------------------
        best_val_auroc = -1.0
        patience_ctr   = 0
        best_state     = None
        best_epoch     = 1

        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(
                model, train_ld, optimizer, scheduler,
                criterion, device, pos_weight, amp_scaler,
            )

            # Validation AUROC for early stopping
            p_dl_va   = get_probabilities(model, val_ld, device)
            va_fused  = fuse_predictions(p_dl_va, p_xgb_va)
            y_va_np   = y_np[inner_va_idx]
            try:
                val_auroc = roc_auc_score(y_va_np, va_fused)
            except Exception:
                val_auroc = 0.0

            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"   Epoch {epoch:2d}/{EPOCHS}  "
                      f"loss={loss:.4f}  val_AUROC={val_auroc:.4f}")

            # Checkpoint every N epochs
            if epoch % CHECKPOINT_EVERY == 0:
                torch.save(model.state_dict(),
                           os.path.join(ckpt_dir, f"fold{fold}_epoch{epoch}.pth"))

            # Early stopping check
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_state     = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}
                best_epoch     = epoch
                patience_ctr   = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"   Early stopping at epoch {epoch}, "
                          f"best AUROC epoch was {best_epoch} "
                          f"(val_AUROC={best_val_auroc:.4f})")
                    break

        # Load best checkpoint
        model.load_state_dict(best_state)

        # -- Threshold calibration on val fused probs (FIX 1) ----------------
        p_dl_va_best  = get_probabilities(model, val_ld, device)
        va_fused_best = fuse_predictions(p_dl_va_best, p_xgb_va)
        tau_cal       = calibrate_threshold(va_fused_best, y_va_np)
        cal_taus.append(tau_cal)
        print(f"   Calibrated threshold (val) = {tau_cal:.2f}")

        # -- Test evaluation --------------------------------------------------
        p_dl_te  = get_probabilities(model, test_ld, device)
        p_xgb_te = clf.predict_proba(feats_np[va_idx])[:, 1]
        p_fused  = fuse_predictions(p_dl_te, p_xgb_te)
        y_te_np  = y_np[va_idx]

        m_cal   = compute_metrics(y_te_np, p_fused, tau_cal,   prefix="Fused_")
        m_fixed = compute_metrics(y_te_np, p_fused, FIXED_TAU, prefix="Fixed_")

        print(f"   [Cal τ={tau_cal:.2f}] "
              f"F1={m_cal['Fused_F1']:.4f}  "
              f"AUROC={m_cal['Fused_AUROC']:.4f}  "
              f"Prec={m_cal['Fused_Precision']:.4f}  "
              f"Rec={m_cal['Fused_Recall']:.4f}")
        print(f"   [Fixed τ={FIXED_TAU}] "
              f"F1={m_fixed['Fixed_F1']:.4f}")

        row = {
            "Fold":      fold,
            "threshold": tau_cal,
            **m_cal,
            "F1_fixed_tau": m_fixed["Fixed_F1"],
        }
        fold_rows.append(row)

        # Save best model globally
        f1_fused = m_cal["Fused_F1"]
        if f1_fused > best_f1:
            best_f1 = f1_fused
            best_weights_path = os.path.join(model_dir, "gridguard_sgcc_best.pth")
            torch.save(model.state_dict(), best_weights_path)
            best_xgb_path = os.path.join(model_dir, "xgboost_sgcc_edge.pkl")
            with open(best_xgb_path, "wb") as fh:
                pickle.dump({"clf": clf, "tau_edge": tau_cal}, fh)
            print(f"   * New best fold (F1={best_f1:.4f}) -- weights saved")

        # Save partial progress
        pd.DataFrame(fold_rows).to_csv(partial_csv, index=False)

    # =========================================================================
    #  Summary Statistics
    # =========================================================================
    results_df = pd.DataFrame(fold_rows)
    key_cols   = ["Fused_F1", "Fused_AUROC", "Fused_Precision",
                  "Fused_Recall", "Fused_Brier"]

    summary: Dict = {
        "Fold":      "mean +/- SD",
        "threshold": round(float(np.mean(cal_taus)), 3),
    }
    for col in key_cols:
        vals = results_df[col].values.astype(float)
        mu   = vals.mean()
        sd   = vals.std(ddof=1)
        ci   = ci_95(vals)
        summary[col] = f"{mu:.4f} +/- {sd:.4f}  [95%CI +/-{ci:.4f}]"
    summary["F1_fixed_tau"] = (
        f"{results_df['F1_fixed_tau'].astype(float).mean():.4f}"
    )
    # Confusion matrix totals across all folds
    for c in ["Fused_TN", "Fused_FP", "Fused_FN", "Fused_TP"]:
        if c in results_df.columns:
            summary[c] = int(results_df[c].astype(float).sum())

    final_df = pd.concat(
        [results_df, pd.DataFrame([summary])], ignore_index=True
    )
    csv_path = os.path.join(result_dir, "exp1_standard_cv.csv")
    final_df.to_csv(csv_path, index=False)

    # Clean up partial CSV
    if os.path.exists(partial_csv):
        os.remove(partial_csv)

    print(f"\n{'='*60}")
    print(f"  Experiment 1 complete.")
    print(f"  Best fold F1 (calibrated) : {best_f1:.4f}")
    print(f"  Mean calibrated threshold : {np.mean(cal_taus):.3f}")
    print(f"  Results  -> {csv_path}")
    print(f"  DL model -> {best_weights_path}")
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
    run_standard_cv(X, y, output_dir=args.output_dir)
