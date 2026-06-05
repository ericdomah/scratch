"""
Experiment 1 — Standard 10-Fold Stratified CV on SGCC real data.
"""
from __future__ import annotations

import os
import pickle
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from gridguard_model import GridGuardUniversalHybrid
from sgcc_pipeline import compute_tabular_features

BATCH_SIZE    = 64
PATIENCE      = 8
FIXED_TAU     = 0.5270    # documented for comparison only
DL_WEIGHT     = 0.70
XGB_WEIGHT    = 0.30


# ---------------------------------------------------------------------------
# Per-fold training
# ---------------------------------------------------------------------------

def train_one_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val:   np.ndarray, y_val:   np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    fold_num: int, output_dir: str,
    epochs: int, device: torch.device,
    pos_weight: float,
) -> dict:
    """Train one fold; return metrics dict."""

    # ------------------------------------------------------------------ XGBoost
    print(f"\n  [Fold {fold_num}] Training XGBoost (pos_weight={pos_weight:.2f}) ...")
    feat_train = compute_tabular_features(X_train)
    feat_val   = compute_tabular_features(X_val)
    feat_test  = compute_tabular_features(X_test)

    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=pos_weight, random_state=42,
        use_label_encoder=False, eval_metric="logloss",
        verbosity=0,
    )
    xgb_model.fit(feat_train, y_train.astype(int))

    # Tune tau_edge on val: route ~97% of normal consumers locally
    xgb_val_probs = xgb_model.predict_proba(feat_val)[:, 1]
    normal_mask   = y_val == 0
    if normal_mask.sum() > 0:
        tau_edge = float(np.percentile(xgb_val_probs[normal_mask], 97))
    else:
        tau_edge = 0.5
    print(f"  [Fold {fold_num}] Tuned tau_edge={tau_edge:.4f}")

    # ------------------------------------------------------------ DL model
    print(f"  [Fold {fold_num}] Training DL model for up to {epochs} epochs ...")
    model = GridGuardUniversalHybrid().to(device)

    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.FloatTensor(y_train)
    X_vl_t = torch.FloatTensor(X_val)
    y_vl_t = torch.FloatTensor(y_val)

    train_ds     = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    n_steps      = len(train_loader) * epochs

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = OneCycleLR(optimizer, max_lr=5e-4, total_steps=n_steps)

    best_auroc  = -1.0
    no_improve  = 0
    ckpt_path   = os.path.join(output_dir, f"_fold{fold_num}_ckpt.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b).squeeze()

            bce     = F.binary_cross_entropy(preds, y_b, reduction="none")
            p_t     = torch.where(y_b == 1, preds, 1 - preds)
            focal_w = (1 - p_t) ** 2.0
            alpha_t = torch.where(
                y_b == 1,
                torch.full_like(y_b, 0.92),
                torch.full_like(y_b, 0.08),
            )
            weight  = torch.where(
                y_b == 1,
                torch.full_like(y_b, float(pos_weight)),
                torch.ones_like(y_b),
            )
            loss = (alpha_t * focal_w * bce * weight).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        # Validation AUROC
        model.eval()
        with torch.no_grad():
            val_probs = model(X_vl_t.to(device)).squeeze().cpu().numpy()

        try:
            val_auroc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auroc = 0.5

        print(f"  Epoch {epoch:3d} | Loss {epoch_loss/len(train_loader):.6f} "
              f"| Val AUROC {val_auroc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (patience={PATIENCE})")
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # ------------------------------------------------------------ Fused probabilities
    X_tst_t = torch.FloatTensor(X_test)
    with torch.no_grad():
        dl_probs_val  = model(X_vl_t.to(device)).squeeze().cpu().numpy()
        dl_probs_test = model(X_tst_t.to(device)).squeeze().cpu().numpy()

    xgb_probs_val  = xgb_model.predict_proba(feat_val)[:, 1]
    xgb_probs_test = xgb_model.predict_proba(feat_test)[:, 1]

    fused_val  = DL_WEIGHT * dl_probs_val  + XGB_WEIGHT * xgb_probs_val
    fused_test = DL_WEIGHT * dl_probs_test + XGB_WEIGHT * xgb_probs_test

    # ------------------------------------------------------------ Calibrate tau on val
    best_tau, best_f1_val = FIXED_TAU, 0.0
    for tau in np.arange(0.05, 0.95, 0.01):
        y_pred_v = (fused_val >= tau).astype(int)
        f1_v     = f1_score(y_val, y_pred_v, average="binary", zero_division=0)
        if f1_v > best_f1_val:
            best_f1_val = f1_v
            best_tau    = tau

    # ------------------------------------------------------------ Diagnostic printout
    pct_above = lambda t: (fused_test >= t).mean() * 100
    print(f"\n  Fused prob stats (test):")
    print(f"    Min={fused_test.min():.4f}  Max={fused_test.max():.4f}  "
          f"Mean={fused_test.mean():.4f}")
    print(f"    % > 0.10: {pct_above(0.10):.1f}%")
    print(f"    % > 0.20: {pct_above(0.20):.1f}%")
    print(f"    % > 0.50: {pct_above(0.50):.1f}%")
    print(f"    Calibrated tau: {best_tau:.4f}")
    print(f"    Consumers above tau: {int((fused_test >= best_tau).sum())}")

    # ------------------------------------------------------------ Metrics
    y_pred_calib = (fused_test >= best_tau).astype(int)
    y_pred_fixed = (fused_test >= FIXED_TAU).astype(int)

    def _metrics(y_true, y_pred, probs):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "F1":        f1_score(y_true, y_pred, average="binary", zero_division=0),
            "Precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
            "Recall":    recall_score(y_true, y_pred, average="binary", zero_division=0),
            "AUROC":     roc_auc_score(y_true, probs),
            "Brier":     brier_score_loss(y_true, probs),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        }

    m_calib = _metrics(y_test, y_pred_calib, fused_test)
    m_fixed = _metrics(y_test, y_pred_fixed, fused_test)

    print(f"\n  [Fold {fold_num}] Calibrated tau={best_tau:.4f}:  "
          f"F1={m_calib['F1']:.4f}  AUROC={m_calib['AUROC']:.4f}  "
          f"P={m_calib['Precision']:.4f}  R={m_calib['Recall']:.4f}")
    print(f"  [Fold {fold_num}] Fixed tau={FIXED_TAU}:          "
          f"F1={m_fixed['F1']:.4f}  AUROC={m_fixed['AUROC']:.4f}")

    # ------------------------------------------------------------ Save fold model
    fold_model_path = os.path.join(output_dir, "..", "models",
                                   f"gridguard_fold{fold_num}.pth")
    os.makedirs(os.path.dirname(fold_model_path), exist_ok=True)
    torch.save(model.state_dict(), fold_model_path)

    # Tidy temp checkpoint
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    row = {
        "Fold": fold_num,
        "tau_calib": round(float(best_tau), 4),
        **{f"Fused_{k}": round(v, 6) for k, v in m_calib.items()},
        **{f"Fixed_{k}":  round(v, 6) for k, v in m_fixed.items()},
        "pos_weight": round(pos_weight, 4),
    }
    return row, xgb_model, fold_model_path


# ---------------------------------------------------------------------------
# Main CV runner
# ---------------------------------------------------------------------------

def run_standard_cv(
    X: np.ndarray, y: np.ndarray,
    output_dir: str,
    n_folds: int = 10,
    epochs: int  = 30,
) -> Tuple[float, str, str]:
    """
    10-fold stratified CV.

    Returns
    -------
    mean_tau        : float — average calibrated threshold across folds
    best_model_path : str   — path to best fold model (.pth)
    best_xgb_path   : str   — path to best fold XGBoost (.pkl)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Exp1] Device: {device}")
    print(f"[Exp1] {n_folds}-fold stratified CV | epochs={epochs}")

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    rows   = []
    best_auroc_overall = -1.0
    best_model_path    = ""
    best_xgb_path      = ""

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(X, y.astype(int))):
        fold_num = fold_idx + 1
        print(f"\n{'='*55}")
        print(f"  FOLD {fold_num}/{n_folds}")
        print(f"{'='*55}")

        X_tv, y_tv = X[train_val_idx], y[train_val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 15% of train+val as validation
        n_val = max(1, int(0.15 * len(y_tv)))
        X_train, y_train = X_tv[:-n_val], y_tv[:-n_val]
        X_val,   y_val   = X_tv[-n_val:],  y_tv[-n_val:]

        pos_weight = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))
        print(f"  Train={len(y_train):,}  Val={len(y_val):,}  "
              f"Test={len(y_test):,}  pos_weight={pos_weight:.2f}")

        row, xgb_model, fold_model_path = train_one_fold(
            X_train, y_train, X_val, y_val, X_test, y_test,
            fold_num, output_dir, epochs, device, pos_weight,
        )
        rows.append(row)

        if row["Fused_AUROC"] > best_auroc_overall:
            best_auroc_overall = row["Fused_AUROC"]
            best_model_path    = fold_model_path
            # Save best XGBoost
            best_xgb_path = os.path.join(
                output_dir, "..", "models", "xgboost_sgcc_best.pkl"
            )
            os.makedirs(os.path.dirname(best_xgb_path), exist_ok=True)
            with open(best_xgb_path, "wb") as f:
                pickle.dump(xgb_model, f)

    # ------------------------------------------------------------ Summary
    df = pd.DataFrame(rows)

    numeric_cols = [c for c in df.columns
                    if c not in ("Fold",) and pd.api.types.is_numeric_dtype(df[c])]
    mean_row = {c: df[c].mean() for c in numeric_cols}
    sd_row   = {c: df[c].std()  for c in numeric_cols}
    ci_t     = 2.262   # t_{0.975, 9} for n=10
    ci_row   = {c: ci_t * sd_row[c] / np.sqrt(n_folds) for c in numeric_cols}

    summary_row = {c: f"{mean_row[c]:.4f} +/- {sd_row[c]:.4f} "
                      f"(95% CI: {mean_row[c]-ci_row[c]:.4f} - {mean_row[c]+ci_row[c]:.4f})"
                   for c in numeric_cols}
    summary_row["Fold"] = "mean +/- SD"

    df_out = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    results_path = os.path.join(output_dir, "exp1_standard_cv.csv")
    os.makedirs(output_dir, exist_ok=True)
    df_out.to_csv(results_path, index=False)
    print(f"\n[Exp1] Results saved -> {results_path}")

    mean_tau = float(mean_row["tau_calib"])
    print(f"\n[Exp1] Summary:")
    print(f"  Fused F1    : {mean_row['Fused_F1']:.4f} +/- {sd_row['Fused_F1']:.4f}")
    print(f"  Fused AUROC : {mean_row['Fused_AUROC']:.4f} +/- {sd_row['Fused_AUROC']:.4f}")
    print(f"  Mean tau    : {mean_tau:.4f}")
    print(f"  Best model  : {best_model_path}")

    # Also copy best model to canonical path
    canonical = os.path.join(output_dir, "..", "models", "gridguard_sgcc_best.pth")
    if best_model_path and os.path.exists(best_model_path):
        import shutil
        shutil.copy2(best_model_path, canonical)
        print(f"  Best model  -> {canonical}")

    return mean_tau, canonical, best_xgb_path
