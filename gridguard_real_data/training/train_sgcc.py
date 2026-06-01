"""
Experiment 1 -- SGCC Standard Cross-Validation Training (Protocol A*)
=====================================================================
10-fold StratifiedKFold on all SGCC windows.
Also retrains the XGBoost edge filter on SGCC tabular features.

Outputs
-------
  models/gridguard_sgcc_best.pth   -- best DL fold by validation F1
  models/xgboost_sgcc_edge.pkl     -- XGBoost edge filter
  results/exp1_standard_cv.csv     -- per-fold metrics + summary
"""

from __future__ import annotations

import os
import sys
import pickle
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    classification_report,
)
from scipy.stats import t as t_dist
import xgboost as xgb
import pandas as pd
from tqdm import tqdm

# -- path setup so imports work when run directly ------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.gridguard_model import (
    GridGuardUniversalHybrid, AsymmetricFocalLoss, build_model
)
from preprocessing.sgcc_pipeline import compute_tabular_features

# -----------------------------------------------------------------------------
#  Constants
# -----------------------------------------------------------------------------
SEED          = 42
BATCH_SIZE    = 128
EPOCHS        = 25
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
MAX_LR        = 2e-3
GRAD_CLIP     = 1.0
THRESHOLD     = 0.5270
N_FOLDS       = 10
CHECKPOINT_EVERY = 5   # save intermediate checkpoint every N epochs

XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=SEED,
)

# -----------------------------------------------------------------------------
#  Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -----------------------------------------------------------------------------
#  Metrics Helper
# -----------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = THRESHOLD,
    prefix: str = "",
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        f"{prefix}F1":        f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}Precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}Recall":    recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}AUROC":     roc_auc_score(y_true, y_prob),
        f"{prefix}Brier":     brier_score_loss(y_true, y_prob),
        f"{prefix}TN": float(tn), f"{prefix}FP": float(fp),
        f"{prefix}FN": float(fn), f"{prefix}TP": float(tp),
    }
    return metrics


def ci_95(values: np.ndarray) -> float:
    """95 % CI half-width using t-distribution."""
    n  = len(values)
    sd = values.std(ddof=1)
    return t_dist.ppf(0.975, df=n - 1) * sd / np.sqrt(n)


# -----------------------------------------------------------------------------
#  Training / Inference Loop
# -----------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: str,
    grad_clip: float = GRAD_CLIP,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
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
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        p = model(X_batch).squeeze().cpu().numpy()
        if p.ndim == 0:
            p = p.reshape(1)
        probs.append(p)
    return np.concatenate(probs)


# -----------------------------------------------------------------------------
#  XGBoost Edge Filter Training
# -----------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    edge_threshold_target_fpr: float = 0.03,
) -> tuple:
    """
    Train the XGBoost edge filter and tune ?_edge to hit the target FPR
    (?3 % of normal traffic forwarded to cloud).

    Returns
    -------
    clf          : trained XGBClassifier
    tau_edge     : tuned routing threshold
    val_probs    : XGB probabilities on validation set
    """
    n_normal = (y_train == 0).sum()
    n_theft  = (y_train == 1).sum()
    spw      = n_normal / max(n_theft, 1)

    clf = xgb.XGBClassifier(
        scale_pos_weight=spw,
        **XGB_PARAMS,
    )
    clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False)

    val_probs = clf.predict_proba(X_val)[:, 1]

    # Tune ?_edge: find largest threshold where FPR ? edge_threshold_target_fpr
    normal_mask = y_val == 0
    best_tau = 0.60   # thesis default
    for tau in np.linspace(0.30, 0.95, 130):
        fpr = (val_probs[normal_mask] >= tau).mean()
        if fpr <= edge_threshold_target_fpr:
            best_tau = tau
            break

    print(f"[XGB] ?_edge tuned to {best_tau:.4f}  "
          f"(FPR on normals: "
          f"{(val_probs[normal_mask] >= best_tau).mean():.3%})")

    return clf, best_tau, val_probs


# -----------------------------------------------------------------------------
#  Late Fusion
# -----------------------------------------------------------------------------

def fuse_predictions(
    p_dl: np.ndarray,
    p_xgb: np.ndarray,
    alpha_dl: float = 0.70,
) -> np.ndarray:
    """Late fusion: 0.70 x P_DL + 0.30 x P_XGB"""
    return alpha_dl * p_dl + (1 - alpha_dl) * p_xgb


# -----------------------------------------------------------------------------
#  Main: 10-fold Standard CV  (Experiment 1)
# -----------------------------------------------------------------------------

def run_standard_cv(
    X: torch.Tensor,
    y: torch.Tensor,
    output_dir: str,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run 10-fold StratifiedKFold CV and return per-fold metric DataFrame.

    Best fold (by fusion F1) model is saved to output_dir/models/.

    Parameters
    ----------
    X          : (N, 26, 2) FloatTensor
    y          : (N,)       FloatTensor
    output_dir : project root directory
    n_folds    : number of CV folds (default 10)
    verbose    : print per-epoch progress

    Returns
    -------
    results_df : DataFrame with rows = folds + summary row
    """
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 1 -- Standard {n_folds}-Fold CV   device={device}")
    print(f"{'='*60}")

    model_dir  = os.path.join(output_dir, "models")
    result_dir = os.path.join(output_dir, "results")
    ckpt_dir   = os.path.join(output_dir, "checkpoints", "exp1")
    for d in (model_dir, result_dir, ckpt_dir):
        os.makedirs(d, exist_ok=True)

    X_np = X.numpy()
    y_np = y.numpy().astype(int)

    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_rows = []
    best_f1   = -1.0
    best_weights_path = None

    # Resume from partial
    partial_csv = os.path.join(result_dir, "exp1_standard_cv_partial.csv")
    completed_folds = set()
    if os.path.isfile(partial_csv):
        partial_df = pd.read_csv(partial_csv)
        fold_rows = partial_df.to_dict('records')
        completed_folds = {int(r["Fold"]) for r in fold_rows}
        if fold_rows:
            best_f1 = max([float(r.get("Fused_F1", -1.0)) for r in fold_rows])
        print(f"\n  [Resume] Loaded {len(completed_folds)} completed folds from partial CSV. Best F1 so far: {best_f1:.4f}")

    # --- precompute XGB tabular features once ---
    feats_np = compute_tabular_features(X_np)   # (N, 5)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y_np), 1):
        print(f"\n-- Fold {fold}/{n_folds} ------------------------------")
        if fold in completed_folds:
            print("   [SKIPPED] Already completed.")
            continue

        # DL loaders
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        train_ds = TensorDataset(X_tr, y_tr)
        val_ds   = TensorDataset(X_va, y_va)
        train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
        val_ld   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

        # Fresh model each fold
        set_seed(SEED + fold)
        model     = GridGuardUniversalHybrid().to(device)
        criterion = AsymmetricFocalLoss()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=LR, weight_decay=WEIGHT_DECAY)
        steps     = EPOCHS * len(train_ld)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR, total_steps=steps
        )

        # Training loop
        for epoch in range(1, EPOCHS + 1):
            loss = train_one_epoch(
                model, train_ld, optimizer, scheduler, criterion, device
            )
            if verbose and (epoch % 5 == 0 or epoch == 1):
                print(f"   Fold {fold}  Epoch {epoch:2d}/{EPOCHS}  "
                      f"loss={loss:.4f}")

            # Checkpoint every CHECKPOINT_EVERY epochs
            if epoch % CHECKPOINT_EVERY == 0:
                ckpt_path = os.path.join(
                    ckpt_dir, f"fold{fold}_epoch{epoch}.pth"
                )
                torch.save(model.state_dict(), ckpt_path)

        # DL probabilities on validation set
        p_dl = get_probabilities(model, val_ld, device)
        y_va_np = y_va.numpy().astype(int)

        # XGBoost on same fold split
        xgb_tr_feat = feats_np[tr_idx]
        xgb_va_feat = feats_np[va_idx]
        xgb_tr_y    = y_np[tr_idx]

        clf, tau_edge, p_xgb = train_xgboost(
            xgb_tr_feat, xgb_tr_y,
            xgb_va_feat, y_va_np,
        )

        # Late fusion
        p_fused = fuse_predictions(p_dl, p_xgb)

        # Metrics: DL-only, XGB-only, Fused
        m_dl    = compute_metrics(y_va_np, p_dl,    prefix="DL_")
        m_xgb   = compute_metrics(y_va_np, p_xgb,   prefix="XGB_")
        m_fused = compute_metrics(y_va_np, p_fused,  prefix="Fused_")

        row = {"Fold": fold, "tau_edge": round(tau_edge, 4), **m_dl, **m_xgb, **m_fused}
        fold_rows.append(row)

        f1_fused = m_fused["Fused_F1"]
        print(f"   Fold {fold}  Fused F1={f1_fused:.4f}  "
              f"AUROC={m_fused['Fused_AUROC']:.4f}  "
              f"Precision={m_fused['Fused_Precision']:.4f}  "
              f"Recall={m_fused['Fused_Recall']:.4f}")

        # Save best model
        if f1_fused > best_f1:
            best_f1 = f1_fused
            best_weights_path = os.path.join(model_dir, "gridguard_sgcc_best.pth")
            torch.save(model.state_dict(), best_weights_path)
            # Also save the XGBoost from the best fold
            best_xgb_path = os.path.join(model_dir, "xgboost_sgcc_edge.pkl")
            with open(best_xgb_path, "wb") as fh:
                pickle.dump({"clf": clf, "tau_edge": tau_edge}, fh)
            print(f"   *  New best fold (F1={best_f1:.4f}) -- weights saved")

        # Save partial progress after each fold
        pd.DataFrame(fold_rows).to_csv(partial_csv, index=False)

    # -- Summary statistics ----------------------------------------------------
    results_df = pd.DataFrame(fold_rows)

    key_cols = ["Fused_F1", "Fused_AUROC", "Fused_Precision",
                "Fused_Recall", "Fused_Brier"]
    summary: Dict = {"Fold": "mean +/- SD"}
    for col in key_cols:
        vals = results_df[col].values
        mu   = vals.mean()
        sd   = vals.std(ddof=1)
        ci   = ci_95(vals)
        summary[col] = f"{mu:.4f} +/- {sd:.4f}  [95%CI +/-{ci:.4f}]"

    summary_df = pd.DataFrame([summary])
    final_df   = pd.concat([results_df, summary_df], ignore_index=True)

    csv_path = os.path.join(result_dir, "exp1_standard_cv.csv")
    final_df.to_csv(csv_path, index=False)
    
    # Remove partial CSV since we are done
    if os.path.exists(partial_csv):
        os.remove(partial_csv)

    print(f"\n{'='*60}")
    print(f"  Experiment 1 complete.  Best fold F1 = {best_f1:.4f}")
    print(f"  Results  -> {csv_path}")
    print(f"  DL model -> {best_weights_path}")
    print(f"{'='*60}\n")

    return final_df


# -----------------------------------------------------------------------------
#  CLI entry-point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sgcc_path",   default="data/sgcc",
                    help="Dir containing SGCC CSV")
    ap.add_argument("--output_dir",  default=".",
                    help="Project root (models/ and results/ created here)")
    args = ap.parse_args()

    from preprocessing.sgcc_pipeline import run_sgcc_pipeline
    X, y, meta = run_sgcc_pipeline(
        args.sgcc_path,
        cache_path=os.path.join(args.sgcc_path, "sgcc_processed.pt"),
    )
    run_standard_cv(X, y, output_dir=args.output_dir)
