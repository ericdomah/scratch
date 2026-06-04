"""
Experiment 2 -- SGCC Walk-Forward Temporal Validation (Protocol A -- Primary)
============================================================================
7 expanding-window rounds.  Each round trains from scratch on a growing
prefix of the timeline and evaluates on the subsequent 14 % block.

Both GridGuardUniversalHybrid and BiGRU-BiLSTM baseline are run so that
Cohen's d significance can be computed.

Round schedule
--------------
 Round  Train end  Test window
   1       54 %    54 %?68 %
   2       61 %    61 %?75 %
   3       68 %    68 %?82 %
   4       75 %    75 %?89 %
   5       82 %    82 %?96 %
   6       89 %    89 %?100 %  (?14 %)
   7       93 %    93 %?100 %  (?7 %)  -- added to complete 7 rounds

Temporal ordering: windows sorted by absolute start week.

Outputs
-------
  results/exp2_walkforward.csv       -- per-round metrics for both models
"""

from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
)
from scipy.stats import t as t_dist
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.gridguard_model import (
    GridGuardUniversalHybrid, BiGRUBiLSTMBaseline,
    AsymmetricFocalLoss, build_model,
)

# -----------------------------------------------------------------------------
#  Constants  (same as train_sgcc.py)
# -----------------------------------------------------------------------------
SEED         = 42
BATCH_SIZE   = 2048
EPOCHS       = 25
LR           = 1e-4
WEIGHT_DECAY = 1e-4
MAX_LR       = 2e-3
GRAD_CLIP    = 1.0
THRESHOLD    = 0.5270
CHECKPOINT_EVERY = 5

# Walk-forward round boundaries (train_end %, test_end %)
# Test window = [train_end, min(train_end + 0.14, 1.0)]
ROUND_TRAIN_ENDS = [0.54, 0.61, 0.68, 0.75, 0.82, 0.89, 0.93]
TEST_WINDOW_FRAC = 0.14
WINDOW_SIZE      = 26   # must match thesis architecture (T=26 weekly timesteps)


# -----------------------------------------------------------------------------
#  Seed / Metrics (duplicated from train_sgcc for standalone usage)
# -----------------------------------------------------------------------------

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_metrics(y_true, y_prob, threshold=THRESHOLD, prefix=""):
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
        f"{prefix}F1":        f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}Precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}Recall":    recall_score(y_true, y_pred, zero_division=0),
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
    """Cohen's d: effect size between two series of F1 scores."""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return (a.mean() - b.mean()) / (pooled_std + 1e-12)


# -----------------------------------------------------------------------------
#  Training / Inference helpers
# -----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, criterion,
                    device, grad_clip=GRAD_CLIP):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_probabilities(model, loader, device):
    model.eval()
    probs = []
    for X_b, _ in loader:
        p = model(X_b.to(device)).squeeze().cpu().numpy()
        if p.ndim == 0:
            p = p.reshape(1)
        probs.append(p)
    return np.concatenate(probs)


def train_model_on_split(
    model_type: str,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test:  torch.Tensor,
    y_test:  torch.Tensor,
    device:  str,
    round_id: int,
    ckpt_dir: str,
) -> np.ndarray:
    """
    Train a fresh model on (X_train, y_train) and return probabilities on
    (X_test, y_test).

    Parameters
    ----------
    model_type : 'gridguard' | 'bigru_bilstm'
    """
    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=(device == "cuda"))
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2)

    model     = build_model(model_type, device=device)
    criterion = AsymmetricFocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    steps     = EPOCHS * len(train_ld)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=steps
    )

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(
            model, train_ld, optimizer, scheduler, criterion, device
        )
        if epoch % 5 == 0 or epoch == EPOCHS:
            print(f"     [{model_type}] Round {round_id}  "
                  f"Epoch {epoch:2d}/{EPOCHS}  loss={loss:.4f}")
        # Checkpoint
        if epoch % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(
                ckpt_dir,
                f"{model_type}_round{round_id}_epoch{epoch}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)

    return get_probabilities(model, test_ld, device)


# -----------------------------------------------------------------------------
#  Walk-Forward Split Builder
# -----------------------------------------------------------------------------

def build_wf_splits(
    win_start: np.ndarray,
    n_full_weeks: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build 7 (train_mask, test_mask) index pairs using absolute start-week
    ordering.

    Parameters
    ----------
    win_start    : (N,) array of window start weeks for each sample
    n_full_weeks : total weeks in the timeline (? 147 for SGCC)

    Returns
    -------
    splits : list of (train_indices, test_indices)
    """
    splits = []
    for train_end_frac in ROUND_TRAIN_ENDS:
        train_end = int(n_full_weeks * train_end_frac)
        test_end  = min(int(n_full_weeks * (train_end_frac + TEST_WINDOW_FRAC)),
                        n_full_weeks)

        tr_mask = win_start <  train_end
        te_mask = (win_start >= train_end) & (win_start < test_end)

        if tr_mask.sum() == 0 or te_mask.sum() == 0:
            print(f"  [WF] Skipping split: train_end={train_end_frac:.0%} -- "
                  f"empty split (train={tr_mask.sum()}, test={te_mask.sum()})")
            continue

        splits.append((np.where(tr_mask)[0], np.where(te_mask)[0]))

    return splits


# -----------------------------------------------------------------------------
#  Main: Walk-Forward Temporal Validation  (Experiment 2)
# -----------------------------------------------------------------------------

def run_walk_forward(
    X: torch.Tensor,
    y: torch.Tensor,
    metadata: dict,
    output_dir: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run 7-round expanding-window walk-forward validation for both
    GridGuardUniversalHybrid and BiGRU-BiLSTM baseline.

    Parameters
    ----------
    X          : (N, 26, 2) FloatTensor
    y          : (N,)       FloatTensor
    metadata   : dict from sgcc_pipeline (must contain 'win_start_week' and
                 'n_full_weeks')
    output_dir : project root

    Returns
    -------
    results_df : per-round metrics + summary row
    """
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 2 -- Walk-Forward Temporal Validation  device={device}")
    print(f"{'='*60}")

    result_dir = os.path.join(output_dir, "results")
    ckpt_dir   = os.path.join(output_dir, "checkpoints", "exp2")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ckpt_dir,   exist_ok=True)

    win_start    = metadata["win_start_week"]    # (N,) np array
    n_full_weeks = metadata["n_full_weeks"]

    splits = build_wf_splits(win_start, n_full_weeks)
    print(f"  Walk-forward rounds: {len(splits)}")

    # -- Guard: no valid splits (dataset too short) ----------------------------
    if len(splits) == 0:
        min_weeks_needed = int(n_full_weeks / ROUND_TRAIN_ENDS[0]) + WINDOW_SIZE
        print(f"  [WF] No valid walk-forward splits could be formed.")
        print(f"  Dataset has {n_full_weeks} weeks, window_size={WINDOW_SIZE}.")
        print(f"  win_start_week range: {win_start.min()} to {win_start.max()}")
        print(f"  The 54% cutoff falls at week {int(n_full_weeks * 0.54)}, but")
        print(f"  all windows start before that -- no samples remain for testing.")
        print(f"  Root cause: likely using synthetic stub data (120 rows, 52 weeks)")
        print(f"  instead of real SGCC (~42k rows, ~147 weeks).")
        print(f"  Fix: upload the real SGCC data.csv to data/sgcc/ and delete")
        print(f"  data/sgcc/sgcc_processed.pt, then re-run.")
        empty_df = pd.DataFrame(columns=[
            "Round", "Train_samples", "Test_samples", "Theft_frac_test",
            "GG_F1", "GG_AUROC", "GG_Precision", "GG_Recall", "GG_Brier",
            "Base_F1", "Base_AUROC", "Base_Precision", "Base_Recall",
        ])
        csv_path = os.path.join(result_dir, "exp2_walkforward.csv")
        empty_df.to_csv(csv_path, index=False)
        print(f"  Empty results saved to {csv_path}")
        print(f"{'='*60}\n")
        return empty_df

    y_np = y.numpy().astype(int)
    rows = []

    gg_f1s   = []   # GridGuard F1 per round
    base_f1s = []   # Baseline F1 per round

    for rnd, (tr_idx, te_idx) in enumerate(splits, 1):
        print(f"\n-- Round {rnd}/{len(splits)} ----------------------------")
        print(f"   Train: {len(tr_idx):,} samples  |  "
              f"Test: {len(te_idx):,} samples  |  "
              f"Theft (train): {y_np[tr_idx].mean():.3%}  |  "
              f"Theft (test): {y_np[te_idx].mean():.3%}")

        X_tr = X[tr_idx];  y_tr = y[tr_idx]
        X_te = X[te_idx];  y_te = y[te_idx]
        y_te_np = y_np[te_idx]

        # -- GridGuardUniversalHybrid ------------------------------------------
        set_seed(SEED + rnd)
        print(f"  Training GridGuardUniversalHybrid...")
        p_gg = train_model_on_split(
            "gridguard", X_tr, y_tr, X_te, y_te, device, rnd, ckpt_dir
        )
        m_gg = compute_metrics(y_te_np, p_gg, prefix="GG_")
        gg_f1s.append(m_gg["GG_F1"])

        # -- BiGRU-BiLSTM Baseline ---------------------------------------------
        set_seed(SEED + rnd + 100)
        print(f"  Training BiGRU-BiLSTM Baseline...")
        p_base = train_model_on_split(
            "bigru_bilstm", X_tr, y_tr, X_te, y_te, device, rnd, ckpt_dir
        )
        m_base = compute_metrics(y_te_np, p_base, prefix="Base_")
        base_f1s.append(m_base["Base_F1"])

        row = {
            "Round":          rnd,
            "Train_samples":  len(tr_idx),
            "Test_samples":   len(te_idx),
            "Theft_frac_test": round(float(y_te_np.mean()), 4),
            **m_gg,
            **m_base,
        }
        rows.append(row)

        print(f"  GG   F1={m_gg['GG_F1']:.4f}  "
              f"AUROC={m_gg['GG_AUROC']:.4f}  "
              f"Prec={m_gg['GG_Precision']:.4f}  "
              f"Rec={m_gg['GG_Recall']:.4f}")
        print(f"  Base F1={m_base['Base_F1']:.4f}  "
              f"AUROC={m_base['Base_AUROC']:.4f}  "
              f"Prec={m_base['Base_Precision']:.4f}  "
              f"Rec={m_base['Base_Recall']:.4f}")

    # -- Summary ---------------------------------------------------------------
    results_df = pd.DataFrame(rows)

    if results_df.empty:
        csv_path = os.path.join(result_dir, "exp2_walkforward.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"  No rounds completed.  Results -> {csv_path}")
        print(f"{'='*60}\n")
        return results_df

    gg_f1s   = np.array(gg_f1s)
    base_f1s = np.array(base_f1s)

    d = cohens_d(gg_f1s, base_f1s)

    for metric in ["GG_F1", "GG_AUROC", "GG_Precision", "GG_Recall",
                   "Base_F1", "Base_AUROC"]:
        vals = results_df[metric].values.astype(float)
        mu   = vals.mean()
        sd   = vals.std(ddof=1)
        ci   = ci_95(vals)
        print(f"  {metric}: {mu:.4f} +/- {sd:.4f}  [95%CI +/-{ci:.4f}]")

    print(f"\n  Cohen's d (GridGuard vs Baseline, F1): {d:.4f}")

    summary_row = {
        "Round":    "mean +/- SD",
        "GG_F1":   f"{gg_f1s.mean():.4f} +/- {gg_f1s.std(ddof=1):.4f} [95%CI +/-{ci_95(gg_f1s):.4f}]",
        "Base_F1": f"{base_f1s.mean():.4f} +/- {base_f1s.std(ddof=1):.4f} [95%CI +/-{ci_95(base_f1s):.4f}]",
        "Cohens_d": round(d, 4),
    }
    summary_df = pd.DataFrame([summary_row])
    final_df   = pd.concat([results_df, summary_df], ignore_index=True)

    csv_path = os.path.join(result_dir, "exp2_walkforward.csv")
    final_df.to_csv(csv_path, index=False)

    print(f"\n  Results -> {csv_path}")
    print(f"{'='*60}\n")

    return final_df


# -----------------------------------------------------------------------------
#  CLI entry-point
# -----------------------------------------------------------------------------

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
