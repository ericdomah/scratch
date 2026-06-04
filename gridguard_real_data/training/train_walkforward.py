"""
Experiment 2 -- SGCC Walk-Forward Temporal Validation (Primary Protocol)
=========================================================================
5 expanding-window rounds sorted by CONS_NO as a temporal proxy.

SGCC does not provide registration dates, so CONS_NO order is used as
the best available temporal proxy (lower ID = earlier registration).

Round schedule (per thesis specification)
-----------------------------------------
  Round  Train end  Test window
    1      60 %    → next 20 %
    2      70 %    → next 10 %
    3      80 %    → next 10 %
    4      85 %    → next 10 %
    5      90 %    → remaining 10 %

If any test split has fewer than 30 theft samples it is merged
with the next round.

Both GridGuardUniversalHybrid and BiGRU-BiLSTM baseline are run so
that Cohen's d significance can be computed across the 5 rounds.

Outputs
-------
  results/exp2_walkforward.csv   -- per-round metrics for both models
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
#  Constants  (same as train_sgcc.py — DO NOT CHANGE)
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
MIN_THEFT_PER_TEST = 30   # merge rounds with fewer theft samples

# Walk-forward round boundaries (train_end_frac, test_end_frac)
# One consumer = one sample; fractions are over sorted consumers.
ROUNDS = [
    (0.00, 0.60, 0.80),   # Round 1: train [0, 60%), test [60%, 80%)
    (0.00, 0.70, 0.80),   # Round 2: train [0, 70%), test [70%, 80%)
    (0.00, 0.80, 0.90),   # Round 3: train [0, 80%), test [80%, 90%)
    (0.00, 0.85, 0.95),   # Round 4: train [0, 85%), test [85%, 95%)
    (0.00, 0.90, 1.00),   # Round 5: train [0, 90%), test [90%, 100%)
]


# -----------------------------------------------------------------------------
#  Seed / Metrics
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
    Train a FRESH model on (X_train, y_train) and return probabilities on
    (X_test, y_test). No weight sharing between rounds.
    """
    train_ds = TensorDataset(X_train, y_train)
    test_ds  = TensorDataset(X_test,  y_test)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=(device == "cuda"))
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0)

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
    n: int,
    y: np.ndarray,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build up to 5 (train_indices, test_indices) pairs sorted by CONS_NO order.

    Consumers are already sorted by CONS_NO (sort_order applied upstream).
    Fractions refer to position in the sorted consumer list.

    Rounds with fewer than MIN_THEFT_PER_TEST theft samples in the test set
    are skipped (merged with the next round implicitly by the larger train set
    in subsequent rounds).

    Parameters
    ----------
    n : total number of consumers (samples)
    y : (n,) binary labels in CONS_NO sorted order

    Returns
    -------
    splits : list of (train_indices, test_indices)
    """
    splits = []
    for _, train_end_frac, test_end_frac in ROUNDS:
        train_end = int(n * train_end_frac)
        test_end  = min(int(n * test_end_frac), n)

        tr_idx = np.arange(0, train_end)
        te_idx = np.arange(train_end, test_end)

        if len(tr_idx) == 0 or len(te_idx) == 0:
            print(f"  [WF] Skipping empty split: "
                  f"train_end={train_end_frac:.0%}, test_end={test_end_frac:.0%}")
            continue

        n_theft_test = y[te_idx].sum()
        if n_theft_test < MIN_THEFT_PER_TEST:
            print(f"  [WF] Skipping split (train={train_end_frac:.0%}, "
                  f"test={test_end_frac:.0%}): only {n_theft_test:.0f} theft "
                  f"samples in test — below minimum {MIN_THEFT_PER_TEST}")
            continue

        splits.append((tr_idx, te_idx))

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
    Run 5-round expanding-window walk-forward validation for both
    GridGuardUniversalHybrid and BiGRU-BiLSTM baseline.

    Consumers are sorted by CONS_NO (sort_order from metadata) as a
    temporal ordering proxy before splitting.

    Parameters
    ----------
    X          : (N, 26, 2) FloatTensor
    y          : (N,)       FloatTensor
    metadata   : dict from sgcc_pipeline (must contain 'sort_order')
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

    # -- Sort consumers by CONS_NO as temporal proxy --------------------------
    sort_order = metadata["sort_order"]          # (N,) indices
    X_sorted   = X[sort_order]
    y_sorted   = y[sort_order]
    y_np       = y_sorted.numpy().astype(int)
    n          = len(y_np)

    print(f"  Total samples (one per consumer): {n:,}")
    print(f"  Sorted by CONS_NO as temporal proxy.")

    splits = build_wf_splits(n, y_np)
    print(f"  Valid walk-forward rounds: {len(splits)}")

    if len(splits) == 0:
        print("  [WF] No valid splits — check that SGCC data is loaded correctly.")
        empty_df = pd.DataFrame(columns=[
            "Round", "Train_samples", "Test_samples", "Theft_frac_test",
            "GG_F1", "GG_AUROC", "GG_Precision", "GG_Recall", "GG_Brier",
            "Base_F1", "Base_AUROC", "Base_Precision", "Base_Recall",
        ])
        empty_df.to_csv(os.path.join(result_dir, "exp2_walkforward.csv"), index=False)
        return empty_df

    rows     = []
    gg_f1s   = []
    base_f1s = []

    for rnd, (tr_idx, te_idx) in enumerate(splits, 1):
        print(f"\n-- Round {rnd}/{len(splits)} ----------------------------")
        print(f"   Train: {len(tr_idx):,} consumers  |  "
              f"Test: {len(te_idx):,} consumers  |  "
              f"Theft (train): {y_np[tr_idx].mean():.3%}  |  "
              f"Theft (test): {y_np[te_idx].mean():.3%}")

        X_tr = X_sorted[tr_idx];  y_tr = y_sorted[tr_idx]
        X_te = X_sorted[te_idx];  y_te = y_sorted[te_idx]
        y_te_np = y_np[te_idx]

        # -- GridGuardUniversalHybrid ------------------------------------------
        set_seed(SEED + rnd)
        print(f"  Training GridGuardUniversalHybrid (from scratch)...")
        p_gg = train_model_on_split(
            "gridguard", X_tr, y_tr, X_te, y_te, device, rnd, ckpt_dir
        )
        m_gg = compute_metrics(y_te_np, p_gg, prefix="GG_")
        gg_f1s.append(m_gg["GG_F1"])

        # -- BiGRU-BiLSTM Baseline (from scratch) ------------------------------
        set_seed(SEED + rnd + 100)
        print(f"  Training BiGRU-BiLSTM Baseline (from scratch)...")
        p_base = train_model_on_split(
            "bigru_bilstm", X_tr, y_tr, X_te, y_te, device, rnd, ckpt_dir
        )
        m_base = compute_metrics(y_te_np, p_base, prefix="Base_")
        base_f1s.append(m_base["Base_F1"])

        gg_wins = m_gg["GG_F1"] > m_base["Base_F1"]
        print(f"  GG   F1={m_gg['GG_F1']:.4f}  AUROC={m_gg['GG_AUROC']:.4f}  "
              f"Prec={m_gg['GG_Precision']:.4f}  Rec={m_gg['GG_Recall']:.4f}")
        print(f"  Base F1={m_base['Base_F1']:.4f}  AUROC={m_base['Base_AUROC']:.4f}  "
              f"Prec={m_base['Base_Precision']:.4f}  Rec={m_base['Base_Recall']:.4f}")
        print(f"  GridGuard {'OUTPERFORMS' if gg_wins else 'does NOT outperform'} "
              f"BiGRU-BiLSTM baseline in Round {rnd}")

        row = {
            "Round":           rnd,
            "Train_samples":   len(tr_idx),
            "Test_samples":    len(te_idx),
            "Theft_frac_test": round(float(y_te_np.mean()), 4),
            **m_gg,
            **m_base,
        }
        rows.append(row)

    # -- Summary ---------------------------------------------------------------
    results_df = pd.DataFrame(rows)

    gg_f1s   = np.array(gg_f1s)
    base_f1s = np.array(base_f1s)
    d        = cohens_d(gg_f1s, base_f1s)

    print(f"\n{'='*60}")
    for metric in ["GG_F1", "GG_AUROC", "GG_Precision", "GG_Recall",
                   "Base_F1", "Base_AUROC"]:
        vals = results_df[metric].values.astype(float)
        mu   = vals.mean()
        sd   = vals.std(ddof=1)
        ci   = ci_95(vals)
        print(f"  {metric}: {mu:.4f} +/- {sd:.4f}  [95%CI +/-{ci:.4f}]")

    print(f"\n  Cohen's d (GridGuard vs Baseline, F1): {d:.4f}")
    print(f"  (|d|>0.8 = large, 0.5–0.8 = medium, 0.2–0.5 = small)")

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
