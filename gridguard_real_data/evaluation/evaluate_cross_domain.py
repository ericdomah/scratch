"""
Experiments 3 & 4 — Cross-Domain Zero-Shot Evaluation
=======================================================
Experiment 3: SGCC-trained model  →  Synthetic TRNC (Reverse Transfer)
Experiment 4: SGCC-trained model  →  TDD2022      (Cross-Domain)

Both experiments:
  - Load best weights from models/gridguard_sgcc_best.pth (Experiment 1 best fold)
  - No fine-tuning — zero-shot evaluation only
  - Also apply the saved XGBoost for late fusion where tabular features exist
  - Report: F1, Precision, Recall, AUROC, Brier, degradation % vs SGCC in-domain

For Experiment 3 (Synthetic TRNC): the synthetic test partition is expected
at data/trnc_synthetic_test.pt  (created by the original thesis pipeline).
If not present, Experiment 3 is skipped with a clear message.

Outputs
-------
  results/exp3_reverse_transfer.csv
  results/exp4_cross_domain_tdd.csv
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
    classification_report,
)
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.gridguard_model import GridGuardUniversalHybrid
from preprocessing.sgcc_pipeline import compute_tabular_features

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 128
THRESHOLD  = 0.5270

# Reference SGCC in-domain metrics (from Experiment 1, to compute degradation)
# Updated automatically once Experiment 1 completes; hard-coded fallbacks:
SGCC_INDOMAIN_F1    = None   # will be read from exp1 CSV if available
SGCC_INDOMAIN_AUROC = None


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_best_model(weights_path: str, device: str) -> GridGuardUniversalHybrid:
    model = GridGuardUniversalHybrid().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_xgboost(pkl_path: str) -> Tuple[object, float]:
    """Returns (xgb_classifier, tau_edge)."""
    with open(pkl_path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):
        return obj["clf"], obj.get("tau_edge", 0.60)
    return obj, 0.60


@torch.no_grad()
def get_probabilities(
    model: nn.Module,
    X: torch.Tensor,
    device: str,
) -> np.ndarray:
    ds     = TensorDataset(X, torch.zeros(len(X)))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    probs  = []
    for X_b, _ in loader:
        p = model(X_b.to(device)).squeeze().cpu().numpy()
        if p.ndim == 0:
            p = p.reshape(1)
        probs.append(p)
    return np.concatenate(probs)


def fuse_predictions(p_dl, p_xgb, alpha_dl=0.70):
    return alpha_dl * p_dl + (1 - alpha_dl) * p_xgb


def compute_metrics(y_true, y_prob, threshold=THRESHOLD):
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
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "AUROC":     auroc,
        "Brier":     brier_score_loss(y_true, y_prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def degradation_pct(in_domain_val: Optional[float], out_val: float) -> str:
    if in_domain_val is None or in_domain_val == 0:
        return "N/A"
    return f"{(in_domain_val - out_val) / in_domain_val * 100:+.2f}%"


def load_sgcc_indomain_metrics(result_dir: str) -> Tuple[Optional[float], Optional[float]]:
    """Read Fused_F1 and Fused_AUROC means from exp1_standard_cv.csv."""
    csv_path = os.path.join(result_dir, "exp1_standard_cv.csv")
    if not os.path.isfile(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    # summary row has 'mean ± SD' in Fold column
    summary = df[df["Fold"].astype(str).str.startswith("mean")]
    if summary.empty:
        # fallback: use numeric rows mean
        numeric = df[pd.to_numeric(df["Fold"], errors="coerce").notna()]
        f1   = pd.to_numeric(numeric["Fused_F1"],   errors="coerce").mean()
        auroc = pd.to_numeric(numeric["Fused_AUROC"], errors="coerce").mean()
        return float(f1), float(auroc)
    row = summary.iloc[0]
    # format: "0.9123 ± 0.0045  [95%CI ±0.0034]"
    def parse_mean(s: str) -> Optional[float]:
        try:
            return float(str(s).split("±")[0].strip())
        except Exception:
            return None
    return parse_mean(row.get("Fused_F1")), parse_mean(row.get("Fused_AUROC"))


# ─────────────────────────────────────────────────────────────────────────────
#  Experiment 3 — Synthetic TRNC Reverse Transfer
# ─────────────────────────────────────────────────────────────────────────────

def run_exp3_reverse_transfer(
    weights_path: str,
    xgb_path: str,
    trnc_test_path: str,
    output_dir: str,
    sgcc_f1: Optional[float] = None,
    sgcc_auroc: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """
    Evaluate SGCC-trained model on the synthetic TRNC test partition.

    Parameters
    ----------
    trnc_test_path : path to synthetic test .pt file created by thesis pipeline.
                     Expected keys: 'X' (N,26,2) FloatTensor and 'y' (N,) FloatTensor.
                     If absent, experiment is skipped.
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 3 — Reverse Transfer: SGCC → Synthetic TRNC")
    print(f"{'='*60}")

    if not os.path.isfile(trnc_test_path):
        print(f"  [SKIP] Synthetic TRNC test file not found: {trnc_test_path}")
        print(f"  Place the synthetic test partition at that path and re-run.")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load test data
    ckpt = torch.load(trnc_test_path, map_location="cpu")
    X = ckpt["X"].float()
    y = ckpt["y"].float()
    print(f"  TRNC test: {X.shape}  theft={y.mean():.3%}")

    # Load model
    if not os.path.isfile(weights_path):
        print(f"  [ERROR] Model weights not found: {weights_path}")
        print(f"  Run Experiment 1 first.")
        return None

    model = load_best_model(weights_path, device)
    p_dl  = get_probabilities(model, X, device)
    y_np  = y.numpy().astype(int)

    # XGBoost (if available)
    if os.path.isfile(xgb_path):
        clf, tau_edge = load_xgboost(xgb_path)
        feats = compute_tabular_features(X.numpy())
        p_xgb  = clf.predict_proba(feats)[:, 1]
        p_final = fuse_predictions(p_dl, p_xgb)
        label  = "SGCC-trained (Fused) → TRNC zero-shot"
    else:
        p_final = p_dl
        label   = "SGCC-trained (DL-only) → TRNC zero-shot"
        print("  [WARN] XGBoost model not found — using DL-only probabilities")

    metrics = compute_metrics(y_np, p_final)
    print(f"\n  {label}")
    print(f"  F1={metrics['F1']:.4f}  AUROC={metrics['AUROC']:.4f}  "
          f"Prec={metrics['Precision']:.4f}  Rec={metrics['Recall']:.4f}  "
          f"Brier={metrics['Brier']:.4f}")

    if sgcc_f1 is not None:
        deg_f1   = degradation_pct(sgcc_f1,   metrics["F1"])
        deg_auroc = degradation_pct(sgcc_auroc, metrics["AUROC"])
        print(f"  Degradation vs SGCC in-domain:  F1={deg_f1}  AUROC={deg_auroc}")
        metrics["Degrad_F1"]    = deg_f1
        metrics["Degrad_AUROC"] = deg_auroc

    print(f"\n  Classification Report:")
    y_pred = (p_final >= THRESHOLD).astype(int)
    print(classification_report(y_np, y_pred, target_names=["Normal", "Theft"],
                                zero_division=0))

    row_df = pd.DataFrame([{"Experiment": label, **metrics}])
    csv_path = os.path.join(output_dir, "results", "exp3_reverse_transfer.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row_df.to_csv(csv_path, index=False)
    print(f"  Results → {csv_path}")

    return row_df


# ─────────────────────────────────────────────────────────────────────────────
#  Experiment 4 — TDD2022 Cross-Domain
# ─────────────────────────────────────────────────────────────────────────────

def run_exp4_cross_domain_tdd(
    weights_path: str,
    xgb_path: str,
    tdd_path: str,
    output_dir: str,
    sgcc_f1: Optional[float] = None,
    sgcc_auroc: Optional[float] = None,
) -> Optional[pd.DataFrame]:
    """
    Evaluate SGCC-trained model on TDD2022 dataset (zero-shot).

    Parameters
    ----------
    tdd_path : directory containing TDD2022 CSV
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 4 — Cross-Domain: SGCC → TDD2022")
    print(f"{'='*60}")

    if not os.path.isdir(tdd_path) or not any(
        f.endswith(".csv") for f in os.listdir(tdd_path)
    ):
        print(f"  [SKIP] TDD2022 CSV not found in: {tdd_path}")
        print(f"  Download from https://data.mendeley.com/datasets/c3c7329tjj/1")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load & preprocess TDD2022
    from preprocessing.tdd2022_pipeline import run_tdd2022_pipeline
    cache = os.path.join(tdd_path, "tdd2022_processed.pt")
    X, y, meta = run_tdd2022_pipeline(tdd_path, cache_path=cache)
    y_np = y.numpy().astype(int)
    print(f"  TDD2022: {X.shape}  theft={y.mean():.3%}")

    if not os.path.isfile(weights_path):
        print(f"  [ERROR] Model weights not found: {weights_path}")
        return None

    model = load_best_model(weights_path, device)
    p_dl  = get_probabilities(model, X, device)

    if os.path.isfile(xgb_path):
        clf, _ = load_xgboost(xgb_path)
        feats   = compute_tabular_features(X.numpy())
        p_xgb   = clf.predict_proba(feats)[:, 1]
        p_final = fuse_predictions(p_dl, p_xgb)
        label   = "SGCC-trained (Fused) → TDD2022 zero-shot"
    else:
        p_final = p_dl
        label   = "SGCC-trained (DL-only) → TDD2022 zero-shot"
        print("  [WARN] XGBoost model not found — using DL-only probabilities")

    metrics = compute_metrics(y_np, p_final)
    print(f"\n  {label}")
    print(f"  F1={metrics['F1']:.4f}  AUROC={metrics['AUROC']:.4f}  "
          f"Prec={metrics['Precision']:.4f}  Rec={metrics['Recall']:.4f}  "
          f"Brier={metrics['Brier']:.4f}")

    if sgcc_f1 is not None:
        deg_f1    = degradation_pct(sgcc_f1,   metrics["F1"])
        deg_auroc = degradation_pct(sgcc_auroc, metrics["AUROC"])
        print(f"  Degradation vs SGCC in-domain:  F1={deg_f1}  AUROC={deg_auroc}")
        metrics["Degrad_F1"]    = deg_f1
        metrics["Degrad_AUROC"] = deg_auroc

    print(f"\n  Classification Report:")
    y_pred = (p_final >= THRESHOLD).astype(int)
    print(classification_report(y_np, y_pred, target_names=["Normal", "Theft"],
                                zero_division=0))

    row_df = pd.DataFrame([{"Experiment": label, **metrics}])
    csv_path = os.path.join(output_dir, "results", "exp4_cross_domain_tdd.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    row_df.to_csv(csv_path, index=False)
    print(f"  Results → {csv_path}")

    return row_df


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",    default="models/gridguard_sgcc_best.pth")
    ap.add_argument("--xgb",        default="models/xgboost_sgcc_edge.pkl")
    ap.add_argument("--trnc_test",  default="data/trnc_synthetic_test.pt",
                    help="Synthetic TRNC holdout .pt file")
    ap.add_argument("--tdd_path",   default="data/tdd2022")
    ap.add_argument("--output_dir", default=".")
    args = ap.parse_args()

    result_dir = os.path.join(args.output_dir, "results")
    sgcc_f1, sgcc_auroc = load_sgcc_indomain_metrics(result_dir)

    run_exp3_reverse_transfer(
        args.weights, args.xgb, args.trnc_test, args.output_dir,
        sgcc_f1=sgcc_f1, sgcc_auroc=sgcc_auroc,
    )
    run_exp4_cross_domain_tdd(
        args.weights, args.xgb, args.tdd_path, args.output_dir,
        sgcc_f1=sgcc_f1, sgcc_auroc=sgcc_auroc,
    )
