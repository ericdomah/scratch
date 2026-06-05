"""
Experiments 3 and 4 — Cross-Domain Transfer Evaluation.

Experiment 3: SGCC-trained model -> Synthetic TRNC (zero-shot)
Experiment 4: SGCC-trained model -> TDD2022      (zero-shot)

No fine-tuning. Uses mean_tau calibrated on SGCC validation sets.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, brier_score_loss, confusion_matrix,
)

from gridguard_model import GridGuardUniversalHybrid
from sgcc_pipeline import compute_tabular_features

DL_WEIGHT  = 0.70
XGB_WEIGHT = 0.30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(weights_path: str, device: torch.device) -> GridGuardUniversalHybrid:
    model = GridGuardUniversalHybrid().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def _load_xgb(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _fused_probs(model, xgb_model, X: np.ndarray, device: torch.device) -> np.ndarray:
    X_t = torch.FloatTensor(X)
    with torch.no_grad():
        dl_probs = model(X_t.to(device)).squeeze().cpu().numpy()
    feat     = compute_tabular_features(X)
    xgb_prob = xgb_model.predict_proba(feat)[:, 1]
    return DL_WEIGHT * dl_probs + XGB_WEIGHT * xgb_prob


def _metrics_row(y_true, y_pred, probs, experiment: str, domain: str, tau: float) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Experiment":  experiment,
        "Domain":      domain,
        "Tau":         round(float(tau), 4),
        "F1":          round(f1_score(y_true, y_pred, average="binary", zero_division=0), 6),
        "Precision":   round(precision_score(y_true, y_pred, average="binary", zero_division=0), 6),
        "Recall":      round(recall_score(y_true, y_pred, average="binary", zero_division=0), 6),
        "AUROC":       round(roc_auc_score(y_true, probs), 6),
        "Brier":       round(brier_score_loss(y_true, probs), 6),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "N_total":     len(y_true),
        "N_theft":     int(y_true.sum()),
    }


def _print_diagnostic(probs, label):
    pct = lambda t: (probs >= t).mean() * 100
    print(f"  Fused prob stats ({label}):")
    print(f"    Min={probs.min():.4f}  Max={probs.max():.4f}  Mean={probs.mean():.4f}")
    print(f"    % > 0.10: {pct(0.10):.1f}%   % > 0.20: {pct(0.20):.1f}%   "
          f"% > 0.50: {pct(0.50):.1f}%")


# ---------------------------------------------------------------------------
# Experiment 3 — SGCC -> Synthetic TRNC
# ---------------------------------------------------------------------------

def _run_exp3(
    output_dir: str, mean_tau: float,
    weights_path: str, xgb_path: str,
    trnc_data: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> None:
    print("\n[Exp3] SGCC -> Synthetic TRNC (zero-shot)")
    X_trnc, y_trnc = trnc_data

    model     = _load_model(weights_path, device)
    xgb_model = _load_xgb(xgb_path)

    probs  = _fused_probs(model, xgb_model, X_trnc, device)
    y_pred = (probs >= mean_tau).astype(int)

    _print_diagnostic(probs, "TRNC")
    print(f"  Tau={mean_tau:.4f}  Consumers above tau: {int(y_pred.sum())}")

    row = _metrics_row(y_trnc.astype(int), y_pred, probs,
                       "Exp3_SGCC2TRNC", "Synthetic TRNC", mean_tau)

    print(f"  F1={row['F1']:.4f}  AUROC={row['AUROC']:.4f}  "
          f"P={row['Precision']:.4f}  R={row['Recall']:.4f}")

    df = pd.DataFrame([row])
    path = os.path.join(output_dir, "exp3_reverse_transfer.csv")
    df.to_csv(path, index=False)
    print(f"[Exp3] Saved -> {path}")


# ---------------------------------------------------------------------------
# Experiment 4 — SGCC -> TDD2022
# ---------------------------------------------------------------------------

def _run_exp4(
    output_dir: str, mean_tau: float,
    weights_path: str, xgb_path: str,
    tdd_data: Tuple[np.ndarray, np.ndarray],
    device: torch.device,
) -> None:
    print("\n[Exp4] SGCC -> TDD2022 (zero-shot)")
    X_tdd, y_tdd = tdd_data

    model     = _load_model(weights_path, device)
    xgb_model = _load_xgb(xgb_path)

    probs  = _fused_probs(model, xgb_model, X_tdd, device)
    y_pred = (probs >= mean_tau).astype(int)

    _print_diagnostic(probs, "TDD2022")
    print(f"  Tau={mean_tau:.4f}  Windows above tau: {int(y_pred.sum())}")

    row = _metrics_row(y_tdd.astype(int), y_pred, probs,
                       "Exp4_SGCC2TDD", "TDD2022", mean_tau)

    print(f"  F1={row['F1']:.4f}  AUROC={row['AUROC']:.4f}  "
          f"P={row['Precision']:.4f}  R={row['Recall']:.4f}")

    df = pd.DataFrame([row])
    path = os.path.join(output_dir, "exp4_cross_domain_tdd.csv")
    df.to_csv(path, index=False)
    print(f"[Exp4] Saved -> {path}")


# ---------------------------------------------------------------------------
# Combined runner
# ---------------------------------------------------------------------------

def run_transfer_experiments(
    output_dir: str,
    mean_tau: float,
    best_fold_model_path: str,
    best_fold_xgb_path: str,
    tdd_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    trnc_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Experiment 3
    if trnc_data is None:
        print("\n[Exp3] SKIPPED — no TRNC data supplied (--trnc_path not provided).")
        print("       To run Exp3 supply: --trnc_path <path/to/trnc_synthetic_test.pt>")
    elif not os.path.exists(best_fold_model_path):
        print(f"\n[Exp3] SKIPPED — model not found at {best_fold_model_path}")
    elif not os.path.exists(best_fold_xgb_path):
        print(f"\n[Exp3] SKIPPED — XGBoost not found at {best_fold_xgb_path}")
    else:
        _run_exp3(output_dir, mean_tau, best_fold_model_path, best_fold_xgb_path,
                  trnc_data, device)

    # Experiment 4
    if tdd_data is None:
        print("\n[Exp4] SKIPPED — no TDD2022 data supplied (--tdd_path not provided).")
        print("       To run Exp4 supply: --tdd_path <path/to/tdd2022/df.csv>")
    elif not os.path.exists(best_fold_model_path):
        print(f"\n[Exp4] SKIPPED — model not found at {best_fold_model_path}")
    elif not os.path.exists(best_fold_xgb_path):
        print(f"\n[Exp4] SKIPPED — XGBoost not found at {best_fold_xgb_path}")
    else:
        _run_exp4(output_dir, mean_tau, best_fold_model_path, best_fold_xgb_path,
                  tdd_data, device)
