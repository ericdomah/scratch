"""
Experiment 2 — Walk-Forward Temporal Validation on SGCC real data.
5 rounds with expanding training windows, both GridGuard and BiGRU-BiLSTM
trained from scratch per round.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, confusion_matrix
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from gridguard_model import GridGuardUniversalHybrid, BiGRUBiLSTMBaseline
from sgcc_pipeline import compute_tabular_features

BATCH_SIZE = 64
PATIENCE   = 8
DL_WEIGHT  = 0.70
XGB_WEIGHT = 0.30
FIXED_TAU  = 0.5270

# Walk-forward splits: (train_end_frac, test_end_frac)
WF_SPLITS = [
    (0.60, 0.80),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.85, 0.95),
    (0.90, 1.00),
]


def _train_dl(model, X_train, y_train, X_val, y_val, epochs, device, pos_weight):
    """Train a DL model with focal loss + OneCycleLR. Return best-AUROC weights."""
    X_tr_t = torch.FloatTensor(X_train)
    y_tr_t = torch.FloatTensor(y_train)
    X_vl_t = torch.FloatTensor(X_val)
    y_vl_t = torch.FloatTensor(y_val)

    loader   = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    n_steps  = len(loader) * epochs
    optim    = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched    = OneCycleLR(optim, max_lr=5e-4, total_steps=n_steps)

    best_auroc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b).squeeze()

            bce     = F.binary_cross_entropy(preds, y_b, reduction="none")
            p_t     = torch.where(y_b == 1, preds, 1 - preds)
            focal_w = (1 - p_t) ** 2.0
            alpha_t = torch.where(y_b == 1,
                                  torch.full_like(y_b, 0.92),
                                  torch.full_like(y_b, 0.08))
            weight  = torch.where(y_b == 1,
                                  torch.full_like(y_b, float(pos_weight)),
                                  torch.ones_like(y_b))
            loss = (alpha_t * focal_w * bce * weight).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            ep_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_probs = model(X_vl_t.to(device)).squeeze().cpu().numpy()
        try:
            val_auroc = roc_auc_score(y_val, val_probs)
        except ValueError:
            val_auroc = 0.5

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _calibrate_tau(fused_val, y_val):
    best_tau, best_f1 = FIXED_TAU, 0.0
    for tau in np.arange(0.05, 0.95, 0.01):
        y_pred = (fused_val >= tau).astype(int)
        f1     = f1_score(y_val, y_pred, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = tau
    return float(best_tau)


def _eval_fused(model, xgb_model, X, y, device):
    X_t = torch.FloatTensor(X)
    model.eval()
    with torch.no_grad():
        dl_probs = model(X_t.to(device)).squeeze().cpu().numpy()
    feat = compute_tabular_features(X)
    xgb_probs = xgb_model.predict_proba(feat)[:, 1]
    fused = DL_WEIGHT * dl_probs + XGB_WEIGHT * xgb_probs
    return fused


def _prob_diagnostic(fused, label):
    pct = lambda t: (fused >= t).mean() * 100
    print(f"  Fused prob stats ({label}):")
    print(f"    Min={fused.min():.4f}  Max={fused.max():.4f}  Mean={fused.mean():.4f}")
    print(f"    % > 0.10: {pct(0.10):.1f}%   % > 0.20: {pct(0.20):.1f}%   "
          f"% > 0.50: {pct(0.50):.1f}%")


def _metrics(y_true, y_pred, probs):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        F1        = f1_score(y_true, y_pred, average="binary", zero_division=0),
        AUROC     = roc_auc_score(y_true, probs),
        Brier     = brier_score_loss(y_true, probs),
        TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
    )


def run_walkforward(
    X: np.ndarray, y: np.ndarray,
    output_dir: str,
    epochs: int = 30,
) -> float:
    """
    Walk-forward temporal validation.
    Returns mean calibrated threshold across rounds.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n      = len(y)
    print(f"\n[Exp2] Walk-forward | device={device} | N={n:,} | epochs={epochs}")

    rows = []

    for round_idx, (train_end_frac, test_end_frac) in enumerate(WF_SPLITS):
        round_num  = round_idx + 1
        train_end  = int(n * train_end_frac)
        test_end   = int(n * test_end_frac)
        test_start = train_end

        X_train_all = X[:train_end]
        y_train_all = y[:train_end]
        X_test      = X[test_start:test_end]
        y_test      = y[test_start:test_end]

        # Verify at least 30 theft consumers in test
        n_theft_test = int((y_test == 1).sum())
        if n_theft_test < 30:
            print(f"  [Round {round_num}] Only {n_theft_test} theft in test — merging with next")
            continue

        # Validation = last 15% of training set
        n_val   = max(1, int(0.15 * len(y_train_all)))
        X_train = X_train_all[:-n_val]
        y_train = y_train_all[:-n_val]
        X_val   = X_train_all[-n_val:]
        y_val   = y_train_all[-n_val:]

        pos_weight = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))

        print(f"\n{'='*55}")
        print(f"  === Round {round_num}/5 ===")
        print(f"  Train={len(y_train):,}  Val={len(y_val):,}  "
              f"Test={len(y_test):,}  theft_test={n_theft_test}")
        print(f"  pos_weight={pos_weight:.2f}")

        # ---- XGBoost (same for both models)
        feat_train = compute_tabular_features(X_train)
        feat_val   = compute_tabular_features(X_val)
        feat_test  = compute_tabular_features(X_test)

        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=pos_weight, random_state=42,
            use_label_encoder=False, eval_metric="logloss", verbosity=0,
        )
        xgb_model.fit(feat_train, y_train.astype(int))

        # ---- GridGuard
        print(f"  [Round {round_num}] Training GridGuard from scratch ...")
        gg_model = GridGuardUniversalHybrid().to(device)
        gg_model = _train_dl(gg_model, X_train, y_train, X_val, y_val,
                             epochs, device, pos_weight)

        fused_gg_val  = _eval_fused(gg_model, xgb_model, X_val,  y_val,  device)
        fused_gg_test = _eval_fused(gg_model, xgb_model, X_test, y_test, device)

        gg_tau = _calibrate_tau(fused_gg_val, y_val)
        _prob_diagnostic(fused_gg_test, f"GridGuard round {round_num}")
        print(f"  Calibrated tau (GG): {gg_tau:.4f}  "
              f"Consumers above tau: {int((fused_gg_test >= gg_tau).sum())}")

        y_pred_gg = (fused_gg_test >= gg_tau).astype(int)
        m_gg = _metrics(y_test, y_pred_gg, fused_gg_test)

        # ---- Baseline BiGRU-BiLSTM
        print(f"  [Round {round_num}] Training BiGRU-BiLSTM baseline from scratch ...")
        base_model = BiGRUBiLSTMBaseline().to(device)
        base_model = _train_dl(base_model, X_train, y_train, X_val, y_val,
                               epochs, device, pos_weight)

        # Baseline uses same XGBoost
        fused_base_val  = _eval_fused(base_model, xgb_model, X_val,  y_val,  device)
        fused_base_test = _eval_fused(base_model, xgb_model, X_test, y_test, device)

        base_tau = _calibrate_tau(fused_base_val, y_val)
        _prob_diagnostic(fused_base_test, f"Baseline round {round_num}")

        y_pred_base = (fused_base_test >= base_tau).astype(int)
        m_base = _metrics(y_test, y_pred_base, fused_base_test)

        winner = "WINS" if m_gg["F1"] > m_base["F1"] else "LOSES"
        print(f"\n  === Round {round_num}/5 ===")
        print(f"  GridGuard F1 : {m_gg['F1']:.4f} at tau={gg_tau:.4f}")
        print(f"  Baseline  F1 : {m_base['F1']:.4f}")
        print(f"  GridGuard {winner} by {abs(m_gg['F1'] - m_base['F1']):.4f}")

        rows.append({
            "Round":      round_num,
            "GG_F1":      round(m_gg["F1"],    4),
            "GG_AUROC":   round(m_gg["AUROC"],  4),
            "GG_Brier":   round(m_gg["Brier"],  6),
            "GG_tau":     round(gg_tau, 4),
            "Base_F1":    round(m_base["F1"],   4),
            "Base_AUROC": round(m_base["AUROC"], 4),
            "Base_Brier": round(m_base["Brier"], 6),
            "Base_tau":   round(base_tau, 4),
            "pos_weight": round(pos_weight, 4),
        })

    if not rows:
        raise RuntimeError("No valid walk-forward rounds completed — check data size.")

    df = pd.DataFrame(rows)
    numeric_cols = [c for c in df.columns if c != "Round"]

    mean_row = {c: df[c].mean() for c in numeric_cols}
    sd_row   = {c: df[c].std()  for c in numeric_cols}
    n_rounds = len(df)
    ci_t     = 2.776   # t_{0.975, 4} for n=5
    ci_row   = {c: ci_t * sd_row[c] / np.sqrt(n_rounds) for c in numeric_cols}

    # Cohen's d for GG vs Baseline F1
    if sd_row["GG_F1"] > 0 or sd_row["Base_F1"] > 0:
        pooled_sd = np.sqrt((sd_row["GG_F1"] ** 2 + sd_row["Base_F1"] ** 2) / 2)
        cohens_d  = (mean_row["GG_F1"] - mean_row["Base_F1"]) / (pooled_sd + 1e-12)
    else:
        cohens_d = 0.0

    summary_row = {c: f"{mean_row[c]:.4f} +/- {sd_row[c]:.4f} "
                      f"(95% CI: {mean_row[c]-ci_row[c]:.4f} - "
                      f"{mean_row[c]+ci_row[c]:.4f})"
                   for c in numeric_cols}
    summary_row["Round"]    = "mean +/- SD"
    summary_row["Cohens_d"] = f"{cohens_d:.4f}"

    df_out = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    results_path = os.path.join(output_dir, "exp2_walkforward.csv")
    os.makedirs(output_dir, exist_ok=True)
    df_out.to_csv(results_path, index=False)
    print(f"\n[Exp2] Results saved -> {results_path}")
    print(f"[Exp2] GG F1  : {mean_row['GG_F1']:.4f} +/- {sd_row['GG_F1']:.4f}")
    print(f"[Exp2] Base F1: {mean_row['Base_F1']:.4f} +/- {sd_row['Base_F1']:.4f}")
    print(f"[Exp2] Cohen's d: {cohens_d:.4f}")

    return float(mean_row["GG_tau"])
