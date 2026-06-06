"""
GridGuard AI — Train on datasetsmall.csv
=========================================
One-window-per-consumer (26 daily readings → reshaped as 26 timesteps × 2 features).
Since this dataset only has 26 columns, GLI is synthesized as a z-score of daily usage.
Uses:
  - 10-fold stratified CV
  - Per-fold threshold calibration (sweep 0.05–0.95)
  - Class-weighted focal loss (pos_weight = n_normal / n_theft)
  - AsymmetricFocalLoss(alpha=0.92, gamma_pos=2.0, gamma_neg=2.0)
  - Early stopping on val AUROC (patience=8)
  - epochs=30, max_lr=5e-4, batch_size=128, AdamW(lr=1e-4)
  - Fusion: 0.70 × P_DL + 0.30 × P_XGB
"""

import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (f1_score, roc_auc_score, precision_score,
                             recall_score, brier_score_loss, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import pickle

warnings.filterwarnings("ignore")

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 128
EPOCHS       = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
MAX_LR       = 5e-4
GRAD_CLIP    = 1.0
N_FOLDS      = 10
PATIENCE     = 8
FIXED_TAU    = 0.5270

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32       = False

set_seed(SEED)

# ── Model ──────────────────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.res  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
    def forward(self, x):
        out = self.conv(x)[:, :, :-self.conv.padding[0]]
        out = F.relu(self.drop(out))
        return F.relu(out + (self.res(x) if self.res else x))

class GridGuardUniversalHybrid(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_heads=4,
                 num_lstm_layers=2, dropout=0.2):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(input_dim,  hidden_dim, dilation=1, dropout=dropout),
            TCNBlock(hidden_dim, hidden_dim, dilation=2, dropout=dropout),
        )
        self.tcn_pool = nn.AdaptiveAvgPool1d(1)
        self.lstm = nn.LSTM(input_dim, hidden_dim//2, num_layers=num_lstm_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if num_lstm_layers > 1 else 0.0)
        enc_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads,
                                               dim_feedforward=hidden_dim*4,
                                               dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        tcn_vec  = self.tcn_pool(self.tcn(x.transpose(1,2))).squeeze(-1)
        lstm_out, _ = self.lstm(x)
        trans_vec   = self.transformer(lstm_out)[:, -1, :]
        return self.fc(torch.cat([tcn_vec, trans_vec], dim=1))

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.92, gamma_pos=2.0, gamma_neg=2.0):
        super().__init__()
        self.alpha=alpha; self.gamma_pos=gamma_pos; self.gamma_neg=gamma_neg
    def forward(self, preds, targets):
        preds   = preds.squeeze()
        bce     = F.binary_cross_entropy(preds, targets, reduction='none')
        p_t     = torch.where(targets==1, preds, 1-preds)
        gamma   = torch.where(targets==1,
                    torch.tensor(self.gamma_pos, device=preds.device),
                    torch.tensor(self.gamma_neg, device=preds.device))
        focal_w = (1-p_t)**gamma
        alpha_t = torch.where(targets==1,
                    torch.tensor(self.alpha,   device=preds.device),
                    torch.tensor(1-self.alpha, device=preds.device))
        return (alpha_t * focal_w * bce)   # return per-sample (no .mean())

# ── Preprocessing ──────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    date_cols = [c for c in df.columns if c not in ['CONS_NO','FLAG']]
    X_raw = df[date_cols].values.astype(np.float32)
    y     = df['FLAG'].values.astype(np.float32)

    # Interpolate NaNs
    X_raw = pd.DataFrame(X_raw).interpolate(axis=1, limit_direction='both').values

    # Synthesise GLI channel: z-score of each row's daily usage
    row_mean = X_raw.mean(axis=1, keepdims=True)
    row_std  = X_raw.std(axis=1, keepdims=True) + 1e-8
    gli      = (X_raw - row_mean) / row_std

    # Stack: (N, 26, 2)
    X = np.stack([X_raw, gli], axis=2)

    # Normalise kWh channel across training set (done globally here for simplicity)
    scaler = StandardScaler()
    N, T, _ = X.shape
    X[:,:,0] = scaler.fit_transform(X[:,:,0])

    print(f"\n{'='*55}")
    print(f"  datasetsmall Preprocessing Summary")
    print(f"{'='*55}")
    print(f"  Consumers       : {N:,}")
    print(f"  Normal (FLAG=0) : {(y==0).sum():,}")
    print(f"  Theft  (FLAG=1) : {(y==1).sum():,}")
    print(f"  Theft rate      : {y.mean()*100:.2f}%")
    print(f"  Tensor shape    : {X.shape}")
    print(f"{'='*55}\n")

    return torch.FloatTensor(X), torch.FloatTensor(y)

# ── Tabular features for XGB ───────────────────────────────────────────────────
def tabular_features(X_np):
    kwh = X_np[:,:,0]
    return np.column_stack([
        kwh.mean(1), kwh.std(1), kwh.max(1), kwh.min(1),
        np.percentile(kwh,75,axis=1)-np.percentile(kwh,25,axis=1),
    ])

# ── Training helpers ───────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, sched, criterion, device, pos_weight):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        opt.zero_grad()
        per_sample_loss = criterion(model(Xb), yb)
        theft_mask  = (yb == 1).float()
        normal_mask = (yb == 0).float()
        loss = (per_sample_loss * theft_mask * pos_weight +
                per_sample_loss * normal_mask * 1.0).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); sched.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def get_probs(model, loader, device):
    model.eval()
    all_p = []
    for Xb, _ in loader:
        all_p.append(model(Xb.to(device)).squeeze().cpu().numpy())
    return np.concatenate(all_p)

def calibrate_threshold(fused_probs, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(labels, (fused_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return round(best_t, 2)

def evaluate(fused_probs, labels, threshold):
    preds = (fused_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    return {
        "F1":        round(f1_score(labels, preds,       zero_division=0), 4),
        "AUROC":     round(roc_auc_score(labels, fused_probs),              4),
        "Precision": round(precision_score(labels, preds, zero_division=0), 4),
        "Recall":    round(recall_score(labels, preds,   zero_division=0),  4),
        "Brier":     round(brier_score_loss(labels, fused_probs),           4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }

# ── Main 10-fold CV ────────────────────────────────────────────────────────────
def run_cv(X, y, output_dir):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    X_np     = X.numpy(); y_np = y.numpy().astype(int)
    feats_np = tabular_features(X_np)

    print(f"\n{'='*55}")
    print(f"  EXPERIMENT — 10-Fold CV   device={device}")
    print(f"{'='*55}")

    skf       = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_rows = []
    best_f1   = -1.0
    best_weights_path = None
    calibrated_thresholds = []

    os.makedirs(f"{output_dir}/models",  exist_ok=True)
    os.makedirs(f"{output_dir}/results", exist_ok=True)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_np, y_np), 1):
        print(f"\n-- Fold {fold}/{N_FOLDS} ------------------------------")

        # Inner val split for early stopping & threshold calibration
        tr_inner, va_idx = train_test_split(
            tr_idx, test_size=0.1, stratify=y_np[tr_idx], random_state=SEED+fold
        )

        X_tr, y_tr = X[tr_inner], y[tr_inner]
        X_va, y_va = X[va_idx],   y[va_idx]
        X_te, y_te = X[te_idx],   y[te_idx]

        # Class weights
        n_theft  = (y_np[tr_inner] == 1).sum()
        n_normal = (y_np[tr_inner] == 0).sum()
        pos_weight = n_normal / max(n_theft, 1)
        print(f"   pos_weight = {pos_weight:.2f}  "
              f"(n_normal={n_normal:,}, n_theft={n_theft:,})")

        tr_ld = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                           shuffle=True,  pin_memory=(device=="cuda"))
        va_ld = DataLoader(TensorDataset(X_va, y_va), batch_size=BATCH_SIZE,
                           shuffle=False)
        te_ld = DataLoader(TensorDataset(X_te, y_te), batch_size=BATCH_SIZE,
                           shuffle=False)

        set_seed(SEED + fold)
        model     = GridGuardUniversalHybrid().to(device)
        criterion = AsymmetricFocalLoss(alpha=0.92, gamma_pos=2.0, gamma_neg=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR, total_steps=EPOCHS*len(tr_ld))

        # Early stopping
        best_auroc    = -1.0
        patience_ctr  = 0
        best_state    = None
        best_epoch    = 1

        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, tr_ld, optimizer, scheduler,
                               criterion, device, pos_weight)

            # Validate
            va_probs = get_probs(model, va_ld, device)
            # XGB on val for fusion
            clf_va = GradientBoostingClassifier(**{
                'n_estimators':200,'max_depth':6,'learning_rate':0.05,
                'subsample':0.8,'random_state':SEED}).fit(
                    feats_np[tr_inner], y_np[tr_inner])
            va_xgb = clf_va.predict_proba(feats_np[va_idx])[:,1]
            va_fused = 0.70 * va_probs + 0.30 * va_xgb
            try:
                val_auroc = roc_auc_score(y_np[va_idx], va_fused)
            except Exception:
                val_auroc = 0.0

            if epoch % 5 == 0 or epoch == 1:
                print(f"   Epoch {epoch:2d}/{EPOCHS}  loss={loss:.4f}  val_AUROC={val_auroc:.4f}")

            if val_auroc > best_auroc:
                best_auroc   = val_auroc
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch   = epoch
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"   Early stopping at epoch {epoch}, "
                          f"best AUROC epoch was {best_epoch} (AUROC={best_auroc:.4f})")
                    break

        # Load best checkpoint
        model.load_state_dict(best_state)

        # Full XGB on training split
        clf = GradientBoostingClassifier(**{
            'n_estimators':200,'max_depth':6,'learning_rate':0.05,
            'subsample':0.8,'random_state':SEED}).fit(
                feats_np[tr_inner], y_np[tr_inner])

        # Threshold calibration on validation fusion
        va_probs_best = get_probs(model, va_ld, device)
        va_xgb_best   = clf.predict_proba(feats_np[va_idx])[:,1]
        va_fused_best = 0.70 * va_probs_best + 0.30 * va_xgb_best
        tau_cal = calibrate_threshold(va_fused_best, y_np[va_idx])
        calibrated_thresholds.append(tau_cal)
        print(f"   Calibrated threshold (val) = {tau_cal:.2f}")

        # Test evaluation
        te_probs  = get_probs(model, te_ld, device)
        te_xgb    = clf.predict_proba(feats_np[te_idx])[:,1]
        te_fused  = 0.70 * te_probs + 0.30 * te_xgb

        metrics_cal   = evaluate(te_fused, y_np[te_idx], tau_cal)
        metrics_fixed = evaluate(te_fused, y_np[te_idx], FIXED_TAU)

        print(f"   [Calibrated τ={tau_cal:.2f}] "
              f"F1={metrics_cal['F1']:.4f}  AUROC={metrics_cal['AUROC']:.4f}  "
              f"Prec={metrics_cal['Precision']:.4f}  Rec={metrics_cal['Recall']:.4f}")
        print(f"   [Fixed    τ={FIXED_TAU}] "
              f"F1={metrics_fixed['F1']:.4f}")

        row = {"Fold": fold, "threshold": tau_cal, **metrics_cal,
               "F1_fixed_tau": metrics_fixed["F1"]}
        fold_rows.append(row)

        # Save best model globally
        if metrics_cal["F1"] > best_f1:
            best_f1 = metrics_cal["F1"]
            best_weights_path = f"{output_dir}/models/smalldata_best.pth"
            torch.save(model.state_dict(), best_weights_path)
            with open(f"{output_dir}/models/smalldata_xgb_best.pkl","wb") as fh:
                pickle.dump(clf, fh)
            print(f"   * New best fold (F1={best_f1:.4f}) — weights saved")

    # ── Summary ────────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(fold_rows)
    key_cols   = ["F1","AUROC","Precision","Recall","Brier"]
    t_val      = 2.262  # 10-fold t-value for 95% CI

    summary = {"Fold": "mean +/- SD", "threshold": round(np.mean(calibrated_thresholds),3)}
    for col in key_cols:
        vals = results_df[col].values.astype(float)
        mu   = vals.mean()
        sd   = vals.std(ddof=1)
        ci   = t_val * sd / np.sqrt(N_FOLDS)
        summary[col] = f"{mu:.4f} +/- {sd:.4f}  [95%CI +/-{ci:.4f}]"
    summary["F1_fixed_tau"] = f"{results_df['F1_fixed_tau'].mean():.4f}"

    final_df = pd.concat([results_df, pd.DataFrame([summary])], ignore_index=True)
    csv_path = f"{output_dir}/results/smalldata_cv_results.csv"
    final_df.to_csv(csv_path, index=False)

    print(f"\n{'='*55}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*55}")
    print(final_df[["Fold","threshold","F1","AUROC","Precision","Recall","Brier","F1_fixed_tau"]].to_string(index=False))
    print(f"\n  Mean calibrated threshold : {np.mean(calibrated_thresholds):.3f}")
    print(f"  Best fold F1              : {best_f1:.4f}")
    print(f"  Results saved to          : {csv_path}")
    print(f"{'='*55}")

    return final_df

# ── Entry ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    CSV_PATH   = "data/datasetsmall.csv"
    OUTPUT_DIR = "smalldata_output"
    X, y = load_and_preprocess(CSV_PATH)
    run_cv(X, y, OUTPUT_DIR)
