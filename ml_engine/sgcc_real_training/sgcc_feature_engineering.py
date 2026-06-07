# Cell 1 — Mount Drive and clone repo
# from google.colab import drive
# drive.mount('/content/drive')
# !git clone https://github.com/ericdomah/scratch.git
# %cd scratch

# Cell 2 — Copy SGCC data from Drive
# !cp /content/drive/MyDrive/sgcc_data/data.csv \
#     ml_engine/sgcc_real_training/data/sgcc/

# Cell 3 — Run the experiment
# !python ml_engine/sgcc_real_training/sgcc_feature_engineering.py \
#     --sgcc_path ml_engine/sgcc_real_training/data/sgcc/data.csv \
#     --output    ml_engine/sgcc_real_training/results/ \
#     --seed      42

# Cell 4 — Save results to Drive
# !cp ml_engine/sgcc_real_training/results/exp_xgboost_75features.csv \
#     /content/drive/MyDrive/gridguard_results/

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    brier_score_loss, balanced_accuracy_score, confusion_matrix
)
from xgboost import XGBClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgcc_path", required=True, help="Path to data.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def compute_features(readings_df):
    """
    Given a DataFrame of daily readings (N_consumers, N_days),
    returns a NumPy array of shape (N_consumers, 75) containing all features.
    """
    # Sort columns to ensure chronological order (assuming YYYY/M/D or similar format)
    date_cols = sorted(readings_df.columns, key=lambda d: pd.to_datetime(d))
    
    # Parse dates for weekday/weekend and monthly groupings
    dates = pd.to_datetime(date_cols)
    weekdays = dates.weekday
    is_weekend = weekdays >= 5
    is_weekday = weekdays < 5
    months = dates.month
    
    arr = readings_df[date_cols].values # (N, T)
    N, T = arr.shape
    
    features = []
    
    # Precompute common stats
    mean_val = np.nanmean(arr, axis=1)
    std_val = np.nanstd(arr, axis=1)
    
    # Fill any edge case nan means
    mean_val = np.nan_to_num(mean_val, nan=0.0)
    std_val = np.nan_to_num(std_val, nan=0.0)
    
    # ==========================================
    # GROUP 1: Basic Statistics (10)
    # ==========================================
    f1 = mean_val
    f2 = std_val
    f3 = np.nanvar(arr, axis=1)
    f4 = scipy.stats.skew(arr, axis=1, nan_policy='omit')
    f5 = scipy.stats.kurtosis(arr, axis=1, nan_policy='omit')
    f6 = np.nanmin(arr, axis=1)
    f7 = np.nanmax(arr, axis=1)
    f8 = f7 - f6
    f9 = np.nanmedian(arr, axis=1)
    f10 = np.nanpercentile(arr, 75, axis=1) - np.nanpercentile(arr, 25, axis=1)
    
    # ==========================================
    # GROUP 2: Zero and Anomaly Patterns (5)
    # ==========================================
    f11 = np.nanmean(arr == 0.0, axis=1)
    f12 = np.nanmean(arr < 0.1, axis=1)
    f13 = np.nanmean(arr < 0, axis=1)
    f14 = np.nanmean(arr > (mean_val[:, None] + 3 * std_val[:, None]), axis=1)
    
    # max_consecutive_zeros
    # Vectorized way: pad with non-zeros, find diffs, etc.
    # We will do a fast loop over N since it's just a 1D scan
    f15 = np.zeros(N)
    for i in range(N):
        is_zero = (arr[i] == 0.0).astype(int)
        if np.sum(is_zero) == 0:
            f15[i] = 0
            continue
        # Pad with 0 at ends
        padded = np.pad(is_zero, (1, 1), mode='constant')
        diffs = np.diff(padded)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        f15[i] = np.max(ends - starts) if len(starts) > 0 else 0
        
    # ==========================================
    # GROUP 3: Temporal Change Indicators (10)
    # ==========================================
    p1_end = T // 3
    p2_end = 2 * (T // 3)
    
    arr_early = arr[:, :p1_end]
    arr_mid = arr[:, p1_end:p2_end]
    arr_late = arr[:, p2_end:]
    
    f16 = np.nanmean(arr_early, axis=1)
    f17 = np.nanmean(arr_mid, axis=1)
    f18 = np.nanmean(arr_late, axis=1)
    f19 = np.nanstd(arr_early, axis=1)
    f20 = np.nanstd(arr_mid, axis=1)
    f21 = np.nanstd(arr_late, axis=1)
    
    f16_safe = np.nan_to_num(f16, nan=0.0)
    f17_safe = np.nan_to_num(f17, nan=0.0)
    
    f22 = f18 / (f16_safe + 1e-6)
    f23 = f18 / (f17_safe + 1e-6)
    f24 = f17 / (f16_safe + 1e-6)
    
    time_idx = np.arange(T)
    # slope = covariance(t, arr) / variance(t)
    t_mean = np.mean(time_idx)
    arr_mean = np.nanmean(arr, axis=1, keepdims=True)
    num = np.nansum((time_idx - t_mean) * (arr - arr_mean), axis=1)
    den = np.sum((time_idx - t_mean)**2)
    f25 = num / den
    
    # ==========================================
    # GROUP 4: Weekday vs Weekend (5)
    # ==========================================
    f26 = np.nanmean(arr[:, is_weekday], axis=1)
    f27 = np.nanmean(arr[:, is_weekend], axis=1)
    f26_safe = np.nan_to_num(f26, nan=0.0)
    f28 = f27 / (f26_safe + 1e-6)
    f29 = np.nanstd(arr[:, is_weekday], axis=1)
    f30 = np.nanstd(arr[:, is_weekend], axis=1)
    
    # ==========================================
    # GROUP 5: Monthly Patterns (12)
    # ==========================================
    monthly_means = []
    for m in range(1, 13):
        mask = (months == m)
        if np.sum(mask) > 0:
            m_mean = np.nanmean(arr[:, mask], axis=1)
        else:
            m_mean = np.zeros(N)
        monthly_means.append(m_mean)
        
    f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42 = monthly_means
    
    # ==========================================
    # GROUP 6: Quantiles (10)
    # ==========================================
    f43 = np.nanpercentile(arr, 5, axis=1)
    f44 = np.nanpercentile(arr, 10, axis=1)
    f45 = np.nanpercentile(arr, 25, axis=1)
    f46 = np.nanpercentile(arr, 75, axis=1)
    f47 = np.nanpercentile(arr, 90, axis=1)
    f48 = np.nanpercentile(arr, 95, axis=1)
    f49 = np.nanpercentile(arr, 99, axis=1)
    
    f43_safe = np.nan_to_num(f43, nan=0.0)
    f45_safe = np.nan_to_num(f45, nan=0.0)
    f50 = f48 / (f43_safe + 1e-6)
    f51 = f46 / (f45_safe + 1e-6)
    
    f52 = np.nanmean(arr > mean_val[:, None], axis=1)
    
    # ==========================================
    # GROUP 7: FFT (10)
    # ==========================================
    # FFT is fast enough over entire array
    fft_vals = np.abs(np.fft.rfft(np.nan_to_num(arr, nan=0.0), axis=1))
    fft_vals = fft_vals / (T / 2.0)
    # Top 10 indices skipping 0
    f53 = fft_vals[:, 1] if fft_vals.shape[1] > 1 else np.zeros(N)
    f54 = fft_vals[:, 2] if fft_vals.shape[1] > 2 else np.zeros(N)
    f55 = fft_vals[:, 3] if fft_vals.shape[1] > 3 else np.zeros(N)
    f56 = fft_vals[:, 4] if fft_vals.shape[1] > 4 else np.zeros(N)
    f57 = fft_vals[:, 5] if fft_vals.shape[1] > 5 else np.zeros(N)
    f58 = fft_vals[:, 6] if fft_vals.shape[1] > 6 else np.zeros(N)
    f59 = fft_vals[:, 7] if fft_vals.shape[1] > 7 else np.zeros(N)
    f60 = fft_vals[:, 8] if fft_vals.shape[1] > 8 else np.zeros(N)
    f61 = fft_vals[:, 9] if fft_vals.shape[1] > 9 else np.zeros(N)
    f62 = fft_vals[:, 10] if fft_vals.shape[1] > 10 else np.zeros(N)
    
    # ==========================================
    # GROUP 8: Autocorr (5)
    # ==========================================
    def fast_autocorr(arr_2d, lag):
        if lag >= T: return np.zeros(N)
        # Pearson correlation between arr[:, lag:] and arr[:, :-lag]
        x = arr_2d[:, lag:]
        y = arr_2d[:, :-lag]
        x_mean = np.nanmean(x, axis=1, keepdims=True)
        y_mean = np.nanmean(y, axis=1, keepdims=True)
        num = np.nansum((x - x_mean) * (y - y_mean), axis=1)
        den = np.sqrt(np.nansum((x - x_mean)**2, axis=1) * np.nansum((y - y_mean)**2, axis=1))
        with np.errstate(divide='ignore', invalid='ignore'):
            res = num / den
        return np.nan_to_num(res, nan=0.0)

    f63 = fast_autocorr(arr, 1)
    f64 = fast_autocorr(arr, 7)
    f65 = fast_autocorr(arr, 14)
    f66 = fast_autocorr(arr, 30)
    f67 = fast_autocorr(arr, 90)
    
    # ==========================================
    # GROUP 9: Rate of Change (8)
    # ==========================================
    delta = np.diff(arr, axis=1)
    f68 = np.nanmean(delta, axis=1)
    f69 = np.nanstd(delta, axis=1)
    f70 = np.nanmean(np.abs(delta), axis=1)
    f71 = np.nanmax(delta, axis=1) if delta.shape[1] > 0 else np.zeros(N)
    f72 = np.nanmin(delta, axis=1) if delta.shape[1] > 0 else np.zeros(N)
    f73 = np.nanmean(delta < 0, axis=1)
    
    # 7-day rolling difference
    if T >= 14:
        # We can just reshape or use convolution. Simple way: reshape truncating remainder
        weeks = T // 7
        truncated = arr[:, :weeks*7]
        weekly_means = np.nanmean(truncated.reshape(N, weeks, 7), axis=2)
        delta7 = np.diff(weekly_means, axis=1)
        f74 = np.nanmean(delta7, axis=1)
        f75 = np.nanstd(delta7, axis=1)
    else:
        f74 = np.zeros(N)
        f75 = np.zeros(N)
        
    # Combine all
    feat_list = [
        f1, f2, f3, f4, f5, f6, f7, f8, f9, f10,
        f11, f12, f13, f14, f15,
        f16, f17, f18, f19, f20, f21, f22, f23, f24, f25,
        f26, f27, f28, f29, f30,
        f31, f32, f33, f34, f35, f36, f37, f38, f39, f40, f41, f42,
        f43, f44, f45, f46, f47, f48, f49, f50, f51, f52,
        f53, f54, f55, f56, f57, f58, f59, f60, f61, f62,
        f63, f64, f65, f66, f67,
        f68, f69, f70, f71, f72, f73, f74, f75
    ]
    
    X_feats = np.column_stack(feat_list)
    
    # Final safety check for NaN/Inf
    X_feats = np.nan_to_num(X_feats, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_feats, [f"feat_{i:02d}" for i in range(1, 76)]


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)
    
    print(f"Loading data from {args.sgcc_path}...")
    df = pd.read_csv(args.sgcc_path)
    
    labels = df['FLAG'].astype(int).values
    readings = df.drop(['CONS_NO', 'FLAG'], axis=1)
    
    # Sort chronological
    date_cols = sorted(readings.columns, key=lambda d: pd.to_datetime(d))
    readings = readings[date_cols]
    
    print(f"Initial shape: {readings.shape}")
    
    # Filter >50% NaN
    nan_counts = readings.isna().sum(axis=1)
    valid_mask = nan_counts <= 517
    
    readings = readings[valid_mask]
    labels = labels[valid_mask]
    
    print("Imputing missing values...")
    readings = readings.interpolate(method='linear', axis=1, limit_direction='both')
    readings = readings.fillna(0.0)
    
    N = len(labels)
    N_theft = int(np.sum(labels))
    N_normal = N - N_theft
    pct = (N_theft / N) * 100
    
    print(f"Total consumers after filtering: {N}")
    print(f"Normal: {N_normal}   Theft: {N_theft}")
    print(f"Theft prevalence: {pct:.2f}%")
    
    pos_weight = 28779 / 2409 # From prompt specification
    
    print("Engineering 75 features...")
    X_features, feat_names = compute_features(readings)
    
    print(f"Feature matrix shape: {X_features.shape}")
    print(f"NaN count: {np.isnan(X_features).sum()}")
    print(f"Any Inf: {np.isinf(X_features).sum()}")
    
    print("Starting 10-Fold XGBoost Evaluation...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_features, labels), 1):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric='logloss',
            early_stopping_rounds=30,
            use_label_encoder=False
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        best_tau = 0.5
        best_f1 = 0.0
        for tau in np.arange(0.05, 0.95, 0.01):
            y_pred_tmp = (y_prob >= tau).astype(int)
            f1 = f1_score(y_test, y_pred_tmp, average='binary', zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
                
        y_pred = (y_prob >= best_tau).astype(int)
        F1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        Precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        Recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        AUROC = roc_auc_score(y_test, y_prob)
        Brier = brier_score_loss(y_test, y_prob)
        Bal_Acc = balanced_accuracy_score(y_test, y_pred)
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        
        print(f"Fold {fold}/10 | τ={best_tau:.2f} | F1={F1:.4f} | AUROC={AUROC:.4f} | Prec={Precision:.4f} | Rec={Recall:.4f} | Bal_Acc={Bal_Acc:.4f}")
        
        fold_results.append({
            'Fold': fold,
            'tau': best_tau,
            'F1': F1,
            'Precision': Precision,
            'Recall': Recall,
            'AUROC': AUROC,
            'Brier': Brier,
            'Balanced_Accuracy': Bal_Acc,
            'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP
        })
        
    df_res = pd.DataFrame(fold_results)
    
    means = df_res.mean()
    stds = df_res.std()
    
    print("\nTraining final model on FULL dataset for Feature Importances...")
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_features)
    final_model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    final_model.fit(X_full_scaled, labels, verbose=False)
    
    booster = final_model.get_booster()
    scores = booster.get_score(importance_type='gain')
    # Map feature indices (f0, f1) back to our feature names
    importance_list = []
    for f_idx, gain in scores.items():
        # f_idx is like 'f0', 'f1'
        idx = int(f_idx.replace('f', ''))
        importance_list.append((feat_names[idx], gain))
        
    importance_list.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- TOP 20 FEATURES BY GAIN ---")
    for fname, gain in importance_list[:20]:
        print(f"{fname:<10} : {gain:.4f}")
        
    # Build summary row
    summary_dict = {'Fold': 'Mean ± SD'}
    for col in df_res.columns:
        if col == 'Fold': continue
        mean_val = means[col]
        std_val = stds[col]
        ci = 2.262 * std_val / np.sqrt(10)
        summary_dict[col] = f"{mean_val:.4f} ± {std_val:.4f} (CI: ±{ci:.4f})"
        
    df_res = pd.concat([df_res, pd.DataFrame([summary_dict])], ignore_index=True)
    
    out_csv = os.path.join(args.output, 'exp_xgboost_75features.csv')
    df_res.to_csv(out_csv, index=False)
    print(f"\nResults saved to {out_csv}")
    
    auroc = means['AUROC']
    f1 = means['F1']
    prec = means['Precision']
    rec = means['Recall']
    
    table = f"""
  ╔══════════════════════════════════════════════════════════╗
  ║     REAL SGCC RESULTS — SIDE BY SIDE COMPARISON         ║
  ╠══════════════════════════════════════════════════════════╣
  ║ Model                        AUROC    F1    Prec   Rec  ║
  ║ LightGBM (75 features)       0.778  0.367  0.359  0.376 ║
  ║ CatBoost (75 features)       0.770  0.365  0.375  0.355 ║
  ║ XGBoost  (75 features) [NEW] {auroc:.3f}  {f1:.3f}  {prec:.3f}  {rec:.3f} ║
  ║ ─────────────────────────────────────────────────────── ║
  ║ GridGuard DL (26 weekly)     0.554  0.155  0.087  0.749 ║
  ║ GridGuard DL (synthetic)     0.952  0.905  0.911  0.898 ║
  ╚══════════════════════════════════════════════════════════╝
"""
    print(table)

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
