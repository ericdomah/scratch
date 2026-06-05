"""
SGCC Dataset Preprocessing Pipeline
=====================================
Loads the SGCC CSV (one row per consumer, daily kWh 2014-01-01 to 2016-10-31)
and returns a (N, 26, 2) float32 array ready for GridGuardUniversalHybrid.

ONE WINDOW PER CONSUMER — NON-NEGOTIABLE.
SGCC labels are consumer-level (FLAG 0/1). No theft-onset timestamps exist.
Sliding windows would label pre-theft normal consumption as theft — forbidden.
"""
from __future__ import annotations

import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew

warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW_SIZE = 26      # weeks — must stay 26 to match thesis architecture
MAX_SAMPLES = 50_000  # safety guard — raise ValueError if exceeded


def load_sgcc(sgcc_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess the SGCC CSV.

    Parameters
    ----------
    sgcc_path : path to the SGCC CSV file

    Returns
    -------
    X : float32 array (N, 26, 2) — ch0=kWh normalised, ch1=GLI
    y : float32 array (N,)       — 0=normal, 1=theft
    """
    # Step 1 — Load CSV, identify columns
    print(f"[SGCC] Loading: {sgcc_path}")
    df = pd.read_csv(sgcc_path, low_memory=False)
    print(f"[SGCC] Raw shape: {df.shape}")

    if "CONS_NO" in df.columns and "FLAG" in df.columns:
        id_col    = "CONS_NO"
        label_col = "FLAG"
        date_cols = [c for c in df.columns if c not in (id_col, label_col)]
    else:
        date_cols = []
        for col in df.columns:
            try:
                pd.to_datetime(col)
                date_cols.append(col)
            except (ValueError, TypeError):
                pass
        meta_cols = [c for c in df.columns if c not in date_cols]
        if len(meta_cols) >= 2:
            id_col, label_col = meta_cols[0], meta_cols[1]
        else:
            raise ValueError("Cannot identify CONS_NO / FLAG columns in SGCC CSV.")

    if not date_cols:
        raise ValueError("No date columns found in SGCC CSV.")

    try:
        date_cols = sorted(date_cols, key=pd.to_datetime)
    except Exception:
        pass

    cons_ids = df[id_col].values.astype(str)
    labels   = df[label_col].values.astype(int)
    raw_data = df[date_cols].values.astype(float)

    print(f"[SGCC] Consumers : {len(cons_ids):,}")
    print(f"[SGCC] Date cols  : {len(date_cols)}  ({date_cols[0]} -> {date_cols[-1]})")
    print(f"[SGCC] Theft rate : {labels.mean():.3%}  ({labels.sum():,} / {len(labels):,})")

    # Step 2 — Filter consumers with >50% NaN
    nan_frac = np.isnan(raw_data).mean(axis=1)
    valid    = nan_frac <= 0.50
    print(f"[SGCC] Dropping {(~valid).sum():,} consumers (>50% NaN).  Remaining: {valid.sum():,}")
    raw_data = raw_data[valid]
    labels   = labels[valid]
    cons_ids = cons_ids[valid]

    # Step 3 — Impute NaN: linear interpolation, edge fill 0
    print("[SGCC] Interpolating missing values...")
    consumption = np.empty_like(raw_data)
    for i in range(len(raw_data)):
        s = pd.Series(raw_data[i]).interpolate(method="linear", limit_direction="both")
        consumption[i] = s.fillna(0.0).values

    # Step 4 — Aggregate daily -> weekly (7-day sums), vectorised
    n_consumers, n_days = consumption.shape
    n_full_weeks = n_days // 7
    remainder    = n_days % 7

    daily_trim  = consumption[:, :n_full_weeks * 7]
    weekly_full = daily_trim.reshape(n_consumers, n_full_weeks, 7).sum(axis=2)

    if remainder > 0:
        partial    = consumption[:, n_full_weeks * 7:].sum(axis=1, keepdims=True)
        weekly_raw = np.concatenate([weekly_full, partial], axis=1)
    else:
        weekly_raw = weekly_full

    print(f"[SGCC] Weekly aggregation: {n_consumers:,} consumers x {weekly_raw.shape[1]} weeks")

    # Step 5 — Extract LAST 26 weeks — ONE window per consumer (NON-NEGOTIABLE)
    if weekly_raw.shape[1] < WINDOW_SIZE:
        raise ValueError(
            f"Dataset has only {weekly_raw.shape[1]} weeks — need at least {WINDOW_SIZE}. "
            "Check the full SGCC CSV was supplied."
        )

    print(f"[SGCC] Extracting LAST {WINDOW_SIZE} weeks per consumer (one window — no sliding)...")
    windows  = weekly_raw[:, -WINDOW_SIZE:]   # (N, 26)
    n_samples = len(windows)

    if n_samples > MAX_SAMPLES:
        raise ValueError(
            f"ERROR: {n_samples} samples detected. "
            f"Expected ~31,000 (one per surviving consumer). "
            f"Sliding window was applied — fix Step 5 immediately."
        )
    print(f"[SGCC] Sample count: {n_samples:,} (expected ~31,000) — OK")

    # Step 6 — Per-consumer min-max normalisation to [0, 1]
    w_min  = windows.min(axis=1, keepdims=True)
    w_max  = windows.max(axis=1, keepdims=True)
    w_rng  = np.where(w_max - w_min > 0, w_max - w_min, 1.0)
    kWh_norm = (windows - w_min) / w_rng   # (N, 26)

    # Step 7 — Compute Grid Load Index (GLI)
    aggregate_load = kWh_norm.sum(axis=0)                          # (26,)
    gli_denom      = aggregate_load.max()
    gli            = aggregate_load / (gli_denom if gli_denom > 0 else 1.0)
    gli_matrix     = np.tile(gli, (n_samples, 1))                  # (N, 26)

    # Step 8 — Stack into (N, 26, 2)
    X = np.stack([kWh_norm, gli_matrix], axis=2)   # (N, 26, 2)
    y = labels.astype(np.float32)

    # Step 9 — Verification summary
    n_normal   = int((y == 0).sum())
    n_theft    = int((y == 1).sum())
    prevalence = n_theft / len(y) * 100.0

    print(f"\n{'='*42}")
    print(f"  SGCC DATASET SUMMARY")
    print(f"{'='*42}")
    print(f"  Total samples      : {len(y):,}")
    print(f"  Normal (FLAG=0)    : {n_normal:,}")
    print(f"  Theft  (FLAG=1)    : {n_theft:,}")
    print(f"  Theft prevalence   : {prevalence:.2f}%")
    print(f"  Tensor shape       : {X.shape}")
    print(f"{'='*42}\n")

    if not (1.0 <= prevalence <= 25.0):
        raise ValueError(
            f"Theft prevalence = {prevalence:.2f}% is outside expected 1-25%. "
            "Data has likely been processed incorrectly. Investigate before continuing."
        )

    return X.astype(np.float32), y.astype(np.float32)


def compute_tabular_features(X_np: np.ndarray) -> np.ndarray:
    """
    Derive 5 tabular features from (N, 26, 2) for the XGBoost tier.

    Features: variance_kWh, skewness_kWh, peak_to_avg_ratio, mean_GLI, std_GLI
    Returns (N, 5) float32.
    """
    kwh = X_np[:, :, 0]
    gli = X_np[:, :, 1]

    var_kwh  = kwh.var(axis=1)
    skw_kwh  = np.apply_along_axis(scipy_skew, 1, kwh)
    mean_kwh = np.where(kwh.mean(axis=1) > 0, kwh.mean(axis=1), 1e-8)
    par_kwh  = kwh.max(axis=1) / mean_kwh
    mean_gli = gli.mean(axis=1)
    std_gli  = gli.std(axis=1)

    return np.column_stack([var_kwh, skw_kwh, par_kwh, mean_gli, std_gli]).astype(np.float32)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../../data/sgcc/data.csv"
    X, y = load_sgcc(path)
    print(f"[OK]  X={X.shape}  y={y.shape}  theft={y.mean():.3%}")
    feats = compute_tabular_features(X)
    print(f"[OK]  Tabular features: {feats.shape}")
