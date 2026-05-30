"""
SGCC Dataset Preprocessing Pipeline
=====================================
Converts the raw SGCC CSV (one row per consumer, daily kWh readings 2014-01-01
to 2016-10-31) into a (N, 26, 2) tensor ready for GridGuardUniversalHybrid.

Pipeline steps
--------------
1.  Load CSV — detect CONS_NO / FLAG / date columns automatically
2.  Drop consumers with >50 % NaN readings; linear-interpolate the rest
3.  Per-consumer min-max normalisation → [0, 1]
4.  Daily → weekly aggregation (7-day sums)  →  (N_consumers, ~147 weeks)
5.  Compute Grid Load Index (GLI) across the whole substation
6.  Sliding 26-week windows  →  (N_samples, 26) each for kWh and GLI
7.  Stack into (N_samples, 26, 2)  and split train/test (temporal order)
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import skew as scipy_skew
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW_SIZE = 26   # weeks — must stay 26 to match thesis architecture


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Loader
# ─────────────────────────────────────────────────────────────────────────────

def load_sgcc(data_dir: str) -> pd.DataFrame:
    """
    Locate and load the SGCC CSV from *data_dir*.

    Expected filename patterns (case-insensitive):
      data.csv | sgcc*.csv | *.csv  (falls back to first CSV found)
    """
    data_dir = os.path.abspath(data_dir)
    csv_files = sorted(
        f for f in os.listdir(data_dir) if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Place the SGCC file (data.csv or similar) there and re-run."
        )

    # Prefer files whose names hint at 'data' or 'sgcc'
    preferred = [f for f in csv_files
                 if any(k in f.lower() for k in ("data", "sgcc"))]
    target = preferred[0] if preferred else csv_files[0]
    csv_path = os.path.join(data_dir, target)

    print(f"[SGCC] Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[SGCC] Raw shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  2-7.  Full Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_sgcc(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Run the complete SGCC preprocessing pipeline.

    Parameters
    ----------
    df          : Raw SGCC DataFrame (from load_sgcc)
    window_size : Number of weekly timesteps per sample (keep at 26)
    verbose     : Print progress messages

    Returns
    -------
    X           : FloatTensor (N, 26, 2)  — channel 0 = kWh, channel 1 = GLI
    y           : FloatTensor (N,)        — 0 = normal, 1 = theft
    metadata    : dict with consumer / window bookkeeping arrays
    """

    # ── Step 1 : identify columns ────────────────────────────────────────────
    id_col    = None
    label_col = None
    date_cols: List[str] = []

    # Try known column names first
    if "CONS_NO" in df.columns and "FLAG" in df.columns:
        id_col    = "CONS_NO"
        label_col = "FLAG"
        date_cols = [c for c in df.columns if c not in (id_col, label_col)]
    else:
        # Auto-detect: date columns are parseable by pd.to_datetime
        for col in df.columns:
            try:
                pd.to_datetime(col)
                date_cols.append(col)
            except (ValueError, TypeError):
                pass
        meta_cols = [c for c in df.columns if c not in date_cols]
        if len(meta_cols) >= 2:
            id_col    = meta_cols[0]
            label_col = meta_cols[1]
        elif len(meta_cols) == 1:
            id_col    = meta_cols[0]
            label_col = meta_cols[0]   # degenerate; handle below
        else:
            raise ValueError(
                "Cannot identify consumer-ID / label columns. "
                "Expected CONS_NO and FLAG columns in the CSV."
            )

    if not date_cols:
        raise ValueError("No date columns found in SGCC CSV.")

    cons_ids = df[id_col].values.astype(str)
    labels   = df[label_col].values.astype(int)
    raw_data = df[date_cols].values.astype(float)   # (N_consumers, N_days)

    if verbose:
        theft_rate = labels.mean()
        print(f"[SGCC] Consumers : {len(cons_ids):,}")
        print(f"[SGCC] Date cols  : {len(date_cols)}  "
              f"({date_cols[0]} → {date_cols[-1]})")
        print(f"[SGCC] Theft rate : {theft_rate:.3%}  "
              f"({labels.sum():,} / {len(labels):,})")

    # ── Step 2 : drop >50 % NaN consumers; interpolate the rest ─────────────
    nan_frac = np.isnan(raw_data).mean(axis=1)
    valid    = nan_frac <= 0.50
    if verbose:
        print(f"[SGCC] Dropping {(~valid).sum():,} consumers "
              f"(>{50}% NaN).  Remaining: {valid.sum():,}")

    raw_data = raw_data[valid]
    labels   = labels[valid]
    cons_ids = cons_ids[valid]

    # Linear interpolation along the time axis for each consumer
    consumption_clean = np.empty_like(raw_data)
    for i in range(len(raw_data)):
        s = pd.Series(raw_data[i]).interpolate(
            method="linear", limit_direction="both"
        )
        consumption_clean[i] = s.fillna(0.0).values

    # ── Step 3 : per-consumer min-max normalisation (daily level) ─────────────
    c_min  = consumption_clean.min(axis=1, keepdims=True)
    c_max  = consumption_clean.max(axis=1, keepdims=True)
    c_rng  = np.where(c_max - c_min > 0, c_max - c_min, 1.0)
    daily_norm = (consumption_clean - c_min) / c_rng   # (N, N_days) in [0,1]

    # ── Step 4 : daily → weekly aggregation ──────────────────────────────────
    n_consumers, n_days = daily_norm.shape
    n_full_weeks = n_days // 7
    daily_trim   = daily_norm[:, : n_full_weeks * 7]
    # (N, n_full_weeks, 7) → sum over last axis → (N, n_full_weeks)
    weekly_raw = daily_trim.reshape(n_consumers, n_full_weeks, 7).sum(axis=2)

    # Re-normalise weekly totals per consumer → [0, 1]
    w_min = weekly_raw.min(axis=1, keepdims=True)
    w_max = weekly_raw.max(axis=1, keepdims=True)
    w_rng = np.where(w_max - w_min > 0, w_max - w_min, 1.0)
    weekly = (weekly_raw - w_min) / w_rng   # (N_consumers, n_full_weeks)

    if verbose:
        print(f"[SGCC] Weekly shape: {weekly.shape}  "
              f"(consumers × weeks = {n_consumers} × {n_full_weeks})")

    # ── Step 5 : compute Grid Load Index (GLI) ────────────────────────────────
    # GLI(t) = Σ_i C_i(t) / max_t [ Σ_i C_i(t) ]
    # Treat all consumers as one virtual substation (no topology data for SGCC)
    substation_load = weekly.sum(axis=0)          # (n_full_weeks,)
    gli_denom       = substation_load.max()
    gli             = substation_load / (gli_denom if gli_denom > 0 else 1.0)

    # ── Step 6 : sliding 26-week windows ──────────────────────────────────────
    n_windows_per_consumer = n_full_weeks - window_size + 1
    if n_windows_per_consumer <= 0:
        raise ValueError(
            f"Not enough weeks ({n_full_weeks}) to form even one "
            f"{window_size}-week window.  Check dataset dates."
        )

    total_windows = n_consumers * n_windows_per_consumer
    X_kwh = np.empty((total_windows, window_size), dtype=np.float32)
    X_gli = np.empty((total_windows, window_size), dtype=np.float32)
    y_arr = np.empty(total_windows, dtype=np.float32)
    cons_idx_arr  = np.empty(total_windows, dtype=np.int32)
    win_start_arr = np.empty(total_windows, dtype=np.int32)

    idx = 0
    for i in range(n_consumers):
        for w in range(n_windows_per_consumer):
            X_kwh[idx] = weekly[i, w : w + window_size]
            X_gli[idx] = gli[w : w + window_size]
            y_arr[idx] = labels[i]
            cons_idx_arr[idx]  = i
            win_start_arr[idx] = w
            idx += 1

    # ── Step 7 : stack into (N, 26, 2) ────────────────────────────────────────
    X = np.stack([X_kwh, X_gli], axis=2)   # (N, 26, 2)

    if verbose:
        print(f"[SGCC] Final tensor : {X.shape}   "
              f"theft prevalence = {y_arr.mean():.3%}")

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y_arr)

    metadata: Dict = {
        "cons_ids":       cons_ids,          # (N_consumers,)  string IDs
        "labels":         labels,             # (N_consumers,)  per-consumer label
        "consumer_idx":   cons_idx_arr,       # (N,)  which consumer each window belongs to
        "win_start_week": win_start_arr,      # (N,)  absolute week of window start
        "n_consumers":    n_consumers,
        "n_full_weeks":   n_full_weeks,
        "n_win_per_cons": n_windows_per_consumer,
        "window_size":    window_size,
        "weekly":         weekly,             # (N_consumers, n_full_weeks) normalised weekly kWh
        "gli":            gli,               # (n_full_weeks,)
    }

    return X_tensor, y_tensor, metadata


# ─────────────────────────────────────────────────────────────────────────────
#  Tabular Features for XGBoost Edge Filter
# ─────────────────────────────────────────────────────────────────────────────

def compute_tabular_features(X_np: np.ndarray) -> np.ndarray:
    """
    Derive 5 tabular features from each (26, 2) window for the XGBoost tier.

    Features
    --------
    0  variance of kWh over 26 weeks
    1  skewness of kWh over 26 weeks
    2  peak-to-average ratio  (max / mean)
    3  mean GLI over 26 weeks
    4  std  GLI over 26 weeks

    Parameters
    ----------
    X_np : numpy array  (N, 26, 2)

    Returns
    -------
    features : numpy array  (N, 5)  float32
    """
    kwh = X_np[:, :, 0]   # (N, 26)
    gli = X_np[:, :, 1]   # (N, 26)

    var_kwh = kwh.var(axis=1)
    skw_kwh = np.apply_along_axis(scipy_skew, 1, kwh)
    par_kwh = kwh.max(axis=1) / np.where(kwh.mean(axis=1) > 0,
                                          kwh.mean(axis=1), 1e-8)
    mean_gli = gli.mean(axis=1)
    std_gli  = gli.std(axis=1)

    return np.column_stack(
        [var_kwh, skw_kwh, par_kwh, mean_gli, std_gli]
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Temporal Train / Test Split (for Experiment 1 preprocessing check)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(
    X: torch.Tensor,
    y: torch.Tensor,
    metadata: Dict,
    test_frac: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split windows by absolute start week (temporal order, no shuffling).
    Used for the simple train/test split needed during preprocessing checks.

    For Experiment 2 (walk-forward) a more elaborate split is applied in
    training/train_walkforward.py.
    """
    win_start = metadata["win_start_week"]
    n_weeks   = metadata["n_full_weeks"]
    threshold = int(n_weeks * (1 - test_frac))

    train_mask = win_start < threshold
    test_mask  = win_start >= threshold

    return (X[train_mask], y[train_mask],
            X[test_mask],  y[test_mask])


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_sgcc_pipeline(
    data_dir: str,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Load + preprocess SGCC; optionally cache the result as a .pt file.

    Parameters
    ----------
    data_dir   : directory containing the SGCC CSV
    cache_path : if given, save/load processed tensors here (faster reruns)
    verbose    : print progress

    Returns
    -------
    X, y, metadata
    """
    if cache_path and os.path.isfile(cache_path):
        if verbose:
            print(f"[SGCC] Loading cached tensors from {cache_path}")
        checkpoint = torch.load(cache_path, map_location="cpu")
        return checkpoint["X"], checkpoint["y"], checkpoint["metadata"]

    df         = load_sgcc(data_dir)
    X, y, meta = preprocess_sgcc(df, verbose=verbose)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"X": X, "y": y, "metadata": meta}, cache_path)
        if verbose:
            print(f"[SGCC] Cached tensors → {cache_path}")

    return X, y, meta


# ─────────────────────────────────────────────────────────────────────────────
#  CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sgcc"
    X, y, meta = run_sgcc_pipeline(data_dir)
    print(f"\n✓  X={X.shape}  y={y.shape}  theft={y.mean():.3%}")

    feats = compute_tabular_features(X.numpy())
    print(f"✓  Tabular features for XGBoost: {feats.shape}")
