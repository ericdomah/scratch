"""
SGCC Dataset Preprocessing Pipeline  (Phase 1 — Real-Data Training)
=====================================================================
Converts the raw SGCC CSV (one row per consumer, daily kWh readings 2014-01-01
to 2016-10-31) into a (N, 26, 2) tensor ready for GridGuardUniversalHybrid.

ONE-WINDOW-PER-CONSUMER RULE (NON-NEGOTIABLE):
----------------------------------------------
SGCC labels are at the consumer level. FLAG = 0 (normal) or FLAG = 1 (theft).
There are NO theft onset timestamps. Using sliding windows on theft consumers
would assign label = 1 to windows from BEFORE the theft began (i.e., normal
consumption mislabelled as theft). This destroys interpretability.

CORRECT APPROACH: extract EXACTLY ONE 26-week window per consumer — the LAST
26 weeks of their available data. The most recent period is the one most likely
to contain the confirmed theft behaviour.

Expected output: ~40,000–42,000 samples, one per consumer, ~5% theft.
If you see millions of samples the windowing logic is wrong — STOP and fix it.

Pipeline steps
--------------
1.  Load CSV — detect CONS_NO / FLAG / date columns automatically
2.  Sort date columns chronologically
3.  Drop consumers with >50 % NaN readings
4.  Linear-interpolate remaining NaN gaps; fill edges with 0.0
5.  Aggregate daily → weekly (7-day sums)
6.  Extract LAST 26 weeks per consumer (ONE window only — see rule above)
    Drop consumers with fewer than 26 weekly readings
7.  Per-consumer min-max normalisation → [0, 1]
8.  Compute Grid Load Index (GLI) across all surviving consumers
9.  Stack into (N, 26, 2) tensor
10. Verification: print summary, assert ~5 % theft prevalence
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


# -----------------------------------------------------------------------------
#  1.  Loader
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
#  2–9.  Full Preprocessing
# -----------------------------------------------------------------------------

def preprocess_sgcc(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Run the complete SGCC preprocessing pipeline (one window per consumer).

    Parameters
    ----------
    df          : Raw SGCC DataFrame (from load_sgcc)
    window_size : Number of weekly timesteps per sample (keep at 26)
    verbose     : Print progress messages

    Returns
    -------
    X           : FloatTensor (N, 26, 2)  — channel 0 = kWh, channel 1 = GLI
    y           : FloatTensor (N,)        — 0 = normal, 1 = theft
    metadata    : dict with consumer bookkeeping arrays
    """

    # -- Step 1 : identify columns --------------------------------------------
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
            label_col = meta_cols[0]
        else:
            raise ValueError(
                "Cannot identify consumer-ID / label columns. "
                "Expected CONS_NO and FLAG columns in the CSV."
            )

    if not date_cols:
        raise ValueError("No date columns found in SGCC CSV.")

    # -- Step 2 : sort date columns chronologically ---------------------------
    try:
        date_cols = sorted(date_cols, key=pd.to_datetime)
    except Exception:
        pass  # if not parseable, keep original order

    cons_ids = df[id_col].values.astype(str)
    labels   = df[label_col].values.astype(int)
    raw_data = df[date_cols].values.astype(float)   # (N_consumers, N_days)

    if verbose:
        theft_rate = labels.mean()
        print(f"[SGCC] Consumers : {len(cons_ids):,}")
        print(f"[SGCC] Date cols  : {len(date_cols)}  "
              f"({date_cols[0]} -> {date_cols[-1]})")
        print(f"[SGCC] Theft rate : {theft_rate:.3%}  "
              f"({labels.sum():,} / {len(labels):,})")

    # -- Step 3 : drop >50 % NaN consumers ------------------------------------
    nan_frac = np.isnan(raw_data).mean(axis=1)
    valid    = nan_frac <= 0.50
    if verbose:
        print(f"[SGCC] Dropping {(~valid).sum():,} consumers "
              f"(>50% NaN).  Remaining: {valid.sum():,}")

    raw_data = raw_data[valid]
    labels   = labels[valid]
    cons_ids = cons_ids[valid]
    n_consumers_after_nan = len(labels)

    # -- Step 4 : linear interpolation; fill any edge NaN with 0.0 -----------
    if verbose:
        print(f"[SGCC] Interpolating missing values...")
    consumption_clean = np.empty_like(raw_data)
    for i in range(len(raw_data)):
        s = pd.Series(raw_data[i]).interpolate(
            method="linear", limit_direction="both"
        )
        consumption_clean[i] = s.fillna(0.0).values

    # -- Step 5 : aggregate daily → weekly (7-day sums) -----------------------
    n_consumers, n_days = consumption_clean.shape
    n_full_weeks = n_days // 7
    daily_trim   = consumption_clean[:, : n_full_weeks * 7]
    # (N, n_full_weeks, 7) → sum over days → (N, n_full_weeks)
    weekly_raw = daily_trim.reshape(n_consumers, n_full_weeks, 7).sum(axis=2)

    if verbose:
        print(f"[SGCC] Weekly aggregation: {n_consumers} consumers × "
              f"{n_full_weeks} weeks")

    # -- Step 6 : ⚠️  EXTRACT LAST 26 WEEKS ONLY — ONE WINDOW PER CONSUMER ---
    #
    #  DO NOT create sliding windows here.
    #  Each consumer contributes exactly ONE sample: their last 26 weeks.
    #  This is the only defensible labelling strategy given that SGCC provides
    #  no theft onset timestamps.
    #
    if verbose:
        print(f"[SGCC] Extracting LAST {window_size} weeks per consumer "
              f"(ONE window per consumer — no sliding)...")

    # Drop consumers with fewer than 26 weekly readings
    enough = n_full_weeks >= window_size
    if not enough:
        raise ValueError(
            f"Dataset has only {n_full_weeks} weeks — need at least "
            f"{window_size}. Check that the full SGCC CSV was supplied."
        )

    # For each consumer: take only the LAST window_size weeks
    windows_kwh = weekly_raw[:, -window_size:]   # (N, 26)
    labels_out  = labels                          # (N,)  — one label per consumer
    cons_ids_out = cons_ids

    n_samples = len(windows_kwh)
    if verbose:
        print(f"[SGCC] [OK] Samples after one-window-per-consumer: "
              f"{n_samples:,}  (expected ~40,000-42,000)")
        if n_samples > 50000:
            raise ValueError(
                f"ERROR: {n_samples} samples found. "
                f"Expected ~42000. Sliding window was applied. "
                f"Fix Step 5 before any training."
            )
        print(f"Sample count: {n_samples} — OK if 38000 to 44000")

    # -- Step 7 : per-consumer min-max normalisation --------------------------
    w_min = windows_kwh.min(axis=1, keepdims=True)
    w_max = windows_kwh.max(axis=1, keepdims=True)
    w_rng = np.where(w_max - w_min > 0, w_max - w_min, 1.0)
    kWh_norm = (windows_kwh - w_min) / w_rng   # (N, 26) in [0, 1]
    # Flat consumers (max == min) → all zeros (already handled by w_rng=1.0 above)

    # -- Step 8 : compute Grid Load Index (GLI) --------------------------------
    # GLI(t) = Σ_i C_i(t) / max_t [ Σ_i C_i(t) ]
    # Treat all surviving consumers as one virtual substation.
    # gli shape: (26,) — same value for every consumer at each timestep.
    aggregate_load = kWh_norm.sum(axis=0)            # (26,)
    gli_denom      = aggregate_load.max()
    gli            = aggregate_load / (gli_denom if gli_denom > 0 else 1.0)
    gli_matrix     = np.tile(gli, (n_samples, 1))    # (N, 26)

    # -- Step 9 : stack into (N, 26, 2) ----------------------------------------
    X = np.stack([kWh_norm, gli_matrix], axis=2)   # (N, 26, 2)
    y = labels_out.astype(np.float32)

    # -- Verification summary --------------------------------------------------
    n_normal = int((y == 0).sum())
    n_theft  = int((y == 1).sum())
    prevalence = n_theft / len(y) * 100.0

    print(f"\n{'='*55}")
    print(f"  SGCC PREPROCESSING — VERIFICATION SUMMARY")
    print(f"{'='*55}")
    print(f"  Total samples (one per consumer) : {len(y):,}")
    print(f"  Normal consumers (FLAG=0)        : {n_normal:,}")
    print(f"  Theft  consumers (FLAG=1)        : {n_theft:,}")
    print(f"  Theft prevalence                 : {prevalence:.2f}%")
    print(f"  Tensor shape                     : {X.shape}")
    print(f"{'='*55}\n")

    if not (1.0 <= prevalence <= 25.0):
        raise ValueError(
            f"Theft prevalence = {prevalence:.2f}% is outside the expected "
            f"1–25 % range. Data has likely been processed incorrectly. "
            f"Expected ~5 % for SGCC. STOP and investigate before continuing."
        )

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    # Sort consumer IDs as temporal proxy for walk-forward (Experiment 2)
    try:
        sort_order = np.argsort(cons_ids_out.astype(np.int64))
    except ValueError:
        sort_order = np.argsort(cons_ids_out)

    metadata: Dict = {
        "cons_ids":     cons_ids_out,      # (N,) string IDs — unsorted
        "sort_order":   sort_order,        # (N,) sort indices by CONS_NO
        "labels":       labels_out,        # (N,) per-consumer labels
        "n_consumers":  n_samples,
        "n_full_weeks": n_full_weeks,
        "window_size":  window_size,
        "n_normal":     n_normal,
        "n_theft":      n_theft,
        "prevalence_pct": prevalence,
    }

    return X_tensor, y_tensor, metadata


# -----------------------------------------------------------------------------
#  Tabular Features for XGBoost Edge Filter
# -----------------------------------------------------------------------------

def compute_tabular_features(X_np: np.ndarray) -> np.ndarray:
    """
    Derive 5 tabular features from each (26, 2) window for the XGBoost tier.

    Features
    --------
    0  variance of kWh over 26 weeks
    1  skewness of kWh over 26 weeks
    2  peak-to-average ratio  (max / mean kWh)
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

    var_kwh  = kwh.var(axis=1)
    skw_kwh  = np.apply_along_axis(scipy_skew, 1, kwh)
    mean_kwh = np.where(kwh.mean(axis=1) > 0, kwh.mean(axis=1), 1e-8)
    par_kwh  = kwh.max(axis=1) / mean_kwh
    mean_gli = gli.mean(axis=1)
    std_gli  = gli.std(axis=1)

    return np.column_stack(
        [var_kwh, skw_kwh, par_kwh, mean_gli, std_gli]
    ).astype(np.float32)


# -----------------------------------------------------------------------------
#  Convenience entry-point
# -----------------------------------------------------------------------------

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
        checkpoint = torch.load(cache_path, map_location="cpu", weights_only=False)
        return checkpoint["X"], checkpoint["y"], checkpoint["metadata"]

    df         = load_sgcc(data_dir)
    X, y, meta = preprocess_sgcc(df, verbose=verbose)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save({"X": X, "y": y, "metadata": meta}, cache_path)
        if verbose:
            print(f"[SGCC] Cached tensors -> {cache_path}")

    return X, y, meta


# -----------------------------------------------------------------------------
#  CLI smoke-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sgcc"
    X, y, meta = run_sgcc_pipeline(data_dir)
    print(f"\n[OK]  X={X.shape}  y={y.shape}  theft={y.mean():.3%}")

    feats = compute_tabular_features(X.numpy())
    print(f"[OK]  Tabular features for XGBoost: {feats.shape}")
