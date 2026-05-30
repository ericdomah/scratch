"""
TDD2022 Dataset Preprocessing Pipeline
=======================================
Converts the TDD2022 (Mendeley Data DOI: 10.17632/c3c7329tjj.1) hourly CSV
into a (N, 26, 2) tensor matching the GridGuardUniversalHybrid input format.

Real df.csv layout (Mendeley download):
  - Long-format: one row per hour per (building-class, theft-type) combination
  - Columns: '0' (hour index), 'Electricity:Facility [kW](Hourly)' (consumption),
    'Class' (16 building types), 'theft' (Normal | Theft1..Theft6)
  - Each (Class, theft) pair = one "consumer" time series
  - 16 classes × 7 theft/normal labels × 35,040 h ≈ 208 weeks per consumer

Pipeline steps
--------------
1.  Load CSV — detect native TDD2022 format or fallback to wide/long
2.  Group by (Class, theft) → separate consumer time series
3.  Aggregate hourly → weekly totals (168 h per week)
4.  Per-consumer min-max normalisation → [0, 1]
5.  Sliding 26-week windows
6.  Binary labels: all Theft* → 1, Normal → 0
7.  Compute GLI across all consumers
8.  Stack into (N, 26, 2) tensor
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW_SIZE    = 26    # weeks — must match thesis architecture
HOURS_PER_WEEK = 168   # 7 × 24

# Column name in the real Mendeley df.csv
_TDD_KWH_COL   = "Electricity:Facility [kW](Hourly)"
_TDD_CLASS_COL = "Class"
_TDD_THEFT_COL = "theft"


# ─────────────────────────────────────────────────────────────────────────────
#  1.  Loader & Format Detection
# ─────────────────────────────────────────────────────────────────────────────

def load_tdd2022(data_dir: str) -> pd.DataFrame:
    """Locate and load the TDD2022 CSV from *data_dir*."""
    data_dir = os.path.abspath(data_dir)
    csv_files = sorted(
        f for f in os.listdir(data_dir) if f.lower().endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{data_dir}'. "
            "Download the TDD2022 CSV from Mendeley DOI:10.17632/c3c7329tjj.1 "
            "and place it in this directory."
        )

    # Pick largest CSV file (df.csv is ~61 MB, much larger than any index file)
    csv_sizes = {f: os.path.getsize(os.path.join(data_dir, f)) for f in csv_files}
    target    = max(csv_sizes, key=csv_sizes.get)
    csv_path  = os.path.join(data_dir, target)

    print(f"[TDD2022] Loading: {csv_path}  ({csv_sizes[target]/1e6:.1f} MB)")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"[TDD2022] Raw shape: {df.shape}")
    print(f"[TDD2022] Columns: {list(df.columns)}")
    return df


def _detect_format(df: pd.DataFrame) -> str:
    """
    Detect CSV format:
      'tdd_native' — real Mendeley df.csv (Class + theft + Electricity columns)
      'wide'       — rows=hours, columns=consumers
      'long'       — generic long format
    """
    cols = list(df.columns)
    has_elec  = any("Electricity" in c and "Facility" in c for c in cols)
    has_class = _TDD_CLASS_COL in cols
    has_theft = _TDD_THEFT_COL in cols
    if has_elec and has_class and has_theft:
        return "tdd_native"
    cols_lower = [c.lower() for c in cols]
    if any(k in cols_lower for k in ("consumer_id", "label", "kwh", "consumption")):
        return "long"
    return "wide"


# ─────────────────────────────────────────────────────────────────────────────
#  Native TDD2022 handler  (real Mendeley df.csv)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tdd_native(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse the real Mendeley df.csv format.

    Structure:
      - Each row = one hourly reading for a (Class, theft) consumer group
      - Groups are identified by unique (Class, theft) combinations
      - Consumption column: 'Electricity:Facility [kW](Hourly)'
      - Label: theft == 'Normal' -> 0,  'Theft1'..'Theft6' -> 1

    Returns
    -------
    consumption : (N_consumers, N_hours)  float32 array
    labels      : (N_consumers,)          int array (0/1)
    """
    # Sort by (Class, theft, hour-index) to guarantee temporal order
    hour_col = "0"  # sequential hour index column
    sort_cols = [_TDD_CLASS_COL, _TDD_THEFT_COL]
    if hour_col in df.columns:
        sort_cols.append(hour_col)
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    # Group by (Class, theft)
    groups = df_sorted.groupby([_TDD_CLASS_COL, _TDD_THEFT_COL], sort=False)

    series_list: List[np.ndarray] = []
    label_list:  List[int]        = []
    consumer_names: List[str]     = []

    for (cls, theft_type), grp in groups:
        kwh_vals = grp[_TDD_KWH_COL].values.astype(np.float32)
        series_list.append(kwh_vals)
        # Binary label: Normal=0, anything else=1
        label_list.append(0 if str(theft_type).strip().lower() == "normal" else 1)
        consumer_names.append(f"{cls}__{theft_type}")

    if not series_list:
        raise ValueError("No consumer groups found in TDD2022 native format.")

    # Pad all series to the same length (in case group lengths differ slightly)
    max_len     = max(len(s) for s in series_list)
    n_consumers = len(series_list)
    consumption = np.zeros((n_consumers, max_len), dtype=np.float32)
    for i, s in enumerate(series_list):
        consumption[i, : len(s)] = s

    return consumption, np.array(label_list, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
#  Wide-format handler (Format A)
# ─────────────────────────────────────────────────────────────────────────────

_THEFT_PATTERNS = re.compile(r"theft|attack|tamper|anomal", re.IGNORECASE)
_NORMAL_PATTERNS = re.compile(r"normal|legit|genuine", re.IGNORECASE)


def _parse_wide(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Parse wide-format TDD2022.

    Returns
    -------
    consumption : (N_consumers, N_hours)  float array
    labels      : (N_consumers,)          int array (0/1)
    col_indices : list of DataFrame column indices used
    """
    # Drop any obvious non-numeric / timestamp columns
    drop_cols = []
    consumer_cols = []
    for col in df.columns:
        try:
            df[col].astype(float)
            consumer_cols.append(col)
        except (ValueError, TypeError):
            drop_cols.append(col)

    data   = df[consumer_cols].values.T.astype(float)   # (N_consumers, N_hours)
    labels = []
    for col in consumer_cols:
        if _THEFT_PATTERNS.search(col):
            labels.append(1)
        else:
            labels.append(0)   # default to normal if ambiguous
    labels = np.array(labels, dtype=int)
    return data, labels, list(range(len(consumer_cols)))


# ─────────────────────────────────────────────────────────────────────────────
#  Long-format handler (Format B)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_long(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse long-format TDD2022.

    Returns
    -------
    consumption : (N_consumers, N_hours)  — zero-padded if unequal lengths
    labels      : (N_consumers,)
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Identify key columns
    label_col = cols_lower.get("label", cols_lower.get("class",
                cols_lower.get("type", None)))
    kwh_col   = cols_lower.get("kwh", cols_lower.get("consumption",
                cols_lower.get("value", cols_lower.get("energy", None))))
    id_col    = cols_lower.get("consumer_id", cols_lower.get("id",
                cols_lower.get("customer_id", None)))

    if kwh_col is None:
        # Last numeric column is most likely the kWh reading
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        kwh_col = numeric_cols[-1] if numeric_cols else None
    if id_col is None:
        id_col = df.columns[0]

    if kwh_col is None:
        raise ValueError("Cannot identify kWh consumption column in TDD2022 CSV.")

    groups = df.groupby(id_col)
    series_list: List[np.ndarray] = []
    label_list: List[int] = []

    for name, grp in groups:
        kwh_vals = grp[kwh_col].values.astype(float)
        series_list.append(kwh_vals)
        if label_col and label_col in grp.columns:
            raw_label = grp[label_col].iloc[0]
            # Binary: 0 = normal, 1 = any theft class
            if isinstance(raw_label, str):
                label = 0 if _NORMAL_PATTERNS.search(raw_label) else 1
            else:
                label = 0 if int(raw_label) == 0 else 1
        else:
            label = 0
        label_list.append(label)

    # Pad all series to the same length
    max_len = max(len(s) for s in series_list)
    consumption = np.zeros((len(series_list), max_len), dtype=float)
    for i, s in enumerate(series_list):
        consumption[i, : len(s)] = s

    return consumption, np.array(label_list, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
#  2-7.  Full Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_tdd2022(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Run the complete TDD2022 preprocessing pipeline.

    Parameters
    ----------
    df          : Raw TDD2022 DataFrame
    window_size : Number of weekly timesteps per sample (keep at 26)
    verbose     : Print progress

    Returns
    -------
    X           : FloatTensor (N, 26, 2)
    y           : FloatTensor (N,)
    metadata    : bookkeeping dict
    """

    # ── Step 1 : parse by format ──────────────────────────────────────────────
    fmt = _detect_format(df)
    if verbose:
        print(f"[TDD2022] Detected format: {fmt!r}")

    if fmt == "tdd_native":
        consumption, labels = _parse_tdd_native(df)
    elif fmt == "wide":
        consumption, labels, _ = _parse_wide(df)
    else:
        consumption, labels = _parse_long(df)

    n_consumers, n_hours = consumption.shape
    if verbose:
        print(f"[TDD2022] Consumers : {n_consumers}")
        print(f"[TDD2022] Hours     : {n_hours}")
        print(f"[TDD2022] Theft rate: {labels.mean():.3%}  "
              f"({labels.sum()} / {len(labels)})")

    # ── Step 2 : hourly → weekly totals ──────────────────────────────────────
    n_full_weeks = n_hours // HOURS_PER_WEEK
    if n_full_weeks < window_size:
        raise ValueError(
            f"TDD2022 has only {n_full_weeks} full weeks but "
            f"window_size={window_size}.  "
            f"Check that you have ≥{window_size} weeks of data."
        )
    hour_trim = consumption[:, : n_full_weeks * HOURS_PER_WEEK]
    # (N, n_full_weeks, 168) → sum → (N, n_full_weeks)
    weekly_raw = hour_trim.reshape(n_consumers, n_full_weeks, HOURS_PER_WEEK).sum(axis=2)

    # ── Step 3 : per-consumer min-max normalisation ───────────────────────────
    w_min = weekly_raw.min(axis=1, keepdims=True)
    w_max = weekly_raw.max(axis=1, keepdims=True)
    w_rng = np.where(w_max - w_min > 0, w_max - w_min, 1.0)
    weekly = (weekly_raw - w_min) / w_rng   # (N, n_full_weeks)

    # ── Step 4 : sliding 26-week windows ─────────────────────────────────────
    n_win = n_full_weeks - window_size + 1
    total = n_consumers * n_win
    X_kwh = np.empty((total, window_size), dtype=np.float32)
    X_gli = np.empty((total, window_size), dtype=np.float32)
    y_arr = np.empty(total, dtype=np.float32)
    cons_idx_arr  = np.empty(total, dtype=np.int32)
    win_start_arr = np.empty(total, dtype=np.int32)

    # ── Step 6 : compute GLI (same definition as SGCC) ───────────────────────
    substation_load = weekly.sum(axis=0)          # (n_full_weeks,)
    gli_denom       = substation_load.max()
    gli             = substation_load / (gli_denom if gli_denom > 0 else 1.0)

    idx = 0
    for i in range(n_consumers):
        for w in range(n_win):
            X_kwh[idx] = weekly[i, w : w + window_size]
            X_gli[idx] = gli[w : w + window_size]
            y_arr[idx] = labels[i]
            cons_idx_arr[idx]  = i
            win_start_arr[idx] = w
            idx += 1

    # ── Step 7 : stack into (N, 26, 2) ───────────────────────────────────────
    X = np.stack([X_kwh, X_gli], axis=2)

    if verbose:
        print(f"[TDD2022] Final tensor: {X.shape}  "
              f"theft={y_arr.mean():.3%}")

    return (
        torch.FloatTensor(X),
        torch.FloatTensor(y_arr),
        {
            "consumer_idx":   cons_idx_arr,
            "win_start_week": win_start_arr,
            "n_consumers":    n_consumers,
            "n_full_weeks":   n_full_weeks,
            "n_win_per_cons": n_win,
            "window_size":    window_size,
            "labels":         labels,
            "weekly":         weekly,
            "gli":            gli,
            "format":         fmt,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience entry-point
# ─────────────────────────────────────────────────────────────────────────────

def run_tdd2022_pipeline(
    data_dir: str,
    cache_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Load + preprocess TDD2022; optionally cache the result."""
    if cache_path and os.path.isfile(cache_path):
        if verbose:
            print(f"[TDD2022] Loading cached tensors from {cache_path}")
        ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
        return ckpt["X"], ckpt["y"], ckpt["metadata"]

    df         = load_tdd2022(data_dir)
    X, y, meta = preprocess_tdd2022(df, verbose=verbose)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({"X": X, "y": y, "metadata": meta}, cache_path)
        if verbose:
            print(f"[TDD2022] Cached tensors → {cache_path}")

    return X, y, meta


# ─────────────────────────────────────────────────────────────────────────────
#  CLI smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/tdd2022"
    X, y, meta = run_tdd2022_pipeline(data_dir)
    print(f"\n✓  X={X.shape}  y={y.shape}  theft={y.mean():.3%}")
