"""
TDD2022 Dataset Preprocessing Pipeline
========================================
Long-format CSV: one row per hourly timestep.
Expected columns:
  - numeric energy columns (kW readings)
  - 'Class'  : building profile / consumer ID
  - 'theft'  : label column ('Normal' = 0, anything else = 1)

SLIDING WINDOWS are valid for TDD2022 — theft is injected throughout
the full time series so every window from a theft consumer has label=1.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW_SIZE    = 26
HOURS_PER_WEEK = 168

# Known label column names (checked in order)
LABEL_CANDIDATES = ["theft", "Theft", "label", "Label", "FLAG", "flag", "type", "Type"]
# Known ID column names
ID_CANDIDATES    = ["Class", "class", "consumer_id", "ID", "id", "CONS_NO", "profile_id"]


def load_tdd2022(tdd_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load TDD2022 long-format CSV and return (N_windows, 26, 2) arrays.

    Parameters
    ----------
    tdd_path : path to df.csv or directory containing df.csv

    Returns
    -------
    X : float32 (N_windows, 26, 2)
    y : float32 (N_windows,)
    """
    import os

    if os.path.isdir(tdd_path):
        candidates = [f for f in os.listdir(tdd_path) if f.endswith(".csv")]
        if not candidates:
            raise FileNotFoundError(f"No CSV found in {tdd_path}")
        csv_file = os.path.join(tdd_path, candidates[0])
    else:
        csv_file = tdd_path

    print(f"[TDD2022] Loading: {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"[TDD2022] Raw shape: {df.shape}")
    print(f"[TDD2022] Columns: {list(df.columns)}")
    print(f"[TDD2022] First 3 rows:\n{df.head(3)}\n")

    # ------------------------------------------------------------------ identify columns
    # Label column
    label_col = None
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        # Fall back: last non-numeric column
        for c in reversed(df.columns):
            if df[c].dtype == object:
                label_col = c
                break
    if label_col is None:
        raise ValueError("Cannot find label column in TDD2022 CSV.")

    # Consumer / profile ID column
    id_col = None
    for c in ID_CANDIDATES:
        if c in df.columns and c != label_col:
            id_col = c
            break
    if id_col is None:
        # Fall back: first non-numeric, non-label column
        for c in df.columns:
            if c != label_col and df[c].dtype == object:
                id_col = c
                break
    if id_col is None:
        raise ValueError("Cannot find consumer ID column in TDD2022 CSV.")

    # Consumption columns: numeric only, excluding id and label
    skip = {label_col, id_col}
    # Also skip pure integer index columns (column named '0', '1', etc.)
    for c in df.columns:
        if str(c).strip().lstrip('-').isdigit():
            skip.add(c)

    cons_cols = []
    for c in df.columns:
        if c in skip:
            continue
        try:
            pd.to_numeric(df[c], errors='raise')
            cons_cols.append(c)
        except (ValueError, TypeError):
            pass

    print(f"[TDD2022] id_col={id_col!r}  label_col={label_col!r}  "
          f"consumption_cols={len(cons_cols)}: {cons_cols[:4]} ...")

    if not cons_cols:
        raise ValueError("No numeric consumption columns found in TDD2022 CSV.")

    # Use total facility electricity as primary signal (first consumption column)
    primary_col = cons_cols[0]
    print(f"[TDD2022] Primary consumption column: {primary_col!r}")

    # ------------------------------------------------------------------ group by consumer
    # Get consumer-level label (majority vote in case of mixed rows)
    consumer_labels = {}
    for cid, grp in df.groupby(id_col):
        raw = grp[label_col].iloc[0]
        consumer_labels[cid] = _binary_label(raw)

    consumer_ids = sorted(consumer_labels.keys())
    print(f"[TDD2022] Unique consumers (building profiles): {len(consumer_ids):,}")

    all_windows: list[np.ndarray] = []
    all_labels:  list[int]        = []

    for cid in consumer_ids:
        grp    = df[df[id_col] == cid]
        label  = consumer_labels[cid]
        hourly = pd.to_numeric(grp[primary_col], errors='coerce').fillna(0.0).values.astype(float)

        if len(hourly) < HOURS_PER_WEEK:
            continue

        # Aggregate hourly -> weekly
        n_full = len(hourly) // HOURS_PER_WEEK
        weekly = hourly[:n_full * HOURS_PER_WEEK].reshape(n_full, HOURS_PER_WEEK).sum(axis=1)

        if len(weekly) < WINDOW_SIZE:
            continue

        # Per-consumer normalisation
        w_min = weekly.min(); w_max = weekly.max()
        w_rng = (w_max - w_min) if (w_max - w_min) > 0 else 1.0
        weekly_norm = (weekly - w_min) / w_rng

        # Sliding windows, stride=1
        for start in range(0, len(weekly_norm) - WINDOW_SIZE + 1):
            all_windows.append(weekly_norm[start:start + WINDOW_SIZE])
            all_labels.append(label)

    if not all_windows:
        raise ValueError("No valid windows extracted from TDD2022. Check file format.")

    windows_np = np.array(all_windows, dtype=np.float32)   # (N, 26)
    labels_np  = np.array(all_labels,  dtype=np.float32)
    n_samples  = len(windows_np)

    # GLI
    agg       = windows_np.sum(axis=0)
    gli       = agg / (agg.max() if agg.max() > 0 else 1.0)
    gli_mat   = np.tile(gli, (n_samples, 1))

    X = np.stack([windows_np, gli_mat], axis=2)
    y = labels_np

    n_normal  = int((y == 0).sum())
    n_theft   = int((y == 1).sum())
    prevalence = n_theft / len(y) * 100.0

    print(f"\n{'='*42}")
    print(f"  TDD2022 DATASET SUMMARY (sliding windows)")
    print(f"{'='*42}")
    print(f"  Total windows      : {len(y):,}")
    print(f"  Normal (label=0)   : {n_normal:,}")
    print(f"  Theft  (label=1)   : {n_theft:,}")
    print(f"  Theft prevalence   : {prevalence:.2f}%")
    print(f"  NOTE: high theft % is expected for TDD2022")
    print(f"  Tensor shape       : {X.shape}")
    print(f"{'='*42}\n")

    return X.astype(np.float32), y.astype(np.float32)


def _binary_label(raw) -> int:
    """'Normal' -> 0, anything else -> 1."""
    s = str(raw).strip().lower()
    if s in ("0", "normal", "false", "no", "none"):
        return 0
    return 1


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../../data/tdd2022"
    X, y = load_tdd2022(path)
    print(f"[OK]  X={X.shape}  y={y.shape}  theft={y.mean():.3%}")
