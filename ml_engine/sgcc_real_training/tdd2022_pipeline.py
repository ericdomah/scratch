"""
TDD2022 Dataset Preprocessing Pipeline
========================================
Loads the TDD2022 CSV and returns (N_windows, 26, 2) float32 arrays.

SLIDING WINDOWS are valid and required for TDD2022.
TDD2022 injects theft patterns throughout the full time series of each consumer.
A consumer labelled as theft has theft behaviour across ALL their readings,
so every 26-week window from that consumer legitimately carries label=1.
Without sliding windows we would have only ~113 samples — too few to be useful.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew

warnings.filterwarnings("ignore", category=RuntimeWarning)

WINDOW_SIZE  = 26    # weeks
HOURS_PER_WEEK = 168


def load_tdd2022(tdd_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess TDD2022 data with SLIDING WINDOWS (stride=1 week).

    Parameters
    ----------
    tdd_path : path to df.csv or directory containing df.csv

    Returns
    -------
    X : float32 array (N_windows, 26, 2)
    y : float32 array (N_windows,)   — 0=normal, 1=theft (any theft class)
    """
    import os

    # Resolve path — accept file or directory
    if os.path.isdir(tdd_path):
        candidates = [f for f in os.listdir(tdd_path) if f.endswith(".csv")]
        if not candidates:
            raise FileNotFoundError(f"No CSV found in {tdd_path}")
        csv_file = os.path.join(tdd_path, candidates[0])
    else:
        csv_file = tdd_path

    # Step 1 — Load CSV
    print(f"[TDD2022] Loading: {csv_file}")
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"[TDD2022] Raw shape: {df.shape}")
    print(f"[TDD2022] Columns (first 10): {list(df.columns[:10])}")
    print(f"[TDD2022] First 3 rows:\n{df.head(3)}\n")

    # Identify consumer ID, label, and consumption columns
    id_col    = _detect_id_column(df)
    label_col = _detect_label_column(df)
    cons_cols = [c for c in df.columns if c not in (id_col, label_col)]

    print(f"[TDD2022] id_col={id_col}  label_col={label_col}  "
          f"consumption_cols={len(cons_cols)}")

    # Step 2 — Group by consumer, extract full hourly sequence
    # Step 3 — Aggregate hourly -> weekly (168 h = 1 week)
    # Step 4 — Sliding window (stride=1 week)
    all_windows: list[np.ndarray] = []
    all_labels:  list[int]        = []

    consumer_ids = df[id_col].unique()
    print(f"[TDD2022] Unique consumers: {len(consumer_ids):,}")

    for cid in consumer_ids:
        rows  = df[df[id_col] == cid]
        label = _binary_label(rows[label_col].iloc[0])

        # Hourly consumption — handle both long and wide formats
        if len(cons_cols) == 1:
            # Long format: one column contains hourly readings, one row per hour
            hourly = rows[cons_cols[0]].values.astype(float)
        else:
            # Wide format: each row is one consumer, columns are hourly readings
            hourly = rows[cons_cols].values.flatten().astype(float)

        # Impute NaN
        s      = pd.Series(hourly).interpolate(method="linear", limit_direction="both")
        hourly = s.fillna(0.0).values

        # Aggregate hourly -> weekly
        n_hours        = len(hourly)
        n_full_weeks   = n_hours // HOURS_PER_WEEK
        remainder_h    = n_hours % HOURS_PER_WEEK

        if n_full_weeks == 0:
            continue  # not enough data

        weekly_full = hourly[:n_full_weeks * HOURS_PER_WEEK].reshape(
            n_full_weeks, HOURS_PER_WEEK
        ).sum(axis=1)

        if remainder_h > 0:
            partial    = np.array([hourly[n_full_weeks * HOURS_PER_WEEK:].sum()])
            weekly_raw = np.concatenate([weekly_full, partial])
        else:
            weekly_raw = weekly_full

        if len(weekly_raw) < WINDOW_SIZE:
            continue  # consumer too short

        # Per-consumer min-max normalisation
        w_min = weekly_raw.min()
        w_max = weekly_raw.max()
        w_rng = (w_max - w_min) if (w_max - w_min) > 0 else 1.0
        weekly_norm = (weekly_raw - w_min) / w_rng

        # Sliding window, stride=1 week
        n_weeks = len(weekly_norm)
        for start in range(0, n_weeks - WINDOW_SIZE + 1):
            window = weekly_norm[start : start + WINDOW_SIZE]
            all_windows.append(window)
            all_labels.append(label)

    if not all_windows:
        raise ValueError("No valid windows extracted from TDD2022.  Check file format.")

    windows_np = np.array(all_windows, dtype=np.float32)   # (N, 26)
    labels_np  = np.array(all_labels,  dtype=np.float32)   # (N,)
    n_samples  = len(windows_np)

    # Step 6 — GLI across all TDD2022 windows
    aggregate_load = windows_np.sum(axis=0)                          # (26,)
    gli_denom      = aggregate_load.max()
    gli            = aggregate_load / (gli_denom if gli_denom > 0 else 1.0)
    gli_matrix     = np.tile(gli, (n_samples, 1))                    # (N, 26)

    # Step 7 — Stack into (N, 26, 2)
    X = np.stack([windows_np, gli_matrix], axis=2)   # (N, 26, 2)
    y = labels_np

    n_normal   = int((y == 0).sum())
    n_theft    = int((y == 1).sum())
    prevalence = n_theft / len(y) * 100.0

    print(f"\n{'='*42}")
    print(f"  TDD2022 DATASET SUMMARY (sliding windows)")
    print(f"{'='*42}")
    print(f"  Total windows      : {len(y):,}")
    print(f"  Normal (label=0)   : {n_normal:,}")
    print(f"  Theft  (label=1)   : {n_theft:,}")
    print(f"  Theft prevalence   : {prevalence:.2f}%")
    print(f"  NOTE: >80% theft is expected for TDD2022")
    print(f"  Tensor shape       : {X.shape}")
    print(f"{'='*42}\n")

    return X.astype(np.float32), y.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_id_column(df: pd.DataFrame) -> str:
    candidates = ["id", "ID", "consumer_id", "CONS_NO", "customer_id",
                  "profile_id", "ProfileID"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fall back to first non-numeric column
    for c in df.columns:
        if df[c].dtype == object or df[c].nunique() > 50:
            return c
    return df.columns[0]


def _detect_label_column(df: pd.DataFrame) -> str:
    candidates = ["label", "Label", "FLAG", "flag", "class", "Class",
                  "theft", "Theft", "type", "Type"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fall back to last column
    return df.columns[-1]


def _binary_label(raw_label) -> int:
    """Map any theft class (1-6) -> 1; normal (0) -> 0."""
    try:
        v = int(raw_label)
        return 1 if v > 0 else 0
    except (ValueError, TypeError):
        label_str = str(raw_label).strip().lower()
        if label_str in ("0", "normal", "false", "no"):
            return 0
        return 1


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "../../data/tdd2022"
    X, y = load_tdd2022(path)
    print(f"[OK]  X={X.shape}  y={y.shape}  theft={y.mean():.3%}")
