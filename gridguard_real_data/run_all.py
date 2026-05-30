"""
GridGuard AI — Phase 1 Master Script
======================================
Runs all four experiments end-to-end and produces the final comparison table.

Usage
-----
    python run_all.py --sgcc_path data/sgcc --tdd_path data/tdd2022

Optional flags
--------------
    --trnc_test  data/trnc_synthetic_test.pt   path to synthetic holdout .pt
    --output_dir .                              project root (default: current dir)
    --skip_exp1                                 skip standard CV (use saved weights)
    --skip_exp2                                 skip walk-forward
    --skip_exp3                                 skip reverse transfer
    --skip_exp4                                 skip TDD2022 cross-domain

Final results table is saved to:
    results/sgcc_real_training_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from preprocessing.sgcc_pipeline    import run_sgcc_pipeline
from training.train_sgcc            import run_standard_cv
from training.train_walkforward     import run_walk_forward
from evaluation.evaluate_cross_domain import (
    run_exp3_reverse_transfer,
    run_exp4_cross_domain_tdd,
    load_sgcc_indomain_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Global seed
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42


def set_global_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
#  Final Results Table Builder
# ─────────────────────────────────────────────────────────────────────────────

EXISTING_RESULTS = [
    {
        "Experiment":  "Synthetic TRNC → TRNC (existing, Walk-Forward)",
        "F1":    0.893, "AUROC": 0.943,
        "Precision": 0.911, "Recall": 0.875, "Brier": 0.042,
        "Source": "Thesis (synthetic training)",
    },
    {
        "Experiment":  "Synthetic TRNC → SGCC cross-domain zero-shot (existing)",
        "F1":    0.783, "AUROC": 0.871,
        "Precision": 0.842, "Recall": 0.732, "Brier": "—",
        "Source": "Thesis (zero-shot)",
    },
]


def _parse_mean(val) -> str:
    """Extract a display-ready mean from a '0.9123 ± ...' string or float."""
    if isinstance(val, float):
        return f"{val:.4f}"
    s = str(val)
    return s.split("±")[0].strip()


def build_final_table(
    output_dir: str,
    exp1_df:  pd.DataFrame = None,
    exp2_df:  pd.DataFrame = None,
    exp3_df:  pd.DataFrame = None,
    exp4_df:  pd.DataFrame = None,
) -> pd.DataFrame:
    """Merge all experiment results into one comparison table."""

    rows = list(EXISTING_RESULTS)

    def _row_from_df(df: pd.DataFrame, name: str, f1_col: str,
                     auroc_col: str, prec_col: str, rec_col: str,
                     brier_col: str = None) -> dict:
        if df is None:
            return {"Experiment": name, "F1": "TBD", "AUROC": "TBD",
                    "Precision": "TBD", "Recall": "TBD", "Brier": "TBD",
                    "Source": "Phase 1"}
        # Find summary row
        summary = df[df.apply(
            lambda r: str(r.get("Fold", r.get("Round", ""))).startswith("mean"),
            axis=1
        )]
        if summary.empty:
            numeric = df[pd.to_numeric(
                df.get("Fold", df.get("Round", pd.Series())),
                errors="coerce"
            ).notna()]
            f1    = pd.to_numeric(numeric[f1_col],    errors="coerce").mean()
            auroc = pd.to_numeric(numeric[auroc_col], errors="coerce").mean()
            prec  = pd.to_numeric(numeric[prec_col],  errors="coerce").mean()
            rec   = pd.to_numeric(numeric[rec_col],   errors="coerce").mean()
            brier = (pd.to_numeric(numeric[brier_col], errors="coerce").mean()
                     if brier_col and brier_col in numeric.columns else "—")
        else:
            row   = summary.iloc[0]
            f1    = _parse_mean(row.get(f1_col))
            auroc = _parse_mean(row.get(auroc_col))
            prec  = _parse_mean(row.get(prec_col,  "—"))
            rec   = _parse_mean(row.get(rec_col,   "—"))
            brier = _parse_mean(row.get(brier_col, "—")) if brier_col else "—"

        return {
            "Experiment": name,
            "F1":        f1 if isinstance(f1, str) else round(float(f1), 4),
            "AUROC":     auroc if isinstance(auroc, str) else round(float(auroc), 4),
            "Precision": prec if isinstance(prec, str) else round(float(prec), 4),
            "Recall":    rec if isinstance(rec, str) else round(float(rec), 4),
            "Brier":     brier if isinstance(brier, str) else round(float(brier), 4),
            "Source":    "Phase 1 — Real Data",
        }

    # Experiment 1 (standard CV)
    rows.append(_row_from_df(
        exp1_df,
        "Real SGCC → SGCC (Standard CV, 10-fold)",
        "Fused_F1", "Fused_AUROC", "Fused_Precision", "Fused_Recall", "Fused_Brier",
    ))

    # Experiment 2 (walk-forward)
    rows.append(_row_from_df(
        exp2_df,
        "Real SGCC → SGCC (Walk-Forward, 7-round)",
        "GG_F1", "GG_AUROC", "GG_Precision", "GG_Recall", "GG_Brier",
    ))

    # Experiment 3 (reverse transfer)
    if exp3_df is not None and not exp3_df.empty:
        r = exp3_df.iloc[0]
        rows.append({
            "Experiment": "Real SGCC → Synthetic TRNC (Reverse Transfer)",
            "F1":        round(float(r.get("F1",    0)), 4),
            "AUROC":     round(float(r.get("AUROC", 0)), 4),
            "Precision": round(float(r.get("Precision", 0)), 4),
            "Recall":    round(float(r.get("Recall", 0)), 4),
            "Brier":     round(float(r.get("Brier", 0)), 4),
            "Source":    "Phase 1 — Real Data",
        })
    else:
        rows.append({
            "Experiment": "Real SGCC → Synthetic TRNC (Reverse Transfer)",
            "F1": "SKIPPED", "AUROC": "SKIPPED",
            "Precision": "SKIPPED", "Recall": "SKIPPED", "Brier": "SKIPPED",
            "Source": "Skipped (no TRNC holdout .pt found)",
        })

    # Experiment 4 (TDD2022 cross-domain)
    if exp4_df is not None and not exp4_df.empty:
        r = exp4_df.iloc[0]
        rows.append({
            "Experiment": "Real SGCC → TDD2022 (Cross-Domain Zero-Shot)",
            "F1":        round(float(r.get("F1",    0)), 4),
            "AUROC":     round(float(r.get("AUROC", 0)), 4),
            "Precision": round(float(r.get("Precision", 0)), 4),
            "Recall":    round(float(r.get("Recall", 0)), 4),
            "Brier":     round(float(r.get("Brier", 0)), 4),
            "Source":    "Phase 1 — Real Data",
        })
    else:
        rows.append({
            "Experiment": "Real SGCC → TDD2022 (Cross-Domain Zero-Shot)",
            "F1": "SKIPPED", "AUROC": "SKIPPED",
            "Precision": "SKIPPED", "Recall": "SKIPPED", "Brier": "SKIPPED",
            "Source": "Skipped (no TDD2022 CSV found)",
        })

    table = pd.DataFrame(rows, columns=[
        "Experiment", "F1", "AUROC", "Precision", "Recall", "Brier", "Source"
    ])
    return table


def print_final_table(table: pd.DataFrame):
    print("\n" + "=" * 105)
    print("  GRIDGUARD AI — PHASE 1 FINAL RESULTS TABLE")
    print("=" * 105)
    header = (
        f"{'Experiment':<52} {'F1':>7} {'AUROC':>7} "
        f"{'Prec':>7} {'Rec':>7} {'Brier':>7}"
    )
    print(header)
    print("-" * 105)
    for _, row in table.iterrows():
        name  = str(row["Experiment"])[:51]
        f1    = str(row["F1"])[:7]
        auroc = str(row["AUROC"])[:7]
        prec  = str(row["Precision"])[:7]
        rec   = str(row["Recall"])[:7]
        brier = str(row["Brier"])[:7]
        print(f"{name:<52} {f1:>7} {auroc:>7} {prec:>7} {rec:>7} {brier:>7}")
    print("=" * 105)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GridGuard AI Phase 1 — Real-Data Training Pipeline"
    )
    ap.add_argument("--sgcc_path",  required=True,
                    help="Directory containing the SGCC CSV file")
    ap.add_argument("--tdd_path",   default=None,
                    help="Directory containing TDD2022 CSV (Experiment 4)")
    ap.add_argument("--trnc_test",  default=None,
                    help="Path to synthetic TRNC test .pt file (Experiment 3)")
    ap.add_argument("--output_dir", default=".",
                    help="Project root (models/ results/ created here)")
    ap.add_argument("--skip_exp1",  action="store_true")
    ap.add_argument("--skip_exp2",  action="store_true")
    ap.add_argument("--skip_exp3",  action="store_true")
    ap.add_argument("--skip_exp4",  action="store_true")
    args = ap.parse_args()

    set_global_seed(SEED)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(os.path.join(output_dir, "results"),    exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"),     exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'#'*60}")
    print(f"  GridGuard AI — Phase 1 Real-Data Training")
    print(f"  Device : {device}")
    print(f"  Seed   : {SEED}")
    print(f"  Output : {output_dir}")
    print(f"{'#'*60}\n")

    t0 = time.time()

    # ── Load & preprocess SGCC (shared across Exp 1 & 2) ─────────────────────
    sgcc_cache = os.path.join(args.sgcc_path, "sgcc_processed.pt")
    X_sgcc, y_sgcc, meta_sgcc = run_sgcc_pipeline(
        args.sgcc_path, cache_path=sgcc_cache
    )
    print(f"SGCC ready: {X_sgcc.shape}  theft={y_sgcc.mean():.3%}\n")

    # ── Experiment 1: Standard CV ─────────────────────────────────────────────
    exp1_df = None
    if not args.skip_exp1:
        exp1_df = run_standard_cv(X_sgcc, y_sgcc, output_dir=output_dir)
    else:
        csv1 = os.path.join(output_dir, "results", "exp1_standard_cv.csv")
        if os.path.isfile(csv1):
            exp1_df = pd.read_csv(csv1)
            print(f"[Exp 1 SKIPPED] Loaded existing results from {csv1}")
        else:
            print("[Exp 1 SKIPPED] No existing results found.")

    # ── Experiment 2: Walk-Forward ────────────────────────────────────────────
    exp2_df = None
    if not args.skip_exp2:
        exp2_df = run_walk_forward(X_sgcc, y_sgcc, meta_sgcc, output_dir=output_dir)
    else:
        csv2 = os.path.join(output_dir, "results", "exp2_walkforward.csv")
        if os.path.isfile(csv2):
            exp2_df = pd.read_csv(csv2)
            print(f"[Exp 2 SKIPPED] Loaded existing results from {csv2}")
        else:
            print("[Exp 2 SKIPPED] No existing results found.")

    # ── Resolve in-domain SGCC metrics for degradation reporting ─────────────
    sgcc_f1, sgcc_auroc = load_sgcc_indomain_metrics(
        os.path.join(output_dir, "results")
    )

    weights_path = os.path.join(output_dir, "models", "gridguard_sgcc_best.pth")
    xgb_path     = os.path.join(output_dir, "models", "xgboost_sgcc_edge.pkl")
    trnc_path    = args.trnc_test or os.path.join(
        output_dir, "data", "trnc_synthetic_test.pt"
    )
    tdd_dir      = args.tdd_path or os.path.join(output_dir, "data", "tdd2022")

    # ── Experiment 3: Reverse Transfer ────────────────────────────────────────
    exp3_df = None
    if not args.skip_exp3:
        exp3_df = run_exp3_reverse_transfer(
            weights_path, xgb_path, trnc_path, output_dir,
            sgcc_f1=sgcc_f1, sgcc_auroc=sgcc_auroc,
        )

    # ── Experiment 4: TDD2022 Cross-Domain ───────────────────────────────────
    exp4_df = None
    if not args.skip_exp4:
        exp4_df = run_exp4_cross_domain_tdd(
            weights_path, xgb_path, tdd_dir, output_dir,
            sgcc_f1=sgcc_f1, sgcc_auroc=sgcc_auroc,
        )

    # ── Final Table ───────────────────────────────────────────────────────────
    final_table = build_final_table(output_dir, exp1_df, exp2_df, exp3_df, exp4_df)
    print_final_table(final_table)

    csv_out = os.path.join(output_dir, "results", "sgcc_real_training_results.csv")
    final_table.to_csv(csv_out, index=False)
    print(f"\n  ✓  Final results saved → {csv_out}")

    elapsed = time.time() - t0
    h, rem  = divmod(int(elapsed), 3600)
    m, s    = divmod(rem, 60)
    print(f"  ✓  Total wall-clock time: {h:02d}h {m:02d}m {s:02d}s\n")


if __name__ == "__main__":
    main()
