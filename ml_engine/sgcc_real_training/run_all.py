"""
GridGuard AI — SGCC Real-Data Training Pipeline
Master script — runs all four experiments.

Usage:
    python run_all.py \
        --sgcc_path  ../../data/sgcc/data.csv \
        --tdd_path   ../../data/tdd2022/ \
        --epochs     30 \
        --folds      10 \
        --seed       42 \
        --output     results/
"""
from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def print_final_table(output_dir: str) -> None:
    """Read all result CSVs and print a consolidated comparison table."""
    import pandas as pd

    exp_files = {
        "Exp1 — Standard 10-Fold CV":     os.path.join(output_dir, "exp1_standard_cv.csv"),
        "Exp2 — Walk-Forward":             os.path.join(output_dir, "exp2_walkforward.csv"),
        "Exp3 — SGCC->TRNC (zero-shot)":  os.path.join(output_dir, "exp3_reverse_transfer.csv"),
        "Exp4 — SGCC->TDD2022 (zero-shot)": os.path.join(output_dir, "exp4_cross_domain_tdd.csv"),
    }

    print("\n" + "=" * 70)
    print("  GRIDGUARD AI — REAL-DATA TRAINING RESULTS SUMMARY")
    print("=" * 70)

    for label, path in exp_files.items():
        print(f"\n--- {label} ---")
        if not os.path.exists(path):
            print(f"  [NOT RUN / FILE MISSING]: {path}")
            continue
        df = pd.read_csv(path)
        # Show the summary row if it exists
        summary_mask = df.apply(
            lambda r: any(str(v).startswith("mean") for v in r.values), axis=1
        )
        if summary_mask.any():
            print(df[summary_mask].to_string(index=False))
        else:
            print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("  Pre-computed Synthetic TRNC Baseline (from earlier runs):")
    print("  GridGuard MetaEnsemble   F1=0.9234  AUROC=0.9791  (synthetic data)")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GridGuard SGCC Real-Data Training Pipeline"
    )
    parser.add_argument("--sgcc_path", required=True,
                        help="Path to SGCC data.csv")
    parser.add_argument("--tdd_path",  default=None,
                        help="Path to TDD2022 df.csv or directory (optional)")
    parser.add_argument("--trnc_path", default=None,
                        help="Path to synthetic TRNC .pt test file (optional)")
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--folds",     type=int, default=10)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default="results/")
    args = parser.parse_args()

    # Seed FIRST — before any imports that use randomness
    set_seed(args.seed)
    print(f"[Main] Seed={args.seed}  epochs={args.epochs}  folds={args.folds}")
    print(f"[Main] Output dir: {args.output}")
    print(f"[Main] Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    os.makedirs(args.output,  exist_ok=True)
    os.makedirs("models/",    exist_ok=True)

    # Lazy imports so seed is set before torch initialises workers
    from sgcc_pipeline       import load_sgcc
    from tdd2022_pipeline    import load_tdd2022
    from train_sgcc_cv       import run_standard_cv
    from train_walkforward   import run_walkforward
    from evaluate_transfer   import run_transfer_experiments

    # ------------------------------------------------------------------
    # Load SGCC
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Loading SGCC real-data ...")
    print("=" * 55)
    X_sgcc, y_sgcc = load_sgcc(args.sgcc_path)

    # ------------------------------------------------------------------
    # Experiment 1 — Standard 10-Fold CV
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Experiment 1 — Standard 10-Fold Stratified CV")
    print("=" * 55)
    mean_tau, best_model_path, best_xgb_path = run_standard_cv(
        X_sgcc, y_sgcc,
        output_dir=args.output,
        n_folds=args.folds,
        epochs=args.epochs,
    )

    # ------------------------------------------------------------------
    # Experiment 2 — Walk-Forward Temporal Validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Experiment 2 — Walk-Forward Temporal Validation")
    print("=" * 55)
    run_walkforward(
        X_sgcc, y_sgcc,
        output_dir=args.output,
        epochs=args.epochs,
    )

    # ------------------------------------------------------------------
    # Load optional cross-domain datasets
    # ------------------------------------------------------------------
    tdd_data  = None
    trnc_data = None

    if args.tdd_path:
        print("\n" + "=" * 55)
        print("  Loading TDD2022 ...")
        print("=" * 55)
        tdd_data = load_tdd2022(args.tdd_path)

    if args.trnc_path:
        if os.path.exists(args.trnc_path):
            print(f"\n[Main] Loading synthetic TRNC test partition: {args.trnc_path}")
            ckpt = torch.load(args.trnc_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict):
                X_trnc = ckpt["X"].numpy() if hasattr(ckpt["X"], "numpy") else ckpt["X"]
                y_trnc = ckpt["y"].numpy() if hasattr(ckpt["y"], "numpy") else ckpt["y"]
            else:
                raise ValueError(f"Unexpected TRNC file format: {type(ckpt)}")
            trnc_data = (X_trnc.astype(np.float32), y_trnc.astype(np.float32))
        else:
            print(f"[Main] TRNC path not found: {args.trnc_path} — Exp3 will be skipped.")

    # ------------------------------------------------------------------
    # Experiments 3 & 4 — Transfer
    # ------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  Experiments 3 & 4 — Cross-Domain Transfer")
    print("=" * 55)
    run_transfer_experiments(
        output_dir=args.output,
        mean_tau=mean_tau,
        best_fold_model_path=best_model_path,
        best_fold_xgb_path=best_xgb_path,
        tdd_data=tdd_data,
        trnc_data=trnc_data,
    )

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    print_final_table(args.output)
    print("\n[Main] All experiments complete.")


if __name__ == "__main__":
    main()
