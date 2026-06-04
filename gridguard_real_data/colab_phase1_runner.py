# ╔══════════════════════════════════════════════════════════════════╗
# ║   GridGuard AI — Phase 1 Real-Data Training                     ║
# ║   SGCC Dataset: One Window Per Consumer (Thesis-Compliant)       ║
# ║   Copy each cell into a Colab notebook and run top-to-bottom.   ║
# ╚══════════════════════════════════════════════════════════════════╝

# ──────────────────────────────────────────────────────────────────────────────
# CELL 1 ── Runtime check + Mount Google Drive
# ──────────────────────────────────────────────────────────────────────────────
# Before running: Runtime → Change runtime type → GPU (T4 or A100)

import torch, subprocess, os, shutil

print(f"PyTorch  : {torch.__version__}")
print(f"GPU      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT FOUND — change runtime to GPU!'}")
assert torch.cuda.is_available(), "Please enable GPU before continuing."

from google.colab import drive
drive.mount('/content/drive')
print("Drive mounted.")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 2 ── Clone / pull the GridGuard repo
# ──────────────────────────────────────────────────────────────────────────────
REPO_URL = "https://github.com/ericdomah/scratch.git"
REPO_DIR = "/content/scratch"

if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print("Repo cloned.")
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
    print("Repo updated.")

# Set working directory to the Phase 1 module
PHASE1_DIR = f"{REPO_DIR}/gridguard_real_data"
os.chdir(PHASE1_DIR)
print(f"Working directory: {os.getcwd()}")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 3 ── Install dependencies
# ──────────────────────────────────────────────────────────────────────────────
subprocess.run([
    "pip", "install", "-q",
    "xgboost>=2.0",
    "tqdm",
    "scipy",
    "scikit-learn",
], check=True)
print("Dependencies installed.")

import sys
sys.path.insert(0, PHASE1_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# CELL 4 ── Upload SGCC data.csv
#
# The SGCC CSV is too large for GitHub (167 MB).
# Choose ONE of the options below:
# ──────────────────────────────────────────────────────────────────────────────

# ── OPTION A: Upload directly from your laptop ─────────────────────────
from google.colab import files as colab_files

SGCC_DIR = f"{PHASE1_DIR}/data/sgcc"
os.makedirs(SGCC_DIR, exist_ok=True)

print("Click 'Choose Files' and select data.csv from your laptop:")
uploaded = colab_files.upload()
for fname, content in uploaded.items():
    dest = f"{SGCC_DIR}/{fname}"
    with open(dest, "wb") as f:
        f.write(content)
    print(f"  Saved: {dest}  ({len(content)/1e6:.1f} MB)")

# ── OPTION B: Copy from Google Drive (faster if already uploaded) ──────
# Uncomment and adjust the path if you stored data.csv in Drive:
#
# DRIVE_PATH = "/content/drive/MyDrive/GridGuard/data.csv"
# dest       = f"{SGCC_DIR}/data.csv"
# shutil.copy(DRIVE_PATH, dest)
# print(f"Copied from Drive -> {dest}")

# Verify
assert os.path.exists(f"{SGCC_DIR}/data.csv"), \
    "data.csv not found! Run Option A or B above."
size_mb = os.path.getsize(f"{SGCC_DIR}/data.csv") / 1e6
print(f"\n[OK] data.csv found — {size_mb:.1f} MB")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 5 ── Preprocess SGCC  (one window per consumer — thesis rule)
# ──────────────────────────────────────────────────────────────────────────────
from preprocessing.sgcc_pipeline import run_sgcc_pipeline

# Delete any stale cached tensor from a previous (potentially buggy) run
CACHE_PATH = f"{SGCC_DIR}/sgcc_processed.pt"
if os.path.exists(CACHE_PATH):
    os.remove(CACHE_PATH)
    print("Deleted stale cache — will reprocess from raw CSV.")

X_sgcc, y_sgcc, meta_sgcc = run_sgcc_pipeline(
    SGCC_DIR,
    cache_path=CACHE_PATH,
    verbose=True,
)

# ── CRITICAL VERIFICATION ─────────────────────────────────────────────
n          = X_sgcc.shape[0]
prevalence = float(y_sgcc.mean())

print(f"\n{'='*55}")
print(f"  VERIFICATION (must pass before training!)")
print(f"{'='*55}")
print(f"  Samples (one per consumer) : {n:,}  (expect 40k-42k)")
print(f"  Tensor shape               : {X_sgcc.shape}")
print(f"  Theft prevalence           : {prevalence:.3%}  (expect ~5%)")
print(f"{'='*55}")

assert n < 500_000, (
    f"STOP: {n:,} samples — sliding window bug detected! "
    "Expected ~40,000-42,000."
)
assert 0.01 <= prevalence <= 0.25, (
    f"STOP: prevalence {prevalence:.3%} is wrong. "
    "Expected ~5%. Check the data."
)
print("\n[OK] All checks passed — pipeline is correct.\n")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 6 ── EXPERIMENT 1: Standard 5-Fold StratifiedKFold CV
#
# Trains GridGuardUniversalHybrid + XGBoost on all 5 folds.
# Saves best model weights after each fold.
# Expected runtime: ~60-90 min on T4, ~30-45 min on A100.
# ──────────────────────────────────────────────────────────────────────────────
from training.train_sgcc import run_standard_cv

exp1_df = run_standard_cv(X_sgcc, y_sgcc, output_dir=PHASE1_DIR)

# Print summary
import pandas as pd
summary = exp1_df[exp1_df["Fold"] == "mean +/- SD"]
print("\n── Experiment 1 Summary ──────────────────────────")
print(summary[["Fused_F1","Fused_AUROC","Fused_Precision","Fused_Recall","Fused_Brier"]].to_string(index=False))

# Save to Google Drive immediately
DRIVE_RESULTS = "/content/drive/MyDrive/GridGuard_Results"
os.makedirs(f"{DRIVE_RESULTS}/results", exist_ok=True)
os.makedirs(f"{DRIVE_RESULTS}/models",  exist_ok=True)

shutil.copy(f"{PHASE1_DIR}/results/exp1_standard_cv.csv",
            f"{DRIVE_RESULTS}/results/exp1_standard_cv.csv")
shutil.copy(f"{PHASE1_DIR}/models/gridguard_sgcc_best.pth",
            f"{DRIVE_RESULTS}/models/gridguard_sgcc_best.pth")
shutil.copy(f"{PHASE1_DIR}/models/xgboost_sgcc_edge.pkl",
            f"{DRIVE_RESULTS}/models/xgboost_sgcc_edge.pkl")
print("\n[OK] Exp1 results + best model saved to Drive.")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 7 ── EXPERIMENT 2: Walk-Forward Temporal Validation (PRIMARY)
#
# 5 expanding rounds sorted by CONS_NO (temporal proxy).
# Trains BOTH GridGuard AND BiGRU-BiLSTM baseline from scratch each round.
# Expected runtime: ~2-3 hours on T4, ~1 hour on A100.
# ──────────────────────────────────────────────────────────────────────────────
from training.train_walkforward import run_walk_forward

exp2_df = run_walk_forward(X_sgcc, y_sgcc, meta_sgcc, output_dir=PHASE1_DIR)

# Print summary
summary2 = exp2_df[exp2_df["Round"] == "mean +/- SD"]
print("\n── Experiment 2 Summary ──────────────────────────")
print(summary2[["GG_F1","GG_AUROC","Base_F1","Base_AUROC","Cohens_d"]].to_string(index=False))

shutil.copy(f"{PHASE1_DIR}/results/exp2_walkforward.csv",
            f"{DRIVE_RESULTS}/results/exp2_walkforward.csv")
print("[OK] Exp2 results saved to Drive.")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 8 ── EXPERIMENT 3: Reverse Transfer (SGCC-trained → Synthetic TRNC)
#
# Zero-shot only — NO fine-tuning.
# Skip if you do not have the synthetic TRNC test partition.
# ──────────────────────────────────────────────────────────────────────────────
from evaluation.evaluate_cross_domain import (
    run_exp3_reverse_transfer, load_sgcc_indomain_metrics
)

WEIGHTS_PATH = f"{PHASE1_DIR}/models/gridguard_sgcc_best.pth"
XGB_PATH     = f"{PHASE1_DIR}/models/xgboost_sgcc_edge.pkl"
TRNC_PATH    = f"{PHASE1_DIR}/data/trnc_synthetic_test.pt"

sgcc_f1, sgcc_auroc = load_sgcc_indomain_metrics(f"{PHASE1_DIR}/results")

if os.path.exists(TRNC_PATH):
    exp3_df = run_exp3_reverse_transfer(
        WEIGHTS_PATH, XGB_PATH, TRNC_PATH, PHASE1_DIR,
        sgcc_f1=sgcc_f1, sgcc_auroc=sgcc_auroc,
    )
    shutil.copy(f"{PHASE1_DIR}/results/exp3_reverse_transfer.csv",
                f"{DRIVE_RESULTS}/results/exp3_reverse_transfer.csv")
    print("[OK] Exp3 results saved to Drive.")
    exp3_result = exp3_df
else:
    print(f"[SKIPPED] Exp3: {TRNC_PATH} not found.")
    print("  To run: upload trnc_synthetic_test.pt to data/ and re-run this cell.")
    exp3_result = None

# ──────────────────────────────────────────────────────────────────────────────
# CELL 9 ── EXPERIMENT 4: Cross-Domain (SGCC-trained → TDD2022)
#
# TDD2022 df.csv must be present in data/tdd2022/
# It is already on your machine — upload it the same way as SGCC.
# Zero-shot only — NO fine-tuning.
# ──────────────────────────────────────────────────────────────────────────────
from evaluation.evaluate_cross_domain import run_exp4_cross_domain_tdd

TDD_DIR = f"{PHASE1_DIR}/data/tdd2022"
os.makedirs(TDD_DIR, exist_ok=True)

# If TDD2022 df.csv not uploaded yet, upload it here:
if not os.path.exists(f"{TDD_DIR}/df.csv"):
    print("TDD2022 df.csv not found. Uploading now...")
    uploaded_tdd = colab_files.upload()
    for fname, content in uploaded_tdd.items():
        dest = f"{TDD_DIR}/{fname}"
        with open(dest, "wb") as f:
            f.write(content)
        print(f"  Saved: {dest}  ({len(content)/1e6:.1f} MB)")

exp4_df = run_exp4_cross_domain_tdd(
    WEIGHTS_PATH, XGB_PATH, TDD_DIR, PHASE1_DIR,
    sgcc_f1=sgcc_f1, sgcc_auroc=sgcc_auroc,
)
print(exp4_df.to_string(index=False))

shutil.copy(f"{PHASE1_DIR}/results/exp4_cross_domain_tdd.csv",
            f"{DRIVE_RESULTS}/results/exp4_cross_domain_tdd.csv")
print("[OK] Exp4 results saved to Drive.")

# ──────────────────────────────────────────────────────────────────────────────
# CELL 10 ── Build final comparison table and print
# ──────────────────────────────────────────────────────────────────────────────
from run_all import build_final_table, print_final_table

final_table = build_final_table(
    output_dir=PHASE1_DIR,
    exp1_df=exp1_df,
    exp2_df=exp2_df,
    exp3_df=exp3_result,
    exp4_df=exp4_df,
)
print_final_table(final_table)

final_table.to_csv(f"{PHASE1_DIR}/results/sgcc_real_training_results.csv", index=False)
shutil.copy(
    f"{PHASE1_DIR}/results/sgcc_real_training_results.csv",
    f"{DRIVE_RESULTS}/results/sgcc_real_training_results.csv",
)
print(f"\n[OK] Final table saved to Drive:")
print(f"     {DRIVE_RESULTS}/results/sgcc_real_training_results.csv")
print("\nPhase 1 complete!")
