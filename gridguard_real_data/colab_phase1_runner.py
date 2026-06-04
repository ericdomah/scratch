# GridGuard AI — Phase 1 Real-Data Training
# Google Colab Notebook
# Run cell by cell. Requires: SGCC data.csv uploaded to /content/scratch/gridguard_real_data/data/sgcc/

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Mount Drive and clone repo
# ─────────────────────────────────────────────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import subprocess, os

# Clone the repo if not already present
if not os.path.exists('/content/scratch'):
    subprocess.run(['git', 'clone',
                    'https://github.com/ericdomah/scratch.git',
                    '/content/scratch'], check=True)
else:
    subprocess.run(['git', '-C', '/content/scratch', 'pull'], check=True)

os.chdir('/content/scratch/gridguard_real_data')
print("Working directory:", os.getcwd())

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Install dependencies
# ─────────────────────────────────────────────────────────────────────────────
subprocess.run(['pip', 'install', '-q',
                'xgboost', 'tqdm', 'scipy', 'scikit-learn'], check=True)

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Upload SGCC data.csv (if not on Drive already)
# ─────────────────────────────────────────────────────────────────────────────
# OPTION A: Upload from laptop (run this cell)
# from google.colab import files
# import shutil, os
# os.makedirs('data/sgcc', exist_ok=True)
# print("Please click 'Choose Files' and select data.csv from your laptop:")
# uploaded = files.upload()
# for fname in uploaded:
#     shutil.move(fname, f'data/sgcc/{fname}')
#     print(f"Moved {fname} -> data/sgcc/{fname}")

# OPTION B: Copy from Google Drive (uncomment and adjust path)
# import shutil
# shutil.copy('/content/drive/MyDrive/GridGuard/data.csv', 'data/sgcc/data.csv')
# print("Copied from Drive.")

# Verify it exists:
sgcc_csv = 'data/sgcc/data.csv'
assert os.path.exists(sgcc_csv), f"SGCC CSV not found at {sgcc_csv}. Run OPTION A or B above."
print(f"[OK] SGCC CSV found: {os.path.getsize(sgcc_csv) / 1e6:.1f} MB")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Preprocess SGCC (one window per consumer)
# ─────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, '.')

from preprocessing.sgcc_pipeline import run_sgcc_pipeline

CACHE_PATH = 'data/sgcc/sgcc_processed.pt'

# Delete stale cache if present from a previous (buggy) run
if os.path.exists(CACHE_PATH):
    os.remove(CACHE_PATH)
    print("Deleted stale cache.")

X_sgcc, y_sgcc, meta_sgcc = run_sgcc_pipeline('data/sgcc', cache_path=CACHE_PATH)

# ── CRITICAL VERIFICATION ────────────────────────────────────────────────────
n = X_sgcc.shape[0]
prevalence = y_sgcc.mean().item()

print(f"\n{'='*55}")
print(f"  VERIFICATION CHECK")
print(f"{'='*55}")
print(f"  Samples  : {n:,}  (expected 40,000–42,000)")
print(f"  Shape    : {X_sgcc.shape}")
print(f"  Theft    : {prevalence:.3%}  (expected ~5%)")
print(f"{'='*55}")

assert n < 500_000, f"STOP: {n:,} samples — sliding window bug detected!"
assert 0.01 <= prevalence <= 0.25, f"STOP: prevalence {prevalence:.3%} out of range!"
print("\n[OK] All assertions passed — pipeline is correct.\n")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — EXPERIMENT 1: Standard 5-Fold StratifiedKFold CV
# ─────────────────────────────────────────────────────────────────────────────
from training.train_sgcc import run_standard_cv

exp1_df = run_standard_cv(X_sgcc, y_sgcc, output_dir='.')
print(exp1_df[['Fold','Fused_F1','Fused_AUROC','Fused_Precision','Fused_Recall']].to_string())

# Save to Drive for safekeeping
import shutil
os.makedirs('/content/drive/MyDrive/GridGuard_Results/results', exist_ok=True)
os.makedirs('/content/drive/MyDrive/GridGuard_Results/models', exist_ok=True)
shutil.copy('results/exp1_standard_cv.csv',
            '/content/drive/MyDrive/GridGuard_Results/results/exp1_standard_cv.csv')
shutil.copy('models/gridguard_sgcc_best.pth',
            '/content/drive/MyDrive/GridGuard_Results/models/gridguard_sgcc_best.pth')
shutil.copy('models/xgboost_sgcc_edge.pkl',
            '/content/drive/MyDrive/GridGuard_Results/models/xgboost_sgcc_edge.pkl')
print("[OK] Exp1 results and best model saved to Drive.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — EXPERIMENT 2: Walk-Forward Temporal Validation (Primary Protocol)
# ─────────────────────────────────────────────────────────────────────────────
from training.train_walkforward import run_walk_forward

exp2_df = run_walk_forward(X_sgcc, y_sgcc, meta_sgcc, output_dir='.')
print(exp2_df[['Round','Train_samples','Test_samples',
               'GG_F1','GG_AUROC','Base_F1','Base_AUROC']].to_string())

shutil.copy('results/exp2_walkforward.csv',
            '/content/drive/MyDrive/GridGuard_Results/results/exp2_walkforward.csv')
print("[OK] Exp2 results saved to Drive.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — EXPERIMENT 3: Reverse Transfer (SGCC → Synthetic TRNC)
# ─────────────────────────────────────────────────────────────────────────────
from evaluation.evaluate_cross_domain import run_exp3_reverse_transfer

TRNC_PATH    = 'data/trnc_synthetic_test.pt'
WEIGHTS_PATH = 'models/gridguard_sgcc_best.pth'
XGB_PATH     = 'models/xgboost_sgcc_edge.pkl'

# Load SGCC in-domain F1 from Exp1 summary row
import pandas as pd, numpy as np
exp1_summary = pd.read_csv('results/exp1_standard_cv.csv')
sgcc_f1_str = exp1_summary[exp1_summary['Fold'] == 'mean +/- SD']['Fused_F1'].values[0]
sgcc_f1 = float(sgcc_f1_str.split('+/-')[0].strip())
sgcc_auroc_str = exp1_summary[exp1_summary['Fold'] == 'mean +/- SD']['Fused_AUROC'].values[0]
sgcc_auroc = float(sgcc_auroc_str.split('+/-')[0].strip())

if os.path.exists(TRNC_PATH):
    exp3_df = run_exp3_reverse_transfer(
        WEIGHTS_PATH, XGB_PATH, TRNC_PATH, '.',
        sgcc_f1=sgcc_f1, sgcc_auroc=sgcc_auroc,
    )
    shutil.copy('results/exp3_reverse_transfer.csv',
                '/content/drive/MyDrive/GridGuard_Results/results/exp3_reverse_transfer.csv')
    print("[OK] Exp3 done.")
else:
    print(f"[SKIPPED] Exp3: TRNC test file not found at {TRNC_PATH}")
    print("  Upload trnc_synthetic_test.pt to data/ to run this experiment.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — EXPERIMENT 4: Cross-Domain (SGCC → TDD2022)
# ─────────────────────────────────────────────────────────────────────────────
from evaluation.evaluate_cross_domain import run_exp4_cross_domain_tdd

TDD_PATH = 'data/tdd2022'

exp4_df = run_exp4_cross_domain_tdd(
    WEIGHTS_PATH, XGB_PATH, TDD_PATH, '.',
    sgcc_f1=sgcc_f1, sgcc_auroc=sgcc_auroc,
)
print(exp4_df.to_string())

shutil.copy('results/exp4_cross_domain_tdd.csv',
            '/content/drive/MyDrive/GridGuard_Results/results/exp4_cross_domain_tdd.csv')
print("[OK] Exp4 results saved to Drive.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Build Final Comparison Table
# ─────────────────────────────────────────────────────────────────────────────
from run_all import build_final_table, print_final_table

final_table = build_final_table(
    output_dir='.',
    exp1_df=exp1_df,
    exp2_df=exp2_df,
    exp3_df=None,   # replace with exp3_df if Exp3 ran
    exp4_df=exp4_df,
)
print_final_table(final_table)
final_table.to_csv('results/sgcc_real_training_results.csv', index=False)
shutil.copy('results/sgcc_real_training_results.csv',
            '/content/drive/MyDrive/GridGuard_Results/results/sgcc_real_training_results.csv')
print("\n[OK] Final results table saved to Drive.")
print("     -> /content/drive/MyDrive/GridGuard_Results/results/sgcc_real_training_results.csv")
