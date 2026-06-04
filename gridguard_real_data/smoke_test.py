"""
smoke_test.py -- Validates the full GridGuard Phase 1 pipeline without real data.
Creates synthetic SGCC-shaped and TDD2022-shaped CSVs, runs preprocessing,
runs 2-fold CV (abbreviated), runs 2-round walk-forward, runs cross-domain eval.
All on CPU, ~2 min runtime.
"""
import sys, os, random, warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print("="*60)
print("  GridGuard Phase 1 -- Smoke Test (synthetic mini-data)")
print("="*60)

# -- 1. Synthetic SGCC CSV ----------------------------------------------------
print("\n[1/7] Building synthetic SGCC CSV ...")
N_CONS   = 120          # consumers (real: ~42k)
N_DAYS   = 7 * 52      # 52 weeks = 364 days (real: ~1035)
np.random.seed(SEED)

date_cols = pd.date_range("2014-01-01", periods=N_DAYS, freq="D").strftime("%Y-%m-%d").tolist()
data      = np.random.rand(N_CONS, N_DAYS) * 10
# inject some NaN (~10% of entries)
nan_mask  = np.random.rand(N_CONS, N_DAYS) < 0.10
data[nan_mask] = np.nan
# labels: ~5% theft
labels = (np.random.rand(N_CONS) < 0.05).astype(int)

df_sgcc = pd.DataFrame(data, columns=date_cols)
df_sgcc.insert(0, "CONS_NO", [f"C{i:04d}" for i in range(N_CONS)])
df_sgcc.insert(1, "FLAG",    labels)

sgcc_dir = os.path.join(ROOT, "data", "sgcc")
os.makedirs(sgcc_dir, exist_ok=True)
sgcc_csv = os.path.join(sgcc_dir, "data.csv")
df_sgcc.to_csv(sgcc_csv, index=False)
print(f"   Synthetic SGCC: {df_sgcc.shape} -> {sgcc_csv}")

# -- 2. Synthetic TDD2022 CSV (wide format) ------------------------------------
print("\n[2/7] Building synthetic TDD2022 CSV ...")
N_CONS_TDD   = 40
N_HOURS_TDD  = 24 * 7 * 30   # 30 weeks of hourly data
data_tdd = np.random.rand(N_HOURS_TDD, N_CONS_TDD) * 5

col_names = []
for i in range(N_CONS_TDD):
    if i < 4:
        col_names.append(f"consumer_{i}_theft_type{(i%3)+1}")
    else:
        col_names.append(f"consumer_{i}_normal")

df_tdd = pd.DataFrame(data_tdd, columns=col_names)
tdd_dir = os.path.join(ROOT, "data", "tdd2022")
os.makedirs(tdd_dir, exist_ok=True)
tdd_csv = os.path.join(tdd_dir, "tdd2022_data.csv")
df_tdd.to_csv(tdd_csv, index=False)
print(f"   Synthetic TDD2022: {df_tdd.shape} -> {tdd_csv}")

# -- 3. SGCC preprocessing -----------------------------------------------------
print("\n[3/7] Running SGCC preprocessing ...")
from preprocessing.sgcc_pipeline import run_sgcc_pipeline, compute_tabular_features

sgcc_cache = os.path.join(sgcc_dir, "sgcc_processed.pt")
if os.path.exists(sgcc_cache):
    os.remove(sgcc_cache)   # force fresh run

X_sgcc, y_sgcc, meta = run_sgcc_pipeline(sgcc_dir)
feats = compute_tabular_features(X_sgcc.numpy())
print(f"   X_sgcc  : {X_sgcc.shape}   dtype={X_sgcc.dtype}")
print(f"   y_sgcc  : {y_sgcc.shape}   theft={y_sgcc.mean():.3%}")
print(f"   features: {feats.shape}")
assert X_sgcc.shape[1] == 26, f"Expected T=26, got {X_sgcc.shape[1]}"
assert X_sgcc.shape[2] == 2,  f"Expected C=2, got {X_sgcc.shape[2]}"
print("   [OK] shape assertions passed")

# -- 4. TDD2022 preprocessing --------------------------------------------------
print("\n[4/7] Running TDD2022 preprocessing ...")
from preprocessing.tdd2022_pipeline import run_tdd2022_pipeline

tdd_cache = os.path.join(tdd_dir, "tdd2022_processed.pt")
if os.path.exists(tdd_cache):
    os.remove(tdd_cache)

X_tdd, y_tdd, meta_tdd = run_tdd2022_pipeline(tdd_dir)
print(f"   X_tdd   : {X_tdd.shape}   dtype={X_tdd.dtype}")
print(f"   y_tdd   : {y_tdd.shape}   theft={y_tdd.mean():.3%}")
assert X_tdd.shape[1] == 26, f"Expected T=26, got {X_tdd.shape[1]}"
assert X_tdd.shape[2] == 2,  f"Expected C=2, got {X_tdd.shape[2]}"
print("   [OK] shape assertions passed")

# -- 5. Model forward pass -----------------------------------------------------
print("\n[5/7] Model forward pass ...")
from models.gridguard_model import (
    GridGuardUniversalHybrid, BiGRUBiLSTMBaseline,
    AsymmetricFocalLoss, build_model
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")

for name, m in [("GridGuard", GridGuardUniversalHybrid()),
                ("BiGRUBiLSTM", BiGRUBiLSTMBaseline())]:
    m = m.to(device)
    x = torch.randn(8, 26, 2).to(device)
    out = m(x)
    assert out.shape == (8, 1), f"{name} bad output shape: {out.shape}"
    print(f"   [OK] {name}: input (8,26,2) -> output {tuple(out.shape)}")

loss_fn = AsymmetricFocalLoss()
preds   = torch.sigmoid(torch.randn(8, 1))
targets = torch.randint(0, 2, (8,)).float()
loss    = loss_fn(preds, targets)
assert loss.item() > 0
print(f"   [OK] AsymmetricFocalLoss: {loss.item():.4f}")

# -- 6. Abbreviated Standard CV (2 folds, 2 epochs) ---------------------------
print("\n[6/7] Abbreviated 2-fold CV (2 epochs each) ...")
import importlib, types

# Monkey-patch constants for speed
import training.train_sgcc as tsgcc
_orig_epochs    = tsgcc.EPOCHS
_orig_n_folds   = tsgcc.N_FOLDS
_orig_ckpt_ev   = tsgcc.CHECKPOINT_EVERY
tsgcc.EPOCHS          = 2
tsgcc.N_FOLDS         = 2
tsgcc.CHECKPOINT_EVERY = 10   # disable checkpointing in smoke test

results_dir = os.path.join(ROOT, "results")
models_dir  = os.path.join(ROOT, "models")
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

exp1_df = tsgcc.run_standard_cv(X_sgcc, y_sgcc, output_dir=ROOT, n_folds=2)
print(f"   Exp1 result rows: {len(exp1_df)}")
assert os.path.isfile(os.path.join(results_dir, "exp1_standard_cv.csv"))
assert os.path.isfile(os.path.join(models_dir,  "gridguard_sgcc_best.pth"))
assert os.path.isfile(os.path.join(models_dir,  "xgboost_sgcc_edge.pkl"))
print("   [OK] exp1_standard_cv.csv, gridguard_sgcc_best.pth, xgboost_sgcc_edge.pkl saved")

# Restore
tsgcc.EPOCHS          = _orig_epochs
tsgcc.N_FOLDS         = _orig_n_folds
tsgcc.CHECKPOINT_EVERY = _orig_ckpt_ev

# -- 7. Abbreviated Walk-Forward (2 rounds, 2 epochs) -------------------------
print("\n[7/7] Abbreviated 2-round walk-forward (2 epochs each) ...")
import training.train_walkforward as twf
_wf_orig_epochs = twf.EPOCHS
_wf_orig_ckpt   = twf.CHECKPOINT_EVERY
twf.EPOCHS          = 2
twf.CHECKPOINT_EVERY = 10
twf.ROUND_TRAIN_ENDS = [0.54, 0.68]   # just 2 rounds for smoke test

exp2_df = twf.run_walk_forward(X_sgcc, y_sgcc, meta, output_dir=ROOT)
print(f"   Exp2 result rows: {len(exp2_df)}")
assert os.path.isfile(os.path.join(results_dir, "exp2_walkforward.csv"))
print("   [OK] exp2_walkforward.csv saved")

twf.EPOCHS          = _wf_orig_epochs
twf.CHECKPOINT_EVERY = _wf_orig_ckpt

# -- Cross-domain eval with synthetic TDD2022 ---------------------------------
print("\n[+] Cross-domain eval (TDD2022, zero-shot) ...")
from evaluation.evaluate_cross_domain import run_exp4_cross_domain_tdd

exp4_df = run_exp4_cross_domain_tdd(
    weights_path=os.path.join(models_dir, "gridguard_sgcc_best.pth"),
    xgb_path    =os.path.join(models_dir, "xgboost_sgcc_edge.pkl"),
    tdd_path    =tdd_dir,
    output_dir  =ROOT,
    sgcc_f1=0.90,
    sgcc_auroc=0.95,
)
if exp4_df is not None:
    print(f"   [OK] Exp4 done: F1={exp4_df.iloc[0]['F1']:.4f}")

# -- Summary -------------------------------------------------------------------
print("\n" + "="*60)
print("  SMOKE TEST PASSED [OK]  All pipeline stages ran successfully.")
print("  Ready to run with real SGCC + TDD2022 data on GPU.")
print("="*60)
