# GridGuard AI — Phase 1: Real-Data Training Upgrade

MSc Thesis extension that retrains `GridGuardUniversalHybrid` on real-world
electricity theft datasets (SGCC + TDD2022) and evaluates cross-domain
generalisation.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.9+, CUDA 11.8 recommended)
pip install -r requirements.txt

# 2. Place raw data files
#    data/sgcc/   ← data.csv (SGCC, from GitHub or Kaggle)
#    data/tdd2022/← *.csv   (TDD2022, from Mendeley DOI:10.17632/c3c7329tjj.1)
#    data/trnc_synthetic_test.pt  ← optional, for Experiment 3

# 3. Run all experiments
python run_all.py --sgcc_path data/sgcc --tdd_path data/tdd2022
```

---

## File Structure

```
gridguard_real_data/
├── data/
│   ├── sgcc/               ← place SGCC CSV here (data.csv)
│   └── tdd2022/            ← place TDD2022 CSV here
├── models/
│   ├── gridguard_model.py  ← frozen thesis architecture + BiGRU-BiLSTM baseline
│   ├── gridguard_sgcc_best.pth   ← saved after Experiment 1
│   └── xgboost_sgcc_edge.pkl     ← saved after Experiment 1
├── preprocessing/
│   ├── sgcc_pipeline.py    ← 7-step SGCC preprocessing → (N,26,2) tensor
│   └── tdd2022_pipeline.py ← TDD2022 preprocessing  → (N,26,2) tensor
├── training/
│   ├── train_sgcc.py       ← Experiment 1: 10-fold StratifiedKFold CV
│   └── train_walkforward.py← Experiment 2: 7-round expanding walk-forward
├── evaluation/
│   └── evaluate_cross_domain.py  ← Experiments 3 & 4: zero-shot evaluation
├── results/
│   ├── exp1_standard_cv.csv
│   ├── exp2_walkforward.csv
│   ├── exp3_reverse_transfer.csv
│   ├── exp4_cross_domain_tdd.csv
│   └── sgcc_real_training_results.csv  ← FINAL comparison table
├── checkpoints/            ← intermediate epoch checkpoints
├── run_all.py              ← master pipeline script
└── requirements.txt
```

---

## Datasets

### SGCC
- **Source**: https://github.com/henryRDlab/ElectricityTheftDetection  
  or https://www.kaggle.com/datasets/bensalem14/sgcc-dataset
- **Format**: CSV — `CONS_NO`, `FLAG`, then daily kWh date columns  
  (2014-01-01 → 2016-10-31, ~1,035 date cols, ~42,372 rows)
- **Prevalence**: ~5 % theft

### TDD2022
- **Source**: https://data.mendeley.com/datasets/c3c7329tjj/1  
  DOI: 10.17632/c3c7329tjj.1  (no registration required)
- **Format**: Hourly readings, 16 consumer types, 6 theft + 1 normal class

---

## Experiments

| # | Name | Protocol | Key Output |
|---|------|----------|-----------|
| 1 | SGCC Standard CV | 10-fold StratifiedKFold | `exp1_standard_cv.csv` |
| 2 | SGCC Walk-Forward | 7-round expanding window | `exp2_walkforward.csv` |
| 3 | Reverse Transfer | SGCC model → synthetic TRNC | `exp3_reverse_transfer.csv` |
| 4 | Cross-Domain | SGCC model → TDD2022 | `exp4_cross_domain_tdd.csv` |

---

## Architecture (frozen — DO NOT MODIFY)

```
Input (B, 26, 2)
  ├── TCN  [dilation=1] → TCN [dilation=2] → AvgPool → vec₁ (B, 64)
  └── Bi-LSTM → Transformer Encoder (4-head, 2-layer) → last token → vec₂ (B, 64)
                          ↓
              concat [vec₁ ∥ vec₂] → FC(128→64→32→1) → sigmoid
```

Late fusion: **0.70 × P_DL + 0.30 × P_XGB**, threshold **τ = 0.5270**

---

## Run Individual Scripts

```bash
# Preprocessing smoke-test
python preprocessing/sgcc_pipeline.py    data/sgcc
python preprocessing/tdd2022_pipeline.py data/tdd2022

# Experiment 1 only
python training/train_sgcc.py --sgcc_path data/sgcc --output_dir .

# Experiment 2 only
python training/train_walkforward.py --sgcc_path data/sgcc --output_dir .

# Experiments 3 & 4 only (requires Exp 1 weights)
python evaluation/evaluate_cross_domain.py \
    --weights models/gridguard_sgcc_best.pth \
    --xgb     models/xgboost_sgcc_edge.pkl   \
    --trnc_test data/trnc_synthetic_test.pt  \
    --tdd_path  data/tdd2022                 \
    --output_dir .
```

---

## Colab Upload Instructions

1. Upload the entire `gridguard_real_data/` folder to your Google Drive
2. Mount Drive and `cd` into the folder
3. Install requirements with `pip install -r requirements.txt`
4. Upload raw data CSVs into `data/sgcc/` and `data/tdd2022/`
5. Run: `python run_all.py --sgcc_path data/sgcc --tdd_path data/tdd2022`

> Recommended: Google Colab Pro with Tesla T4 GPU, PyTorch 2.1.0+cu118 runtime.
