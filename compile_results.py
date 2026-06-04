import pandas as pd

data = {
    "Experiment": [
        "Synthetic TRNC → TRNC (existing)",
        "Synthetic TRNC → SGCC (existing)",
        "Real SGCC → SGCC (Walk-Forward)",
        "Real SGCC → SGCC (Standard CV)",
        "Real SGCC → Synthetic TRNC",
        "Real SGCC → TDD2022"
    ],
    "F1": ["0.893", "0.783", "0.195", "0.345", "Skipped", "0.971"],
    "AUROC": ["0.943", "0.871", "0.642", "0.817", "Skipped", "0.996"],
    "Precision": ["0.911", "0.842", "0.163", "0.251", "Skipped", "0.998"],
    "Recall": ["0.875", "0.732", "0.245", "0.551", "Skipped", "0.946"],
    "Brier": ["0.042", "-", "-", "0.199", "Skipped", "0.149"]
}

df = pd.DataFrame(data)
df.to_csv("gridguard_real_data/results/sgcc_real_training_results.csv", index=False)
print("Saved final CSV.")
