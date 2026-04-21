import pandas as pd
import os

DATA_PATH = "../../data/data set.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../data/data set.csv"

print(f"Reading {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)
print(f"Read {len(df)} rows.")
