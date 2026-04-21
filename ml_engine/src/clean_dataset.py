import pandas as pd
import numpy as np
import os

# Configuration
RAW_DATA_PATH = "../../data/data set.csv"
FALLBACK_RAW_DATA_PATH = "../data/data set.csv"
CLEANED_DATA_PATH = "../../data/data_set_cleaned.csv"
FALLBACK_CLEANED_DATA_PATH = "../data/data_set_cleaned.csv"

def clean_data():
    data_path = RAW_DATA_PATH if os.path.exists(RAW_DATA_PATH) else FALLBACK_RAW_DATA_PATH
    out_path = CLEANED_DATA_PATH if os.path.exists(RAW_DATA_PATH) else FALLBACK_CLEANED_DATA_PATH
    
    print(f"Reading raw dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Initial shape: {df.shape}")
    
    # Identify consumption columns vs metadata
    meta_cols = ['CONS_NO', 'FLAG']
    cons_cols = [c for c in df.columns if c not in meta_cols]
    
    # Force convert to numeric (turns invalid strings to NaN)
    print("Coercing consumption data to numeric...")
    df[cons_cols] = df[cons_cols].apply(pd.to_numeric, errors='coerce')
    
    # 1. Drop highly sparse rows (>50% NaN)
    print("Dropping rows with >50% missing values...")
    threshold = int(len(cons_cols) * 0.5)
    # Require at least threshold non-NaN values
    df_cleaned = df.dropna(thresh=threshold, subset=cons_cols).copy()
    print(f"Shape after dropping sparse rows: {df_cleaned.shape}")
    
    # 2. Impute remaining missing values
    # We use linear interpolation across the columns (axis=1) for each user
    print("Interpolating remaining missing values...")
    df_cleaned[cons_cols] = df_cleaned[cons_cols].interpolate(method='linear', axis=1, limit_direction='both')
    
    # Fill any remaining NaNs with 0 (just in case)
    df_cleaned[cons_cols] = df_cleaned[cons_cols].fillna(0)
    
    # 3. Cap extreme outliers (e.g. > 10,000 kWh per day is likely a logging error)
    # We'll use a conservative absolute cap to remove impossible values like 92713.9
    print("Capping extreme outliers...")
    cap_value = 10000.0
    df_cleaned[cons_cols] = df_cleaned[cons_cols].clip(upper=cap_value)
    
    # Drop rows where total consumption is 0 (dead meters)
    print("Removing dead meters (0 total consumption)...")
    total_consumption = df_cleaned[cons_cols].sum(axis=1)
    df_cleaned = df_cleaned[total_consumption > 0]
    print(f"Shape after removing dead meters: {df_cleaned.shape}")
    
    print(f"Saving cleaned dataset to {out_path}...")
    df_cleaned.to_csv(out_path, index=False)
    print("Data cleaning complete!")

if __name__ == "__main__":
    clean_data()
