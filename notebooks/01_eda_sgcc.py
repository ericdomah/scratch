import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_explore(file_path):
    print(f"Loading dataset from {file_path}...")
    # The SGCC dataset is usually a large CSV
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print("Error: Dataset file not found. Please download SGCC dataset to the /data folder.")
        return

    print(f"Dataset Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())

    # 1. Check for missing values
    missing_pct = df.isnull().sum().sum() / df.size * 100
    print(f"\nMissing Values: {missing_pct:.2f}%")

    # 2. Check Class Balance (FLAG column)
    if 'FLAG' in df.columns:
        balance = df['FLAG'].value_counts(normalize=True) * 100
        print(f"\nClass Balance:\n{balance}")
        
        sns.countplot(x='FLAG', data=df)
        plt.title("Class Distribution (0: Normal, 1: Theft)")
        plt.show()
    else:
        print("\n'FLAG' column not found! Check dataset format.")

    # 3. Plot a few samples (Normal vs Theft)
    if 'FLAG' in df.columns:
        normal_sample = df[df['FLAG'] == 0].iloc[0, 1:-1]
        theft_sample = df[df['FLAG'] == 1].iloc[0, 1:-1]

        plt.figure(figsize=(15, 6))
        plt.plot(normal_sample.values, label='Normal Customer', alpha=0.8)
        plt.plot(theft_sample.values, label='Theft Customer', color='red', alpha=0.8)
        plt.title("Daily Consumption Patterns (Normal vs Theft)")
        plt.xlabel("Days")
        plt.ylabel("Consumption")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "data.csv") # Adjust filename as needed
    load_and_explore(DATA_PATH)
