import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from preprocessing import DataPreprocessor

class ElectricityDataset(Dataset):
    def __init__(self, csv_path, window_size=20, transform=True):
        self.df = pd.read_csv(csv_path)
        self.window_size = window_size
        self.transform = transform
        self.preprocessor = DataPreprocessor()
        
        # Identification columns
        self.cons_no = self.df['CONS_NO'].values
        self.labels = self.df['FLAG'].values
        
        # Consumption columns
        self.consumption_data = self.df.drop(['CONS_NO', 'FLAG'], axis=1).values
        # Initial cleanup: handle basic NaN to 0 before complex processing
        self.consumption_data = np.nan_to_num(self.consumption_data.astype(float))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_data = self.consumption_data[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Apply Module 3 Preprocessing (Interpolation, Clipping, Normalization)
            user_data = self.preprocessor.process_user_data(user_data)
        
        data_window = user_data[:self.window_size]
        
        return torch.tensor(data_window, dtype=torch.float32).unsqueeze(-1), torch.tensor(label, dtype=torch.float32)

if __name__ == "__main__":
    # Test Module 3 Integration
    csv_path = "../../data/datasetsmall.csv"
    if not os.path.exists(csv_path):
        csv_path = "../data/datasetsmall.csv"
    ds = ElectricityDataset(csv_path, transform=True)
    x, y = ds[0]
    print(f"Preprocessed Sample - min: {x.min().item():.2f}, max: {x.max().item():.2f}, label: {y}")
