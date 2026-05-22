import os
import random
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Configure Logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = "c:/Users/User/Downloads/scratch-main/gridguard/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Set seeds
def set_seed(seed=None):
    if seed is None:
        seed = config["system"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed

set_seed()

class GridGuardDataset(Dataset):
    """
    Robust Dataset Loader for the GridGuard AI system.
    Resolves Fix 3: Sequence Window Contradiction by aggregating profiles into 26-week sequence steps.
    Generates multi-channel inputs of shape (26, 2) [Feature 0: kWh, Feature 1: GLI].
    """
    
    def __init__(self, csv_path=None, train_mode=True, inject_ratio=0.15):
        if csv_path is None:
            csv_path = config["data"]["raw_csv_path"]
            
        self.csv_path = csv_path
        self.train_mode = train_mode
        self.inject_ratio = inject_ratio
        
        # Load TRNC Adaptation Mode from config
        self.trnc_mode = config.get("data", {}).get("trnc_mode", True)
        
        # Load raw data
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Source smart meter CSV database not found at {self.csv_path}")
            
        self.df = pd.read_csv(self.csv_path)
        self.cons_no = self.df["CONS_NO"].values
        self.raw_labels = self.df["FLAG"].values.astype(int)
        
        # Drops CONS_NO and FLAG to extract consumption values (exactly 26 weeks)
        self.raw_consumption = self.df.drop(["CONS_NO", "FLAG"], axis=1).values
        
        # Instantiate TheftInjector
        try:
            from backend.data.theft_injector import TheftInjector
        except ImportError:
            try:
                from .theft_injector import TheftInjector
            except ImportError:
                from theft_injector import TheftInjector
        self.injector = TheftInjector()
        
        # Preprocess dataset
        self.preprocessed_consumption = self._preprocess_all(self.raw_consumption)
        
        # Injected cohort labels
        self.final_labels = self.raw_labels.copy()
        
        # Inject synthetic theft signatures to achieve target class prevalence (15% theft)
        self._balance_dataset()

    def _preprocess_all(self, data):
        """Applies linear interpolation and 3-sigma outlier clipping per consumer profile."""
        processed = np.zeros_like(data, dtype=np.float32)
        for idx in range(len(data)):
            series = pd.Series(data[idx])
            # Linear interpolation for missing readings
            interpolated = series.interpolate(method="linear").fillna(0.0).values
            
            # 3-sigma outlier clipping
            mean = np.mean(interpolated)
            std = np.std(interpolated)
            if std > 0:
                clipped = np.clip(interpolated, mean - 3 * std, mean + 3 * std)
            else:
                clipped = interpolated
                
            # Min-Max normalization per consumer
            c_min = np.min(clipped)
            c_max = np.max(clipped)
            if c_max > c_min:
                normed = (clipped - c_min) / (c_max - c_min)
            else:
                normed = np.zeros_like(clipped)
                
            processed[idx] = normed
        return processed

    def _balance_dataset(self):
        """
        Injects specific theft profiles in some normal lines to achieve the target 15% class ratio.
        Assigns randomized theft types 1 to 5.
        """
        normal_indices = np.where(self.final_labels == 0)[0]
        theft_indices = np.where(self.final_labels == 1)[0]
        
        target_thefts = int(round(len(self.df) * self.inject_ratio))
        thefts_to_inject = target_thefts - len(theft_indices)
        
        # Seed-based deterministic shuffling
        np.random.seed(config["system"]["seed"])
        shuffled_normals = np.random.permutation(normal_indices)
        
        # Convert existing marked thefts (FLAG=1) to a random theft type (1 to 5)
        for idx in theft_indices:
            ttype = (idx % 5) + 1
            self.final_labels[idx] = ttype
            
        # Inject additional thefts to reach target 15%
        if thefts_to_inject > 0:
            for idx in shuffled_normals[:thefts_to_inject]:
                # Inject a random theft type (1 to 5)
                ttype = (idx % 5) + 1
                self.final_labels[idx] = ttype
                
        # Log final stats
        total = len(self.final_labels)
        active_thefts = np.sum(self.final_labels > 0)
        logger.info(f"Balanced Dataset: Total={total} | Theft Count={active_thefts} ({active_thefts/total:.2%})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        kwh = self.preprocessed_consumption[idx]
        label = self.final_labels[idx]
        
        # If in KIB-TEK / TRNC Adaptation Mode, apply 13-week seasonal peak shift
        # This converts SGCC winter peaks into realistic Northern Cyprus summer peaks
        if self.trnc_mode:
            kwh = np.roll(kwh, 13)
            
        # If labeled as active theft, inject the corresponding pattern
        if label > 0:
            kwh_tensor, _ = self.injector.inject_theft(torch.tensor(kwh, dtype=torch.float32), theft_type=label)
            kwh = kwh_tensor.numpy()
            
        # Generate companion Grid Load Index (GLI) sequence (Feature 1) of shape (26)
        # Represents typical regional grid fluctuation centered around 0.5 population mean
        np.random.seed(config["system"]["seed"] + idx)
        
        # In TRNC mode, phase-align the GLI base curve so the grid peak lines up with the summer consumption peak
        if self.trnc_mode:
            gli_base = 0.5 + 0.12 * np.sin(np.linspace(np.pi, 5 * np.pi, 26))
        else:
            gli_base = 0.5 + 0.12 * np.sin(np.linspace(0, 4 * np.pi, 26))
            
        gli = gli_base + np.random.normal(0.0, 0.03, 26)
        gli = np.clip(gli, 0.0, 1.0)
        
        # Stack to form a (26, 2) input tensor
        seq_2d = np.stack([kwh, gli], axis=1) # (26, 2)
        
        return torch.tensor(seq_2d, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    print("--- GridGuardDataset Loader Verification ---")
    dataset = GridGuardDataset(inject_ratio=0.15)
    
    # Print sample shape and label
    x, y = dataset[0]
    print(f"Sample Input Shape: {x.shape} (Expected: torch.Size([26, 2]))")
    print(f"Sample Label: {y.item()} (Expected range: 0 to 5)")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch X Shape: {batch_x.shape} (Expected: torch.Size([16, 26, 2]))")
    print(f"Batch Y Shape: {batch_y.shape} (Expected: torch.Size([16]))")
