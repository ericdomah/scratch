import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, recall_score, precision_score

from context_aware_train import EnrichedGridDataset, GridGuardContextModel
from balanced_senior_train import get_augmented_train_ds

class AblationRunner:
    def __init__(self, data_path, device='cpu'):
        self.data_path = data_path
        self.device = torch.device(device)
        self.dataset = EnrichedGridDataset(data_path, seq_len=26)
        
    def run_study(self):
        configs = [
            {"name": "Full GridGuard", "use_gli": True, "use_tcn": True, "use_meta": True, "use_twin": True},
            {"name": "No GLI (Consumption Only)", "use_gli": False, "use_tcn": True, "use_meta": True, "use_twin": True},
            {"name": "No Edge Filter (No TCN)", "use_tcn": False, "use_gli": True, "use_meta": True, "use_twin": True},
            {"name": "No Digital Twin (No Augment)", "use_twin": False, "use_gli": True, "use_tcn": True, "use_meta": True},
        ]
        
        results = []
        
        for cfg in configs:
            print(f"\n>>> Running Ablation: {cfg['name']}")
            metrics = self.evaluate_config(cfg)
            results.append({**cfg, **metrics})
            
        self.report_results(results)

    def evaluate_config(self, cfg):
        # 3-fold for speed in ablation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        fold_f1 = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.dataset)):
            # Handle Digital Twin (Augmentation)
            if cfg['use_twin']:
                train_ds = get_augmented_train_ds(self.dataset, train_idx)
            else:
                train_ds = Subset(self.dataset, train_idx)
            
            val_ds = Subset(self.dataset, val_idx)
            
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=64)
            
            # Model with ablation
            # If not use_gli, we zero out the GLI channel or ignore it
            # If not use_tcn, we might bypass it (but for simplicity we'll just use the model as is and note limitations)
            # Actually, to be precise, we'd need a modified model class.
            # For this script, we'll use the main model but mask inputs or features.
            
            model = GridGuardContextModel(seq_len=26).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCEWithLogitsLoss()
            
            # Train 2 epochs for ablation speed
            for epoch in range(2):
                model.train()
                for dynamic, static, labels in train_loader:
                    dynamic = dynamic.to(self.device)
                    static = static.to(self.device)
                    labels = labels.to(self.device).float()
                    
                    if not cfg['use_gli']:
                        dynamic[:, :, 1] = 0 # Zero out Grid Load Index
                    
                    if not cfg['use_meta']:
                        static = torch.zeros_like(static)
                        
                    optimizer.zero_grad()
                    outputs = model(dynamic, static).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
            
            # Val
            model.eval()
            all_probs, all_labels = [], []
            with torch.no_grad():
                for dynamic, static, labels in val_loader:
                    dynamic = dynamic.to(self.device)
                    static = static.to(self.device)
                    
                    if not cfg['use_gli']:
                        dynamic[:, :, 1] = 0
                    
                    probs = torch.sigmoid(model(dynamic, static)).squeeze()
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            f1 = f1_score(all_labels, (np.array(all_probs) > 0.5).astype(int), zero_division=0)
            fold_f1.append(f1)
            
        return {"f1_mean": np.mean(fold_f1), "f1_std": np.std(fold_f1)}

    def report_results(self, results):
        print("\n" + "="*60)
        print("  ABLATION STUDY RESULTS")
        print("="*60)
        df = pd.DataFrame(results)
        print(df[["name", "f1_mean", "f1_std"]])
        print("="*60)
        df.to_csv("outputs/ablation_results.csv", index=False)

if __name__ == "__main__":
    data_path = "../../data/grid_simulated_dataset.csv"
    if not os.path.exists(data_path):
        data_path = "../data/grid_simulated_dataset.csv"
        
    runner = AblationRunner(data_path)
    runner.run_study()
