import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import recall_score, precision_score, f1_score

# --- 1. Context-Aware Dataset Class ---

class EnrichedGridDataset(Dataset):
    def __init__(self, csv_path, seq_len=30):
        print(f"Loading Enriched Grid Dataset: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 1. Label Encoding for Categorical Metadata
        self.region_le = LabelEncoder()
        self.type_le = LabelEncoder()
        
        df['region_encoded'] = self.region_le.fit_transform(df['region_id'])
        df['type_encoded'] = self.type_le.fit_transform(df['customer_type'])
        
        # 2. Group by Household to reconstruct sequences
        self.sequences = []
        self.labels = []
        self.meta = []
        
        groups = df.groupby('household_id')
        for hid, group in groups:
            # Sort by timestamp to ensure temporal order
            group = group.sort_values('timestamp')
            
            # Extract dynamic features: [Consumption, Grid Load]
            dynamic_features = group[['consumption_kwh', 'grid_load_index']].values
            
            # Extract static metadata: [Region, Type] (take from first record)
            static_meta = group[['region_encoded', 'type_encoded']].iloc[0].values
            
            # Extract label
            label = group['anomaly_label'].iloc[0]
            
            # Padding/Trimming to seq_len
            if len(dynamic_features) >= seq_len:
                self.sequences.append(dynamic_features[:seq_len])
                self.labels.append(label)
                self.meta.append(static_meta)
        
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)
        self.meta = np.array(self.meta)
        
        print(f"Loaded {len(self.sequences)} households with {seq_len}-day sequences.")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.meta[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# --- 2. Context-Aware Model Architecture ---

class GridGuardContextModel(nn.Module):
    def __init__(self, seq_len=30, dynamic_dim=2, num_regions=4, num_types=2):
        super(GridGuardContextModel, self).__init__()
        
        # 1. Temporal Head (Processes Consumption + Grid Load)
        self.tcn = nn.Sequential(
            nn.Conv1d(dynamic_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # 2. Static Head (Processes Region + Customer Type)
        self.region_emb = nn.Embedding(num_regions, 8)
        self.type_emb = nn.Embedding(num_types, 4)
        
        # 3. Fusion Layer
        self.classifier = nn.Sequential(
            nn.Linear(64 + 8 + 4, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, dynamic, static):
        # dynamic shape: (batch, seq_len, 2) -> (batch, 2, seq_len) for Conv1d
        dynamic = dynamic.permute(0, 2, 1)
        temporal_feat = self.tcn(dynamic).squeeze(-1)
        
        # static shape: (batch, 2)
        reg_feat = self.region_emb(static[:, 0])
        typ_feat = self.type_emb(static[:, 1])
        
        # Concatenate Features
        combined = torch.cat([temporal_feat, reg_feat, typ_feat], dim=1)
        return self.classifier(combined)

# --- 3. Training Loop ---

def train_context_aware():
    print("=" * 60)
    print("  GRIDGUARD AI: CONTEXT-AWARE TRAINING PIPELINE")
    print("=" * 60)

    data_path = "../../data/grid_simulated_dataset.csv"
    if not os.path.exists(data_path):
        data_path = "../data/grid_simulated_dataset.csv"
        
    dataset = EnrichedGridDataset(data_path)
    
    # K-Fold Evaluation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\n>> Starting Fold {fold+1}...")
        
        train_sub = torch.utils.data.Subset(dataset, train_idx)
        val_sub = torch.utils.data.Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=32, shuffle=False)
        
        model = GridGuardContextModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        # Train for a few epochs
        for epoch in range(10):
            model.train()
            for dynamic, static, labels in train_loader:
                dynamic, static, labels = dynamic.to(device), static.to(device), labels.to(device).float()
                
                optimizer.zero_grad()
                outputs = model(dynamic, static).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Validate
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for dynamic, static, labels in val_loader:
                dynamic, static, labels = dynamic.to(device), static.to(device), labels.to(device).float()
                probs = torch.sigmoid(model(dynamic, static)).squeeze()
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Metrics
        probs = np.array(all_probs)
        labels = np.array(all_labels)
        preds = (probs > 0.5).astype(int)
        
        print(f"Fold {fold+1} Results:")
        print(f"  Recall: {recall_score(labels, preds):.4f}")
        print(f"  Precision: {precision_score(labels, preds):.4f}")
        print(f"  F1 Score: {f1_score(labels, preds):.4f}")

    # Save final context model
    torch.save(model.state_dict(), "best_context_aware_model.pth")
    print("\n[SUCCESS] Context-Aware Model Saved: best_context_aware_model.pth")

if __name__ == "__main__":
    train_context_aware()
