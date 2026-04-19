import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import os

class GridGuardTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss: Using pos_weight to handle class imbalance (common in theft detection)
        # Assuming roughly 9:1 ratio for now, can be tuned
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([11.0]).to(device))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits.view(-1), batch_y.view(-1))
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                loss = self.criterion(logits.view(-1), batch_y.view(-1))
                total_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        return avg_loss, f1, precision, recall

    def train(self, epochs=10, save_path='best_model.pth'):
        best_f1 = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, f1, prec, rec = self.evaluate()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(f1)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), save_path)
                print(f"  [BEST] Best model saved with F1: {f1:.4f}")
        return self.history

if __name__ == "__main__":
    # Full Training Run for Super-Hybrid
    from ensemble_model import GridGuardUniversalHybrid
    from data_loader import ElectricityDataset
    
    csv_path = "../../data/datasetsmall.csv"
    if not os.path.exists(csv_path):
        csv_path = "../data/datasetsmall.csv"
        
    dataset = ElectricityDataset(csv_path, window_size=30)
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GridGuardUniversalHybrid()
    trainer = GridGuardTrainer(model, train_loader, val_loader, device=device)
    
    print(f"Starting GridGuard Super-Hybrid Training on {device}...")
    trainer.train(epochs=10, save_path='best_model.pth')
    print("[OK] Training complete. Model saved to best_model.pth")
