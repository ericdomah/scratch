import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np

# Configure pathing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gridguard")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "gridguard", "backend")))

# Import required modules
from gridguard.backend.data.dataset_loader import GridGuardDataset
from ensemble_model import GridGuardUniversalHybrid
from xgboost_model import XGBoostBaseline

# Configure Logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = "c:/Users/User/Downloads/scratch-main/gridguard/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Set seeds
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(config["system"]["seed"])

# Asymmetric Focal Loss for Imbalanced Classification
class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.80, gamma_pos=2.0, gamma_neg=4.0):
        super().__init__()
        self.alpha     = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits, targets):
        probs    = torch.sigmoid(logits).view(-1)
        targets  = targets.view(-1).float()
        bce      = F.binary_cross_entropy_with_logits(logits.view(-1), targets, reduction='none')
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        gamma_t  = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        alpha_t  = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t) ** gamma_t * bce).mean()

def train_hybrid_system():
    logger.info("Initializing GridGuard AI Hybrid Training Pipeline...")
    logger.info(f"Target TRNC Adaptation: {config['data']['trnc_mode']}")
    
    # 1. Instantiate the dataset
    dataset = GridGuardDataset(inject_ratio=config["data"]["class_prevalence"])
    
    # 2. Extract input_dim and window_size from config
    input_dim = config["model"]["input_dim"] # 2
    window_size = config["model"]["seq_len"] # 26
    epochs = config["model"]["epochs"] # 15
    
    logger.info(f"Dataset Size: {len(dataset)}")
    logger.info(f"Input Shape: (26, {input_dim}) | Sequence Length: {window_size}")
    
    # 3. Stratified or Simple Random Split for DL model
    train_size = int((1.0 - config["data"]["test_size"]) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # 4. Instantiate GridGuardUniversalHybrid DL Model
    model = GridGuardUniversalHybrid(window_size=window_size, input_dim=input_dim, hidden_dim=config["model"]["hidden_dim"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["model"]["learning_rate"], weight_decay=1e-4)
    criterion = AsymmetricFocalLoss(alpha=0.80, gamma_neg=2.0)
    
    logger.info("--- Phase 1: Training PyTorch Universal Hybrid Deep Learning Model ---")
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            # Convert multi-class labels (0 to 5) to binary (0 or 1)
            binary_y = (batch_y > 0).long().to(device)
            batch_x = batch_x.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, binary_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                binary_y = (batch_y > 0).long().to(device)
                batch_x = batch_x.to(device)
                
                logits = model(batch_x)
                loss = criterion(logits, binary_y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits).view(-1)
                preds = (probs > 0.5270).long()
                val_correct += (preds == binary_y).sum().item()
                val_total += binary_y.size(0)
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {val_acc:.2%}")
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "best_model_balanced.pth")
            logger.info("  [+] Saved best model checkpoint to best_model_balanced.pth")
            
    logger.info("Universal Hybrid PyTorch Model Training Complete!")
    
    # 5. Train XGBoost Baseline Model
    logger.info("\n--- Phase 2: Training XGBoost Baseline Model ---")
    xgb_model = XGBoostBaseline()
    
    # Extract all elements from the dataset for training XGBoost
    X_list = []
    y_list = []
    
    for i in range(len(dataset)):
        x, y = dataset[i]
        X_list.append(x)
        # Convert to binary label
        binary_label = 1 if y > 0 else 0
        y_list.append(binary_label)
        
    X_all = torch.stack(X_list)
    y_all = torch.tensor(y_list, dtype=torch.long)
    
    xgb_model.train(X_all, y_all)
    xgb_model.save_model("best_xgb_augmented.pkl")
    logger.info("[SUCCESS] Saved trained XGBoost model to best_xgb_augmented.pkl")

if __name__ == "__main__":
    train_hybrid_system()
