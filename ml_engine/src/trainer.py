import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        # Use logsigmoid for better numerical stability
        log_probs = F.logsigmoid(logits)
        log_not_probs = F.logsigmoid(-logits)
        
        # Cross Entropy components
        bce_loss = -(targets * log_probs + (1 - targets) * log_not_probs)
        
        # Focal components
        pt = torch.exp(-bce_loss)
        
        # Apply weighting (alpha) correctly as a positive multiplier
        if self.alpha is not None:
            # We treat alpha as the positive class weight ratio
            weight = torch.where(targets == 1, self.alpha, torch.ones_like(targets))
            loss = weight * (1 - pt) ** self.gamma * bce_loss
        else:
            loss = (1 - pt) ** self.gamma * bce_loss

        return loss.mean()

class GridGuardTrainer:
    def __init__(self, model, train_loader, val_loader, device,
                 loss_type='bce', class_weights=None):

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.loss_type = loss_type

        # Loss selection (Supporting your class_weights parameter)
        if loss_type == 'focal':
            alpha = None
            if class_weights is not None:
                alpha = class_weights[1].to(device)
            self.criterion = FocalLoss(alpha=alpha)
        else:
            pos_weight = None
            if class_weights is not None:
                pos_weight = class_weights[1].to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        self.best_f1 = -1
        self.best_loss = float('inf')
        print(f"⚙️ Trainer ready | Loss: {loss_type} | Weights: {class_weights is not None}")

    def train(self, epochs=10, save_path='best_model.pth'):
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float().to(self.device)

                logits = self.model(batch_x).view(-1)
                loss = self.criterion(logits, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(self.train_loader)
            val_loss, f1, prec, rec = self.validate_with_metrics()

            print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | F1: {f1:.4f} | Prec: {prec:.4f}")

            # Smart Saving: Save if F1 improves OR if loss improves while F1 is 0
            if f1 > self.best_f1 or (f1 == 0 and val_loss < self.best_loss):
                if f1 > self.best_f1: self.best_f1 = f1
                if val_loss < self.best_loss: self.best_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f"  [SAVED] Checkpoint at epoch {epoch+1}")

        print("✅ Training complete")

    def validate_with_metrics(self):
        self.model.eval()
        total_loss = 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.float().to(self.device)

                logits = self.model(batch_x).view(-1)
                loss = self.criterion(logits, batch_y)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_y.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate metrics for the logs
        preds = (torch.tensor(all_probs) > 0.5).float().numpy()
        f1 = f1_score(all_labels, preds, zero_division=0)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        
        return avg_loss, f1, prec, rec
