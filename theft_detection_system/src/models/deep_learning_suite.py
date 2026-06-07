import logging
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import yaml

from theft_detection_system.src.models.deep_learning.lstm import LSTMClassifier
from theft_detection_system.src.models.deep_learning.cnn import CNN1D
from theft_detection_system.src.models.deep_learning.transformer import TransformerClassifier
from theft_detection_system.src.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)

class DeepLearningModelSuite:
    """
    Train and evaluate deep learning models (LSTM, CNN, Transformer) for theft detection.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "output",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "dl_models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.config = self._load_config(config_path)
        self.trained_models = {}
        self.metrics = {}
        logger.info(f"DeepLearningModelSuite initialized using device: {self.device}")

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        if config_path is None or not Path(config_path).exists():
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("deep_learning", {})

    def _prepare_dataloader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int, is_train: bool = False
    ) -> DataLoader:
        # Convert to tensor: (B, seq_len) -> (B, seq_len, 1)
        if len(X.shape) == 2:
            X = np.expand_dims(X, axis=-1)
        
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        dataset = TensorDataset(X_tensor, y_tensor)

        if is_train:
            # Handle imbalance via WeightedRandomSampler instead of SMOTE
            class_counts = np.bincount(y.astype(int))
            total = len(y)
            weights = total / (len(class_counts) * class_counts)
            sample_weights = np.array([weights[int(label)] for label in y])
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=total, replacement=True
            )
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def _build_model(self, name: str, input_dim: int, seq_len: int) -> nn.Module:
        cfg = self.config.get(name, {})
        if name == "lstm":
            return LSTMClassifier(
                input_dim=input_dim,
                seq_len=seq_len,
                hidden_dim=cfg.get("hidden_dim", 128),
                num_layers=cfg.get("num_layers", 2),
                dropout=cfg.get("dropout", 0.3),
            )
        elif name == "cnn":
            return CNN1D(
                input_dim=input_dim,
                seq_len=seq_len,
                channels=cfg.get("channels", [64, 128, 256]),
                dropout=cfg.get("dropout", 0.3),
            )
        elif name == "transformer":
            return TransformerClassifier(
                input_dim=input_dim,
                seq_len=seq_len,
                d_model=cfg.get("d_model", 128),
                nhead=cfg.get("nhead", 8),
                num_encoder_layers=cfg.get("num_encoder_layers", 4),
                dropout=cfg.get("dropout", 0.1),
            )
        else:
            raise ValueError(f"Unknown deep learning model: {name}")

    def train_single(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 128,
        lr: float = 1e-3,
        patience: int = 10,
    ) -> nn.Module:
        logger.info(f"Preparing data for '{name}'...")
        train_loader = self._prepare_dataloader(X_train, y_train, batch_size, is_train=True)
        val_loader = self._prepare_dataloader(X_val, y_val, batch_size, is_train=False)

        input_dim = 1 if len(X_train.shape) == 2 else X_train.shape[2]
        seq_len = X_train.shape[1]

        model = self._build_model(name, input_dim, seq_len).to(self.device)
        criterion = nn.BCELoss() # Using Sigmoid in models
        optimizer = optim.Adam(model.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        logger.info(f"Starting training loop for '{name}'...")
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_b)
                loss = criterion(outputs, y_b)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_b.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    outputs = model(X_b)
                    loss = criterion(outputs, y_b)
                    val_loss += loss.item() * X_b.size(0)
            val_loss /= len(val_loader.dataset)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        if best_model_state:
            model.load_state_dict(best_model_state)
        
        self.trained_models[name] = model
        
        # Save model
        save_path = self.models_dir / f"{name}.pt"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model '{name}' saved to {save_path}")

        return model

    def evaluate_single(
        self, name: str, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5, batch_size: int = 32
    ) -> Dict[str, Any]:
        if name not in self.trained_models:
            raise KeyError(f"Model '{name}' not trained.")
        
        model = self.trained_models[name]
        model.eval()
        test_loader = self._prepare_dataloader(X_test, y_test, batch_size=batch_size, is_train=False)

        all_probs = []
        all_targets = []
        
        t0 = time.perf_counter()
        with torch.no_grad():
            for X_b, y_b in test_loader:
                X_b = X_b.to(self.device)
                probs = model(X_b).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(y_b.numpy())
        
        infer_time = time.perf_counter() - t0
        
        y_prob = np.array(all_probs).squeeze()
        y_true = np.array(all_targets).squeeze()

        evaluator = Evaluator()
        metrics = evaluator.compute_metrics(y_true, y_prob, threshold=threshold)
        metrics["model"] = name
        metrics["inference_time_ms_per_sample"] = (infer_time / len(y_true)) * 1000

        self.metrics[name] = metrics
        return metrics
