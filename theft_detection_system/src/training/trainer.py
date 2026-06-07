"""
trainer.py
==========
Production-quality PyTorch Trainer for the theft detection system.

Features
--------
- Automatic Mixed Precision (AMP) via GradScaler + autocast
- Gradient clipping
- Early stopping (val loss or val F1)
- Best + latest checkpoint saving / resume
- Multi-GPU via nn.DataParallel
- A100 optimisations (allow_tf32, cudnn.benchmark)
- TensorBoard logging
- Weights & Biases logging (optional)
- Optimisers: Adam, AdamW, SGD, Ranger (LookAhead + RAdam)
- Schedulers: CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
- Training history and curve plotting
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore

    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import matplotlib.pyplot as plt  # type: ignore

    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ===========================================================================
# Ranger optimiser  (LookAhead + RAdam)
# ===========================================================================

class _RAdam(optim.Optimizer):
    """Rectified Adam (RAdam) optimiser."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Optional[float]:  # type: ignore[override]
        loss: Optional[float] = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p.data, dtype=torch.float32
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                t = state["step"]

                if group["weight_decay"] != 0:
                    grad = grad.add(p.data.float(), alpha=group["weight_decay"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                beta2_t = beta2**t
                N_sma_max = 2.0 / (1.0 - beta2) - 1.0
                N_sma = N_sma_max - 2.0 * t * beta2_t / (1.0 - beta2_t)

                bias_corr1 = 1.0 - beta1**t
                if N_sma >= 5:
                    step_size = (
                        math.sqrt(
                            (1.0 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        )
                        / bias_corr1
                    )
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p.data.addcdiv_(exp_avg, denom, value=-group["lr"] * step_size)
                else:
                    step_size = 1.0 / bias_corr1
                    p.data.add_(exp_avg, alpha=-group["lr"] * step_size)

        return loss


class _LookAhead(optim.Optimizer):
    """LookAhead wrapper around any base optimiser."""

    def __init__(
        self,
        base_optimizer: optim.Optimizer,
        k: int = 6,
        alpha: float = 0.5,
    ) -> None:
        self.optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self._step_counter = 0
        self.slow_weights: List[torch.Tensor] = []
        for group in base_optimizer.param_groups:
            self.slow_weights.append(
                [p.data.clone() for p in group["params"] if p.requires_grad]
            )
        self.param_groups = base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.state = base_optimizer.state

    def step(self, closure: Any = None) -> Optional[float]:  # type: ignore[override]
        loss = self.optimizer.step(closure)
        self._step_counter += 1
        if self._step_counter % self.k == 0:
            for group, slow_group in zip(self.optimizer.param_groups, self.slow_weights):
                trainable = [p for p in group["params"] if p.requires_grad]
                for fast, slow in zip(trainable, slow_group):
                    slow.add_(fast.data - slow, alpha=self.alpha)
                    fast.data.copy_(slow)
        return loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)


def build_ranger(
    params: Any,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.95, 0.999),
    eps: float = 1e-5,
    weight_decay: float = 0.0,
    k: int = 6,
    alpha: float = 0.5,
) -> _LookAhead:
    """Return a Ranger optimiser (LookAhead + RAdam)."""
    radam = _RAdam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    return _LookAhead(radam, k=k, alpha=alpha)


# ===========================================================================
# Main Trainer
# ===========================================================================


class Trainer:
    """
    A production-quality PyTorch trainer for binary classification.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    config : dict
        Training configuration.  Recognised keys (with defaults):

        ==============================  ==========================================
        Key                             Description
        ==============================  ==========================================
        use_amp                         bool, enable AMP (default True)
        grad_clip                       float, max grad norm (default 1.0)
        early_stopping_patience         int (default 10)
        early_stopping_metric           ``"val_loss"`` or ``"val_f1"`` (default)
        multi_gpu                       bool (default False)
        allow_tf32                      bool, A100 (default False)
        cudnn_benchmark                 bool (default False)
        use_wandb                       bool (default False)
        wandb_project                   str
        wandb_run_name                  str
        optimizer                       ``"adam"`` / ``"adamw"`` / ``"sgd"`` /
                                        ``"ranger"`` (default ``"adamw"``)
        lr                              float (default 1e-3)
        weight_decay                    float (default 1e-4)
        scheduler                       ``"cosine"`` / ``"onecycle"`` /
                                        ``"plateau"`` / ``None`` (default)
        T_max                           int for cosine (default 50)
        max_lr                          float for onecycle (default 1e-2)
        steps_per_epoch                 int for onecycle
        ==============================  ==========================================

    device : torch.device
    output_dir : str or Path
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        output_dir: Union[str, Path],
    ) -> None:
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # A100 / cuDNN tweaks
        if config.get("allow_tf32", False):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled for matmul and cuDNN.")
        if config.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark mode enabled.")

        # Multi-GPU
        self.use_multi_gpu: bool = (
            config.get("multi_gpu", False)
            and torch.cuda.device_count() > 1
        )
        if self.use_multi_gpu:
            model = nn.DataParallel(model)
            logger.info(
                "DataParallel enabled across %d GPUs.", torch.cuda.device_count()
            )
        self.model = model.to(device)

        # AMP
        self.use_amp: bool = config.get("use_amp", True) and device.type == "cuda"
        self.scaler: GradScaler = GradScaler(enabled=self.use_amp)

        # Gradient clipping
        self.grad_clip: float = float(config.get("grad_clip", 1.0))

        # Early stopping
        self.patience: int = int(config.get("early_stopping_patience", 10))
        self.es_metric: str = config.get("early_stopping_metric", "val_f1")
        self._best_es_value: float = -math.inf if "f1" in self.es_metric else math.inf
        self._es_counter: int = 0
        self._best_epoch: int = 0

        # History
        self._history: Dict[str, List[float]] = defaultdict(list)

        # TensorBoard
        tb_dir = self.output_dir / "tensorboard"
        if _TB_AVAILABLE:
            self._writer: Optional[Any] = SummaryWriter(log_dir=str(tb_dir))
            logger.info("TensorBoard logging → %s", tb_dir)
        else:
            self._writer = None
            logger.warning("tensorboard not installed; TB logging disabled.")

        # Weights & Biases
        self.use_wandb: bool = config.get("use_wandb", False) and _WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "theft_detection"),
                name=config.get("wandb_run_name", None),
                config=config,
            )
            logger.info("Weights & Biases initialised.")
        elif config.get("use_wandb", False) and not _WANDB_AVAILABLE:
            logger.warning("wandb not installed; W&B logging disabled.")

        logger.info(
            "Trainer ready | device=%s | amp=%s | grad_clip=%.2f | patience=%d",
            self.device,
            self.use_amp,
            self.grad_clip,
            self.patience,
        )

    # ------------------------------------------------------------------
    # Optimiser factory
    # ------------------------------------------------------------------

    def build_optimizer(self, params: Any) -> optim.Optimizer:
        """Construct and return the configured optimiser."""
        name = self.config.get("optimizer", "adamw").lower()
        lr = float(self.config.get("lr", 1e-3))
        wd = float(self.config.get("weight_decay", 1e-4))

        if name == "adam":
            opt = optim.Adam(params, lr=lr, weight_decay=wd)
        elif name == "adamw":
            opt = optim.AdamW(params, lr=lr, weight_decay=wd)
        elif name == "sgd":
            opt = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        elif name == "ranger":
            opt = build_ranger(params, lr=lr, weight_decay=wd)
        else:
            raise ValueError(
                f"Unknown optimizer '{name}'. Choose from: adam, adamw, sgd, ranger."
            )
        logger.info("Optimizer: %s | lr=%.6f | wd=%.6f", name, lr, wd)
        return opt

    # ------------------------------------------------------------------
    # Scheduler factory
    # ------------------------------------------------------------------

    def build_scheduler(
        self,
        optimizer: optim.Optimizer,
        steps_per_epoch: Optional[int] = None,
        n_epochs: Optional[int] = None,
    ) -> Optional[Any]:
        """Construct and return the configured LR scheduler."""
        name = self.config.get("scheduler", None)
        if name is None:
            return None

        name = name.lower()
        if name == "cosine":
            T_max = int(self.config.get("T_max", n_epochs or 50))
            scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-7)
        elif name == "onecycle":
            if steps_per_epoch is None:
                steps_per_epoch = int(self.config.get("steps_per_epoch", 100))
            max_lr = float(self.config.get("max_lr", 1e-2))
            total_steps = steps_per_epoch * (n_epochs or 30)
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                anneal_strategy="cos",
            )
        elif name == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max" if "f1" in self.es_metric else "min",
                patience=5,
                factor=0.5,
                verbose=True,
            )
        else:
            raise ValueError(
                f"Unknown scheduler '{name}'. Choose from: cosine, onecycle, plateau."
            )
        logger.info("Scheduler: %s", name)
        return scheduler

    # ------------------------------------------------------------------
    # train_epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler: Optional[GradScaler] = None,
    ) -> float:
        """
        Run one training epoch.

        Parameters
        ----------
        loader : DataLoader
        criterion : loss function
        optimizer : optimiser
        scaler : GradScaler (pass ``self.scaler`` or ``None``)

        Returns
        -------
        float — mean training loss for this epoch
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            # Support (X, y) or {"features": X, "label": y} batches
            if isinstance(batch, (list, tuple)):
                X_batch, y_batch = batch[0], batch[1]
            elif isinstance(batch, dict):
                X_batch = batch["features"]
                y_batch = batch["label"]
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")

            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(X_batch)
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                loss = criterion(logits, y_batch)

            if scaler is not None and self.use_amp:
                scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # validate_epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """
        Run one validation epoch.

        Returns
        -------
        dict with keys: val_loss, val_accuracy, val_f1, val_roc_auc
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                X_batch, y_batch = batch[0], batch[1]
            elif isinstance(batch, dict):
                X_batch = batch["features"]
                y_batch = batch["label"]
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")

            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True).float()

            with autocast(enabled=self.use_amp):
                logits = self.model(X_batch)
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                loss = criterion(logits, y_batch)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y_batch.cpu().numpy())

        all_probs_arr = np.concatenate(all_probs)
        all_labels_arr = np.concatenate(all_labels)
        preds = (all_probs_arr >= 0.5).astype(int)

        metrics: Dict[str, float] = {
            "val_loss": total_loss / max(n_batches, 1),
            "val_accuracy": float(accuracy_score(all_labels_arr, preds)),
            "val_f1": float(
                f1_score(all_labels_arr, preds, zero_division=0)
            ),
        }
        try:
            metrics["val_roc_auc"] = float(roc_auc_score(all_labels_arr, all_probs_arr))
        except ValueError:
            metrics["val_roc_auc"] = float("nan")

        return metrics

    # ------------------------------------------------------------------
    # Early stopping helper
    # ------------------------------------------------------------------

    def _check_early_stopping(self, metrics: Dict[str, float]) -> Tuple[bool, bool]:
        """
        Returns (should_stop, is_improved).
        """
        current = metrics.get(self.es_metric, 0.0)
        higher_is_better = "f1" in self.es_metric or "auc" in self.es_metric

        if higher_is_better:
            improved = current > self._best_es_value
        else:
            improved = current < self._best_es_value

        if improved:
            self._best_es_value = current
            self._es_counter = 0
        else:
            self._es_counter += 1

        return self._es_counter >= self.patience, improved

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        scheduler: Optional[Any],
        tag: str = "best",
    ) -> None:
        path = self.output_dir / f"checkpoint_{tag}.pt"
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_es_value": self._best_es_value,
            "es_counter": self._es_counter,
            "history": dict(self._history),
        }
        if scheduler is not None and hasattr(scheduler, "state_dict"):
            state["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(state, str(path))
        logger.info("Checkpoint saved → %s (epoch %d)", path.name, epoch)

    def resume_from_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> int:
        """
        Load model (and optionally optimiser/scheduler) from a checkpoint.

        Returns
        -------
        int — the epoch at which training was saved
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        state = torch.load(str(path), map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state.get("scaler_state_dict", {}))
        self._best_es_value = state.get("best_es_value", self._best_es_value)
        self._es_counter = state.get("es_counter", 0)
        if "history" in state:
            self._history.update(state["history"])

        epoch = int(state.get("epoch", 0))
        logger.info("Resumed from checkpoint '%s' at epoch %d.", path.name, epoch)
        return epoch

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        criterion: nn.Module,
        resume_checkpoint: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        epochs : int — total number of epochs
        criterion : loss function
        resume_checkpoint : optional path to .pt checkpoint

        Returns
        -------
        dict — training history
        """
        optimizer = self.build_optimizer(self.model.parameters())
        scheduler = self.build_scheduler(
            optimizer,
            steps_per_epoch=len(train_loader),
            n_epochs=epochs,
        )

        start_epoch = 0
        if resume_checkpoint is not None:
            start_epoch = self.resume_from_checkpoint(
                resume_checkpoint, optimizer, scheduler
            )

        logger.info(
            "Training started | epochs=%d | start_epoch=%d | train_batches=%d | val_batches=%d",
            epochs,
            start_epoch,
            len(train_loader),
            len(val_loader),
        )

        for epoch in range(start_epoch, epochs):
            t0 = time.time()

            # --- Training ---
            train_loss = self.train_epoch(
                train_loader, criterion, optimizer, self.scaler
            )

            # --- Validation ---
            val_metrics = self.validate_epoch(val_loader, criterion)

            elapsed = time.time() - t0

            # --- Scheduler step ---
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    val_key = (
                        "val_f1" if "f1" in self.es_metric else "val_loss"
                    )
                    scheduler.step(val_metrics.get(val_key, val_metrics["val_loss"]))
                elif not isinstance(scheduler, OneCycleLR):
                    # OneCycleLR is stepped per batch inside train_epoch
                    scheduler.step()

            # --- History ---
            self._history["train_loss"].append(train_loss)
            for k, v in val_metrics.items():
                self._history[k].append(v)
            current_lr = optimizer.param_groups[0]["lr"]
            self._history["lr"].append(current_lr)

            # --- Logging ---
            log_str = (
                f"Epoch {epoch + 1:04d}/{epochs} | "
                f"train_loss={train_loss:.4f} | "
                + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                + f" | lr={current_lr:.2e} | {elapsed:.1f}s"
            )
            logger.info(log_str)

            # TensorBoard
            if self._writer is not None:
                self._writer.add_scalar("Loss/train", train_loss, epoch)
                for k, v in val_metrics.items():
                    self._writer.add_scalar(f"Metrics/{k}", v, epoch)
                self._writer.add_scalar("LR", current_lr, epoch)

            # W&B
            if self.use_wandb:
                wandb.log(
                    {"train_loss": train_loss, **val_metrics, "lr": current_lr},
                    step=epoch,
                )

            # --- Early stopping & checkpointing ---
            should_stop, improved = self._check_early_stopping(val_metrics)

            self._save_checkpoint(epoch, optimizer, scheduler, tag="latest")
            if improved:
                self._best_epoch = epoch
                self._save_checkpoint(epoch, optimizer, scheduler, tag="best")

            if should_stop:
                logger.info(
                    "Early stopping triggered at epoch %d (best epoch=%d, %s=%.4f).",
                    epoch + 1,
                    self._best_epoch + 1,
                    self.es_metric,
                    self._best_es_value,
                )
                break

        if self._writer is not None:
            self._writer.close()
        if self.use_wandb:
            wandb.finish()

        logger.info(
            "Training complete. Best epoch=%d | best %s=%.4f",
            self._best_epoch + 1,
            self.es_metric,
            self._best_es_value,
        )
        return dict(self._history)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_training_history(self) -> Dict[str, List[float]]:
        """Return the training history dict."""
        return dict(self._history)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_training_curves(self, save_path: Union[str, Path]) -> None:
        """
        Plot and save training/validation loss + metric curves.

        Parameters
        ----------
        save_path : path where the PNG is saved
        """
        if not _PLT_AVAILABLE:
            logger.error("matplotlib not installed; cannot plot training curves.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        history = self._history

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax = axes[0]
        if "train_loss" in history:
            ax.plot(history["train_loss"], label="Train Loss", color="steelblue")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Val Loss", color="orange", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Metrics (F1, accuracy, AUC)
        ax = axes[1]
        metric_keys = [k for k in history if k.startswith("val_") and k != "val_loss"]
        colours = ["green", "red", "purple", "brown"]
        for colour, key in zip(colours, metric_keys):
            ax.plot(history[key], label=key, color=colour, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Validation Metrics")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("Training curves saved → %s", save_path)

    def __repr__(self) -> str:
        return (
            f"Trainer(device={self.device}, amp={self.use_amp}, "
            f"grad_clip={self.grad_clip}, patience={self.patience}, "
            f"metric={self.es_metric})"
        )
