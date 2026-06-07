"""
hyperparameter_tuning.py
========================
Optuna-based hyperparameter tuning for the theft detection system.

Usage
-----
>>> tuner = HyperparameterTuner(
...     model_class=MyCNNLSTM,
...     X_train=X_tr, y_train=y_tr,
...     X_val=X_vl,   y_val=y_vl,
...     config=cfg,   device=device,
...     n_trials=50,  direction="maximize",
... )
>>> study = tuner.tune()
>>> best  = tuner.get_best_params()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Optuna
# ---------------------------------------------------------------------------
try:
    import optuna
    from optuna import Study, Trial
    from optuna.samplers import TPESampler

    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    raise ImportError(
        "optuna is required for HyperparameterTuner. "
        "Install it with: pip install optuna"
    )

# ---------------------------------------------------------------------------
# Optional matplotlib
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import optuna.visualization.matplotlib as optuna_vis

    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------------------------------------------------------
# Internal training helpers
# ---------------------------------------------------------------------------


def _make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Wrap numpy arrays in a TensorDataset DataLoader."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    """Train for one epoch. Returns mean loss."""
    model.train()
    total, n = 0.0, 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        loss = criterion(logits, y_batch)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate. Returns (val_loss, val_f1)."""
    model.eval()
    total, n = 0.0, 0
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        logits = model(X_batch)
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        loss = criterion(logits, y_batch)
        total += loss.item()
        n += 1
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y_batch.cpu().numpy())
    all_probs_arr = np.concatenate(all_probs)
    all_labels_arr = np.concatenate(all_labels)
    preds = (all_probs_arr >= 0.5).astype(int)
    val_f1 = float(f1_score(all_labels_arr, preds, zero_division=0))
    val_loss = total / max(n, 1)
    return val_loss, val_f1


# ===========================================================================
# Main class
# ===========================================================================


class HyperparameterTuner:
    """
    Optuna-based hyperparameter search for any PyTorch model.

    Parameters
    ----------
    model_class : type
        Callable that builds an ``nn.Module``.  It is called as::

            model_class(
                input_dim=...,
                hidden_dim=trial_hidden_dim,
                num_layers=trial_num_layers,
                dropout=trial_dropout,
                **config.get("model_kwargs", {}),
            )

    X_train, y_train : np.ndarray — training data
    X_val,   y_val   : np.ndarray — validation data
    config : dict
        Optional search-space overrides.  Recognised keys:

        =====================  ==============================================
        Key                    Default / Description
        =====================  ==============================================
        lr_low / lr_high       1e-5 / 1e-2
        batch_sizes            [32, 64, 128, 256]
        hidden_dims            [64, 128, 256, 512]
        num_layers_low/high    1 / 4
        dropout_low/high       0.1 / 0.5
        wd_low / wd_high       1e-6 / 1e-2
        optimizers             ["adam", "adamw", "sgd", "ranger"]
        schedulers             ["cosine", "plateau", None]
        max_epochs             30
        early_stopping_patience 5
        grad_clip              1.0
        =====================  ==============================================

    device : torch.device
    n_trials : int (default 50)
    direction : ``"maximize"`` or ``"minimize"`` (default ``"maximize"``)
    study_name : str (default ``"theft_detection_hpo"``)
    """

    def __init__(
        self,
        model_class: Type[nn.Module],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        n_trials: int = 50,
        direction: str = "maximize",
        study_name: str = "theft_detection_hpo",
    ) -> None:
        if not _OPTUNA_AVAILABLE:
            raise ImportError("optuna is not installed.")

        self.model_class = model_class
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.X_val = np.asarray(X_val, dtype=np.float32)
        self.y_val = np.asarray(y_val, dtype=np.float32)
        self.config: Dict[str, Any] = config or {}
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.n_trials = n_trials
        self.direction = direction
        self.study_name = study_name

        # Derived
        self._input_dim: int = self.X_train.shape[1]
        self._study: Optional[Study] = None

        logger.info(
            "HyperparameterTuner ready | model=%s | n_trials=%d | direction=%s | device=%s",
            model_class.__name__,
            n_trials,
            direction,
            self.device,
        )

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    def objective(self, trial: "Trial") -> float:
        """
        Optuna objective function.

        Samples hyperparameters, trains the model for up to ``max_epochs``
        with early stopping, and returns the validation F1 score.
        """
        cfg = self.config

        # --- Sample hyperparameters ---
        lr = trial.suggest_float(
            "lr",
            float(cfg.get("lr_low", 1e-5)),
            float(cfg.get("lr_high", 1e-2)),
            log=True,
        )
        batch_size = trial.suggest_categorical(
            "batch_size",
            cfg.get("batch_sizes", [32, 64, 128, 256]),
        )
        hidden_dim = trial.suggest_categorical(
            "hidden_dim",
            cfg.get("hidden_dims", [64, 128, 256, 512]),
        )
        num_layers = trial.suggest_int(
            "num_layers",
            int(cfg.get("num_layers_low", 1)),
            int(cfg.get("num_layers_high", 4)),
        )
        dropout = trial.suggest_float(
            "dropout",
            float(cfg.get("dropout_low", 0.1)),
            float(cfg.get("dropout_high", 0.5)),
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            float(cfg.get("wd_low", 1e-6)),
            float(cfg.get("wd_high", 1e-2)),
            log=True,
        )
        optimizer_name = trial.suggest_categorical(
            "optimizer",
            cfg.get("optimizers", ["adam", "adamw", "sgd", "ranger"]),
        )
        scheduler_name = trial.suggest_categorical(
            "scheduler",
            cfg.get("schedulers", ["cosine", "plateau", None]),
        )

        # --- Build model ---
        try:
            model = self.model_class(
                input_dim=self._input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                **cfg.get("model_kwargs", {}),
            ).to(self.device)
        except TypeError as exc:
            logger.warning(
                "Model construction failed (trial %d): %s. Trying minimal kwargs.",
                trial.number,
                exc,
            )
            try:
                model = self.model_class(
                    input_dim=self._input_dim,
                ).to(self.device)
            except Exception as exc2:
                logger.error("Model construction failed entirely: %s", exc2)
                raise optuna.exceptions.TrialPruned()

        # --- Build data loaders ---
        train_loader = _make_loader(self.X_train, self.y_train, batch_size, shuffle=True)
        val_loader = _make_loader(self.X_val, self.y_val, batch_size, shuffle=False)

        # --- Build optimiser ---
        opt_kwargs: Dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
        if optimizer_name == "sgd":
            opt_kwargs["momentum"] = 0.9
            optimizer = torch.optim.SGD(model.parameters(), **opt_kwargs)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), **opt_kwargs)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)
        elif optimizer_name == "ranger":
            from trainer import build_ranger

            optimizer = build_ranger(  # type: ignore[attr-defined]
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), **opt_kwargs)

        # --- Build scheduler ---
        max_epochs: int = int(cfg.get("max_epochs", 30))
        scheduler: Optional[Any] = None
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-7
            )
        elif scheduler_name == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", patience=3, factor=0.5
            )

        criterion = nn.BCEWithLogitsLoss()
        patience: int = int(cfg.get("early_stopping_patience", 5))
        grad_clip: float = float(cfg.get("grad_clip", 1.0))

        best_val_f1 = 0.0
        no_improve = 0

        for epoch in range(max_epochs):
            _train_one_epoch(
                model, train_loader, criterion, optimizer, self.device, grad_clip
            )
            val_loss, val_f1 = _validate(model, val_loader, criterion, self.device)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_f1)
                else:
                    scheduler.step()

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve = 0
            else:
                no_improve += 1

            # Optuna pruning
            trial.report(val_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if no_improve >= patience:
                break

        logger.debug(
            "Trial %d done | best_val_f1=%.4f | params=%s",
            trial.number,
            best_val_f1,
            trial.params,
        )
        return best_val_f1

    # ------------------------------------------------------------------
    # Run the study
    # ------------------------------------------------------------------

    def tune(self) -> "Study":
        """
        Run the Optuna study and return it.

        Returns
        -------
        optuna.Study
        """
        logger.info(
            "Starting Optuna study '%s' | n_trials=%d | direction=%s",
            self.study_name,
            self.n_trials,
            self.direction,
        )

        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=5, interval_steps=1
        )
        sampler = TPESampler(seed=42)

        study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=sampler,
            pruner=pruner,
        )

        t0 = time.time()
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=None,
            show_progress_bar=True,
            catch=(RuntimeError, ValueError),
        )
        elapsed = time.time() - t0

        self._study = study
        logger.info(
            "Optuna study complete | n_trials=%d | elapsed=%.1fs | "
            "best_trial=%d | best_value=%.4f",
            len(study.trials),
            elapsed,
            study.best_trial.number,
            study.best_value,
        )
        return study

    # ------------------------------------------------------------------
    # Result accessors
    # ------------------------------------------------------------------

    def get_best_params(self) -> Dict[str, Any]:
        """Return the best hyperparameters found."""
        if self._study is None:
            raise RuntimeError("Run tune() first.")
        params = self._study.best_params.copy()
        logger.info("Best params: %s", params)
        return params

    def get_best_trial(self) -> "Trial":
        """Return the best Optuna Trial object."""
        if self._study is None:
            raise RuntimeError("Run tune() first.")
        return self._study.best_trial

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_optimization_history(
        self, save_path: Union[str, Path]
    ) -> None:
        """
        Plot and save the optimisation history (objective vs trial number).

        Parameters
        ----------
        save_path : str or Path
        """
        if self._study is None:
            raise RuntimeError("Run tune() first.")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if _PLT_AVAILABLE:
            try:
                ax = optuna_vis.plot_optimization_history(self._study)
                fig = ax.get_figure()
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("Optimisation history saved → %s", save_path)
                return
            except Exception as exc:
                logger.warning("optuna_vis failed (%s); falling back to manual plot.", exc)

            # Manual fallback
            values = [
                t.value
                for t in self._study.trials
                if t.value is not None
            ]
            best_so_far: List[float] = []
            cur_best = -float("inf") if self.direction == "maximize" else float("inf")
            for v in values:
                if self.direction == "maximize":
                    cur_best = max(cur_best, v)
                else:
                    cur_best = min(cur_best, v)
                best_so_far.append(cur_best)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.scatter(range(len(values)), values, alpha=0.5, label="Trial value", s=20)
            ax.plot(range(len(best_so_far)), best_so_far, color="red", label="Best so far")
            ax.set_xlabel("Trial")
            ax.set_ylabel("Objective (Val F1)")
            ax.set_title("Optuna Optimisation History")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fig.savefig(str(save_path), dpi=150)
            plt.close(fig)
            logger.info("Optimisation history (manual) saved → %s", save_path)
        else:
            logger.error("matplotlib not installed; cannot plot optimisation history.")

    def plot_param_importances(
        self, save_path: Union[str, Path]
    ) -> None:
        """
        Plot and save hyperparameter importances.

        Parameters
        ----------
        save_path : str or Path
        """
        if self._study is None:
            raise RuntimeError("Run tune() first.")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if _PLT_AVAILABLE:
            try:
                ax = optuna_vis.plot_param_importances(self._study)
                fig = ax.get_figure()
                fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info("Param importances saved → %s", save_path)
                return
            except Exception as exc:
                logger.warning(
                    "optuna_vis.plot_param_importances failed (%s); "
                    "falling back to manual importance.",
                    exc,
                )
            # Manual fallback: use fANOVA importances if available
            try:
                importances = optuna.importance.get_param_importances(self._study)
                params = list(importances.keys())
                scores = list(importances.values())
                fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.5)))
                ax.barh(params, scores, color="steelblue")
                ax.set_xlabel("Importance")
                ax.set_title("Hyperparameter Importances")
                ax.grid(True, alpha=0.3, axis="x")
                plt.tight_layout()
                fig.savefig(str(save_path), dpi=150)
                plt.close(fig)
                logger.info("Param importances (manual) saved → %s", save_path)
            except Exception as exc2:
                logger.error("Could not compute param importances: %s", exc2)
        else:
            logger.error("matplotlib not installed; cannot plot param importances.")

    def __repr__(self) -> str:
        study_info = (
            f"best_value={self._study.best_value:.4f}" if self._study else "not run"
        )
        return (
            f"HyperparameterTuner("
            f"model={self.model_class.__name__}, "
            f"n_trials={self.n_trials}, "
            f"direction={self.direction}, "
            f"study={study_info})"
        )
