"""
ensemble_models.py
==================
Production-quality ensemble models for the theft detection system.

Supported ensemble strategies
------------------------------
1. XGBoost + CNN-LSTM weighted average
2. XGBoost + Transformer weighted average
3. Stacking (CV meta-features → meta-learner)
4. Soft-voting (average probabilities from any list of models)
5. Weighted ensemble (scipy-optimised weights on a validation set)

Usage
-----
>>> from ensemble_models import EnsembleModels
>>> em = EnsembleModels(config)
>>> em.train_weighted_ensemble(models, X_val, y_val)
>>> probs = em.predict_weighted(models, X_test)
>>> df   = em.compare_ensembles(X_test, y_test)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------------------
# Optional XGBoost import
# ---------------------------------------------------------------------------
try:
    import xgboost as xgb  # type: ignore

    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _to_numpy(x: Any) -> np.ndarray:
    """Convert a tensor or array-like to a 2-D float32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr


def _get_proba(
    model: Any,
    X: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Return a 1-D probability array (positive-class) for *any* model type:
    - sklearn / XGBoost estimators that expose ``predict_proba``
    - PyTorch nn.Module models (must accept a float32 tensor, return logits)
    """
    if hasattr(model, "predict_proba"):
        # sklearn / XGBoost path
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    if isinstance(model, nn.Module):
        if device is None:
            device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = model(tensor)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            proba = torch.sigmoid(logits).cpu().numpy()
        return proba.astype(np.float32)

    raise TypeError(
        f"Cannot extract probabilities from model of type {type(model).__name__}. "
        "Model must expose predict_proba() or be an nn.Module."
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class EnsembleModels:
    """
    Ensemble strategies for binary classification (theft detection).

    Parameters
    ----------
    config : dict
        Configuration dict.  Recognised keys:
        - ``device`` : str or torch.device (default ``"cpu"``)
        - ``n_splits`` : int — number of CV folds for stacking (default 5)
        - ``meta_learner_C`` : float — regularisation for stacking LR (default 1.0)
        - ``random_state`` : int (default 42)
        - ``xgb_params`` : dict — XGBoost parameters for stacking meta-learner
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

        raw_device = self.config.get("device", "cpu")
        self.device: torch.device = (
            raw_device
            if isinstance(raw_device, torch.device)
            else torch.device(raw_device)
        )

        self.n_splits: int = int(self.config.get("n_splits", 5))
        self.random_state: int = int(self.config.get("random_state", 42))
        self.meta_learner_C: float = float(self.config.get("meta_learner_C", 1.0))

        # Learned artefacts
        self._stacking_meta_learner: Optional[Any] = None
        self._optimised_weights: Optional[np.ndarray] = None

        logger.info(
            "EnsembleModels initialised | device=%s | n_splits=%d",
            self.device,
            self.n_splits,
        )

    # ------------------------------------------------------------------
    # 1. XGBoost + CNN-LSTM weighted average
    # ------------------------------------------------------------------

    def predict_xgb_cnn_lstm(
        self,
        xgb_model: Any,
        cnn_lstm_model: nn.Module,
        X: np.ndarray,
        xgb_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Weighted average of XGBoost and CNN-LSTM probabilities.

        Parameters
        ----------
        xgb_model : XGBoost Booster or sklearn-compatible model
        cnn_lstm_model : nn.Module — CNN-LSTM PyTorch model
        X : np.ndarray of shape (N, features)
        xgb_weight : float in [0, 1]; CNN-LSTM weight = 1 - xgb_weight

        Returns
        -------
        np.ndarray of shape (N,) — blended probabilities
        """
        if not 0.0 <= xgb_weight <= 1.0:
            raise ValueError(f"xgb_weight must be in [0, 1], got {xgb_weight}")

        logger.info(
            "XGBoost+CNN-LSTM ensemble | xgb_weight=%.3f | samples=%d",
            xgb_weight,
            len(X),
        )

        p_xgb = _get_proba(xgb_model, X, self.device)
        p_cnn = _get_proba(cnn_lstm_model, X, self.device)

        blend = xgb_weight * p_xgb + (1.0 - xgb_weight) * p_cnn
        logger.debug("Blend stats — mean=%.4f  std=%.4f", blend.mean(), blend.std())
        return blend.astype(np.float32)

    # ------------------------------------------------------------------
    # 2. XGBoost + Transformer weighted average
    # ------------------------------------------------------------------

    def predict_xgb_transformer(
        self,
        xgb_model: Any,
        transformer_model: nn.Module,
        X: np.ndarray,
        xgb_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Weighted average of XGBoost and Transformer probabilities.

        Parameters
        ----------
        xgb_model : XGBoost Booster or sklearn-compatible model
        transformer_model : nn.Module — Transformer PyTorch model
        X : np.ndarray of shape (N, features)
        xgb_weight : float in [0, 1]

        Returns
        -------
        np.ndarray of shape (N,) — blended probabilities
        """
        if not 0.0 <= xgb_weight <= 1.0:
            raise ValueError(f"xgb_weight must be in [0, 1], got {xgb_weight}")

        logger.info(
            "XGBoost+Transformer ensemble | xgb_weight=%.3f | samples=%d",
            xgb_weight,
            len(X),
        )

        p_xgb = _get_proba(xgb_model, X, self.device)
        p_tfm = _get_proba(transformer_model, X, self.device)

        blend = xgb_weight * p_xgb + (1.0 - xgb_weight) * p_tfm
        logger.debug("Blend stats — mean=%.4f  std=%.4f", blend.mean(), blend.std())
        return blend.astype(np.float32)

    # ------------------------------------------------------------------
    # 3. Stacking ensemble
    # ------------------------------------------------------------------

    def train_stacking(
        self,
        base_models: List[Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        meta_learner: Optional[Any] = None,
    ) -> None:
        """
        Train a stacking ensemble.

        Cross-validated out-of-fold predictions from ``base_models`` are used
        as meta-features to train ``meta_learner``.

        Parameters
        ----------
        base_models : list of fitted or unfitted estimators
            Each model must be clonable (sklearn-style ``set_params``) or be
            an nn.Module.  For nn.Module models only inference is performed
            on held-out folds — the model is assumed already trained.
        X_train : np.ndarray (N, F)
        y_train : np.ndarray (N,)
        meta_learner : optional sklearn estimator (default: LogisticRegression)
        """
        logger.info(
            "Stacking — generating OOF meta-features | n_models=%d | n_folds=%d",
            len(base_models),
            self.n_splits,
        )

        n_samples = len(X_train)
        n_models = len(base_models)
        meta_X = np.zeros((n_samples, n_models), dtype=np.float32)

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_train, y_train), start=1
        ):
            logger.info("Stacking fold %d/%d", fold_idx, self.n_splits)
            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
            y_tr = y_train[train_idx]

            for m_idx, model in enumerate(base_models):
                try:
                    # Try to fit sklearn-style models
                    if hasattr(model, "fit") and not isinstance(model, nn.Module):
                        model.fit(X_tr, y_tr)
                    # nn.Module — already trained; use as-is on val fold
                    oof_prob = _get_proba(model, X_vl, self.device)
                    meta_X[val_idx, m_idx] = oof_prob
                except Exception as exc:
                    logger.error(
                        "Error in fold %d, model %d (%s): %s",
                        fold_idx,
                        m_idx,
                        type(model).__name__,
                        exc,
                    )
                    raise

        # Train meta-learner
        if meta_learner is None:
            meta_learner = LogisticRegression(
                C=self.meta_learner_C, max_iter=1000, random_state=self.random_state
            )

        logger.info(
            "Stacking — training meta-learner (%s)", type(meta_learner).__name__
        )
        meta_learner.fit(meta_X, y_train)
        self._stacking_meta_learner = meta_learner
        logger.info("Stacking meta-learner trained successfully.")

    def predict_stacking(
        self,
        base_models: List[Any],
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Generate predictions using the trained stacking ensemble.

        Parameters
        ----------
        base_models : list of fitted models
        X : np.ndarray (N, F)

        Returns
        -------
        np.ndarray of shape (N,) — probabilities
        """
        if self._stacking_meta_learner is None:
            raise RuntimeError(
                "Stacking meta-learner is not trained. Call train_stacking() first."
            )

        n_models = len(base_models)
        meta_X = np.zeros((len(X), n_models), dtype=np.float32)

        for m_idx, model in enumerate(base_models):
            meta_X[:, m_idx] = _get_proba(model, X, self.device)

        proba = self._stacking_meta_learner.predict_proba(meta_X)[:, 1]
        return proba.astype(np.float32)

    # ------------------------------------------------------------------
    # 4. Soft voting ensemble
    # ------------------------------------------------------------------

    def predict_soft_voting(
        self,
        models: List[Any],
        X: np.ndarray,
        weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """
        Soft voting: (weighted) average of model probabilities.

        Parameters
        ----------
        models : list of fitted models
        X : np.ndarray (N, F)
        weights : optional list of floats (uniform if None)

        Returns
        -------
        np.ndarray of shape (N,) — averaged probabilities
        """
        n = len(models)
        if weights is None:
            weights = [1.0 / n] * n
        weights_arr = np.asarray(weights, dtype=np.float64)
        if len(weights_arr) != n:
            raise ValueError(
                f"Length of weights ({len(weights_arr)}) must match "
                f"number of models ({n})."
            )
        weights_arr = weights_arr / weights_arr.sum()  # normalise

        logger.info(
            "Soft voting | n_models=%d | weights=%s | samples=%d",
            n,
            np.round(weights_arr, 4).tolist(),
            len(X),
        )

        proba_stack = np.zeros((len(X), n), dtype=np.float32)
        for m_idx, model in enumerate(models):
            proba_stack[:, m_idx] = _get_proba(model, X, self.device)

        blended = (proba_stack * weights_arr[None, :]).sum(axis=1)
        return blended.astype(np.float32)

    # ------------------------------------------------------------------
    # 5. Weighted ensemble with scipy.optimize
    # ------------------------------------------------------------------

    def train_weighted_ensemble(
        self,
        models: List[Any],
        X_val: np.ndarray,
        y_val: np.ndarray,
        optimise_metric: str = "f1",
    ) -> np.ndarray:
        """
        Learn optimal weights by minimising a loss on the validation set via
        ``scipy.optimize.minimize`` (Nelder-Mead).

        Parameters
        ----------
        models : list of fitted models
        X_val : np.ndarray (N, F)
        y_val : np.ndarray (N,)
        optimise_metric : ``"f1"`` or ``"auc"``

        Returns
        -------
        np.ndarray — optimised weights (sums to 1)
        """
        logger.info(
            "Weighted ensemble optimisation | n_models=%d | metric=%s | val_samples=%d",
            len(models),
            optimise_metric,
            len(X_val),
        )

        n = len(models)
        # Pre-compute per-model probabilities on val set
        proba_matrix = np.column_stack(
            [_get_proba(m, X_val, self.device) for m in models]
        ).astype(np.float64)

        def objective(raw_weights: np.ndarray) -> float:
            # Softmax to ensure weights are positive and sum to 1
            exp_w = np.exp(raw_weights - raw_weights.max())
            w = exp_w / exp_w.sum()
            blended = (proba_matrix * w[None, :]).sum(axis=1)
            if optimise_metric == "auc":
                score = roc_auc_score(y_val, blended)
            else:
                preds = (blended >= 0.5).astype(int)
                score = f1_score(y_val, preds, zero_division=0)
            return -score  # minimise

        x0 = np.ones(n, dtype=np.float64) / n
        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
        )

        exp_w = np.exp(result.x - result.x.max())
        optimised = (exp_w / exp_w.sum()).astype(np.float32)
        self._optimised_weights = optimised

        logger.info(
            "Optimised weights: %s | best_score=%.4f",
            np.round(optimised, 4).tolist(),
            -result.fun,
        )
        return optimised

    def predict_weighted(
        self,
        models: List[Any],
        X: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict using pre-optimised (or supplied) weights.

        Parameters
        ----------
        models : list of fitted models
        X : np.ndarray (N, F)
        weights : optional override weights; uses ``_optimised_weights`` if None

        Returns
        -------
        np.ndarray of shape (N,) — blended probabilities
        """
        if weights is None:
            if self._optimised_weights is None:
                raise RuntimeError(
                    "Weights not available. Call train_weighted_ensemble() first "
                    "or supply weights explicitly."
                )
            weights = self._optimised_weights

        return self.predict_soft_voting(models, X, weights=weights)

    # ------------------------------------------------------------------
    # 6. Compare all ensembles
    # ------------------------------------------------------------------

    def compare_ensembles(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        ensemble_predictions: Optional[Dict[str, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Evaluate a dictionary of ensemble predictions and return a comparison
        ``DataFrame`` sorted by F1 score descending.

        Parameters
        ----------
        X_test : np.ndarray — held-out feature matrix
        y_test : np.ndarray — ground-truth labels
        ensemble_predictions : dict mapping ensemble_name → probability array
            If ``None`` an empty DataFrame is returned (caller must supply preds).

        Returns
        -------
        pd.DataFrame with columns:
            ensemble, accuracy, f1, roc_auc, threshold_used
        """
        if ensemble_predictions is None or len(ensemble_predictions) == 0:
            logger.warning(
                "compare_ensembles called with no predictions dict. "
                "Pass ensemble_predictions={name: prob_array, ...}."
            )
            return pd.DataFrame(
                columns=["ensemble", "accuracy", "f1", "roc_auc", "threshold_used"]
            )

        records: List[Dict[str, Any]] = []

        for name, probs in ensemble_predictions.items():
            probs = np.asarray(probs, dtype=np.float32)
            try:
                # Find best threshold via F1 grid search
                best_thresh, best_f1 = 0.5, 0.0
                for thr in np.linspace(0.1, 0.9, 81):
                    preds_tmp = (probs >= thr).astype(int)
                    f1_tmp = f1_score(y_test, preds_tmp, zero_division=0)
                    if f1_tmp > best_f1:
                        best_f1 = f1_tmp
                        best_thresh = round(float(thr), 3)

                preds = (probs >= best_thresh).astype(int)
                acc = accuracy_score(y_test, preds)
                try:
                    auc = roc_auc_score(y_test, probs)
                except ValueError:
                    auc = float("nan")

                records.append(
                    {
                        "ensemble": name,
                        "accuracy": round(float(acc), 4),
                        "f1": round(float(best_f1), 4),
                        "roc_auc": round(float(auc), 4),
                        "threshold_used": best_thresh,
                    }
                )
                logger.info(
                    "%-30s  acc=%.4f  f1=%.4f  auc=%.4f  thr=%.3f",
                    name,
                    acc,
                    best_f1,
                    auc,
                    best_thresh,
                )
            except Exception as exc:
                logger.error("Error evaluating ensemble '%s': %s", name, exc)

        df = (
            pd.DataFrame(records)
            .sort_values(["f1", "roc_auc"], ascending=False)
            .reset_index(drop=True)
        )
        return df

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def get_optimised_weights(self) -> Optional[np.ndarray]:
        """Return the weights learned by ``train_weighted_ensemble``."""
        return self._optimised_weights

    def get_stacking_meta_learner(self) -> Optional[Any]:
        """Return the trained stacking meta-learner."""
        return self._stacking_meta_learner

    def __repr__(self) -> str:
        stacking_trained = self._stacking_meta_learner is not None
        weights_trained = self._optimised_weights is not None
        return (
            f"EnsembleModels("
            f"device={self.device}, "
            f"n_splits={self.n_splits}, "
            f"stacking_trained={stacking_trained}, "
            f"weights_optimised={weights_trained})"
        )
