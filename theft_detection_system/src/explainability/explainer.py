"""
explainer.py
============
Production-quality model explainability for the theft detection system.

Provides the ``Explainer`` class which supports:
- SHAP explanations (tree / linear / deep / kernel)
- Permutation feature importance
- sklearn-style feature importance extraction
- Attention weight visualisation (Transformer models)
- Single-prediction narrative reports
- Batch explanation with summary statistics
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance

# ---------------------------------------------------------------------------
# Optional SHAP
# ---------------------------------------------------------------------------
try:
    import shap  # type: ignore

    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional matplotlib
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _model_predict_fn(
    model: Any,
    device: torch.device,
) -> Any:
    """
    Return a callable suitable for SHAP KernelExplainer from an nn.Module.

    The returned function accepts a 2-D float32 numpy array and returns a
    1-D probability array.
    """

    def predict(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = model(tensor)
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

    return predict


# ===========================================================================
# Main class
# ===========================================================================


class Explainer:
    """
    Model-agnostic explainability toolkit.

    Parameters
    ----------
    config : dict, optional
        Recognised keys:
        - ``device`` : str or torch.device (default ``"cpu"``)
        - ``shap_max_display`` : int — max features in SHAP plots (default 20)
        - ``background_samples`` : int — n background samples for KernelSHAP (default 100)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        raw_device = self.config.get("device", "cpu")
        self.device: torch.device = (
            raw_device
            if isinstance(raw_device, torch.device)
            else torch.device(raw_device)
        )
        self.shap_max_display: int = int(self.config.get("shap_max_display", 20))
        self.background_samples: int = int(self.config.get("background_samples", 100))
        logger.info(
            "Explainer initialised | device=%s | max_display=%d",
            self.device,
            self.shap_max_display,
        )

    # ------------------------------------------------------------------
    # SHAP explanations
    # ------------------------------------------------------------------

    def explain_with_shap(
        self,
        model: Any,
        X_background: np.ndarray,
        X_explain: np.ndarray,
        model_type: Literal["tree", "linear", "deep", "kernel"] = "tree",
    ) -> np.ndarray:
        """
        Compute SHAP values for ``X_explain``.

        Parameters
        ----------
        model : trained model
        X_background : np.ndarray (M, F) — background / reference dataset
        X_explain : np.ndarray (N, F) — samples to explain
        model_type : ``"tree"``, ``"linear"``, ``"deep"``, or ``"kernel"``

        Returns
        -------
        np.ndarray of shape (N, F) — SHAP values (positive class)
        """
        if not _SHAP_AVAILABLE:
            raise ImportError(
                "shap is not installed. Install it with: pip install shap"
            )

        X_background = np.asarray(X_background, dtype=np.float32)
        X_explain = np.asarray(X_explain, dtype=np.float32)

        # Subsample background if large
        if len(X_background) > self.background_samples:
            idx = np.random.choice(
                len(X_background), self.background_samples, replace=False
            )
            X_background = X_background[idx]

        logger.info(
            "SHAP explain | type=%s | background=%d | explain=%d",
            model_type,
            len(X_background),
            len(X_explain),
        )

        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
            # XGBoost / sklearn trees may return list for binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        elif model_type == "linear":
            explainer = shap.LinearExplainer(model, X_background)
            shap_values = explainer.shap_values(X_explain)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

        elif model_type == "deep":
            if not isinstance(model, nn.Module):
                raise TypeError(
                    "deep SHAP requires an nn.Module. "
                    f"Got {type(model).__name__}."
                )
            model.eval()
            background_t = torch.tensor(X_background, dtype=torch.float32).to(
                self.device
            )
            explain_t = torch.tensor(X_explain, dtype=torch.float32).to(self.device)
            explainer = shap.DeepExplainer(model, background_t)
            shap_values = explainer.shap_values(explain_t)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            if isinstance(shap_values, torch.Tensor):
                shap_values = shap_values.cpu().numpy()

        elif model_type == "kernel":
            predict_fn = (
                _model_predict_fn(model, self.device)
                if isinstance(model, nn.Module)
                else (
                    model.predict_proba
                    if hasattr(model, "predict_proba")
                    else model
                )
            )
            # Wrap predict_proba to return only positive class
            if hasattr(model, "predict_proba") and not isinstance(model, nn.Module):
                def predict_fn_1d(X: np.ndarray) -> np.ndarray:
                    return model.predict_proba(X)[:, 1]
                predict_fn = predict_fn_1d

            explainer = shap.KernelExplainer(predict_fn, X_background)
            shap_values = explainer.shap_values(X_explain, nsamples="auto")
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

        else:
            raise ValueError(
                f"model_type must be 'tree', 'linear', 'deep', or 'kernel'. "
                f"Got '{model_type}'."
            )

        shap_arr = np.asarray(shap_values, dtype=np.float32)
        logger.info(
            "SHAP values computed | shape=%s | mean_abs=%.4f",
            shap_arr.shape,
            float(np.abs(shap_arr).mean()),
        )
        return shap_arr

    # ------------------------------------------------------------------
    # SHAP summary plot
    # ------------------------------------------------------------------

    def plot_shap_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        save_path: Union[str, Path],
    ) -> None:
        """
        Generate and save a SHAP beeswarm / summary plot.

        Parameters
        ----------
        shap_values : np.ndarray (N, F)
        feature_names : list of str, length F
        save_path : path where PNG is saved
        """
        if not _SHAP_AVAILABLE:
            logger.error("shap not installed; cannot plot SHAP summary.")
            return
        if not _PLT_AVAILABLE:
            logger.error("matplotlib not installed; cannot plot SHAP summary.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(
            figsize=(10, max(5, min(len(feature_names) * 0.4, 20)))
        )
        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=self.shap_max_display,
            show=False,
            plot_type="dot",
        )
        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP summary plot saved → %s", save_path)

    # ------------------------------------------------------------------
    # SHAP waterfall plot (single prediction)
    # ------------------------------------------------------------------

    def plot_shap_waterfall(
        self,
        shap_values: np.ndarray,
        idx: int,
        feature_names: List[str],
        save_path: Union[str, Path],
    ) -> None:
        """
        Save a SHAP waterfall plot for a single prediction.

        Parameters
        ----------
        shap_values : np.ndarray (N, F)
        idx : int — which sample to explain
        feature_names : list of str, length F
        save_path : path where PNG is saved
        """
        if not _SHAP_AVAILABLE:
            logger.error("shap not installed; cannot plot SHAP waterfall.")
            return
        if not _PLT_AVAILABLE:
            logger.error("matplotlib not installed; cannot plot SHAP waterfall.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        sv_single = shap_values[idx]
        base_value = 0.0  # Unknown without explainer; use 0

        try:
            exp_obj = shap.Explanation(
                values=sv_single,
                base_values=base_value,
                feature_names=feature_names,
            )
            shap.waterfall_plot(exp_obj, max_display=self.shap_max_display, show=False)
        except Exception:
            # Fallback: bar plot of SHAP values for this sample
            sorted_idx = np.argsort(np.abs(sv_single))[::-1][: self.shap_max_display]
            fig, ax = plt.subplots(figsize=(10, max(4, len(sorted_idx) * 0.4)))
            colors = ["#d62728" if v > 0 else "#1f77b4" for v in sv_single[sorted_idx]]
            ax.barh(
                [feature_names[i] for i in sorted_idx[::-1]],
                sv_single[sorted_idx[::-1]],
                color=colors[::-1],
            )
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("SHAP value")
            ax.set_title(f"SHAP Waterfall — sample {idx}")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info("SHAP waterfall (fallback bar) saved → %s", save_path)
            return

        plt.tight_layout()
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
        plt.close("all")
        logger.info("SHAP waterfall plot saved → %s (sample %d)", save_path, idx)

    # ------------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------------

    def compute_permutation_importance(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Compute permutation feature importance on a validation set.

        Works with sklearn-compatible models.  For nn.Module models a
        lightweight sklearn wrapper is generated internally.

        Parameters
        ----------
        model : fitted model
        X_val : np.ndarray (N, F)
        y_val : np.ndarray (N,)
        feature_names : list of str, length F
        n_repeats : int (default 10)
        random_state : int (default 42)

        Returns
        -------
        pd.DataFrame with columns: feature, importance_mean, importance_std
            sorted by importance_mean descending.
        """
        logger.info(
            "Permutation importance | n_repeats=%d | features=%d | samples=%d",
            n_repeats,
            len(feature_names),
            len(X_val),
        )
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=int)

        if isinstance(model, nn.Module):
            # Wrap in a sklearn-compatible scorer
            class _TorchWrapper:
                """Sklearn-compatible wrapper for PyTorch nn.Module."""

                def __init__(self, mod: nn.Module, dev: torch.device) -> None:
                    self._mod = mod
                    self._dev = dev

                def predict(self, X: np.ndarray) -> np.ndarray:
                    self._mod.eval()
                    with torch.no_grad():
                        t = torch.tensor(X, dtype=torch.float32).to(self._dev)
                        logits = self._mod(t)
                        if logits.ndim == 2 and logits.shape[1] == 1:
                            logits = logits.squeeze(1)
                        return (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()

                def score(self, X: np.ndarray, y: np.ndarray) -> float:
                    from sklearn.metrics import f1_score as _f1

                    preds = self.predict(X)
                    return float(_f1(y, preds, zero_division=0))

            sklearn_model = _TorchWrapper(model, self.device)
        else:
            sklearn_model = model

        result = permutation_importance(
            sklearn_model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="f1",
        )

        df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        logger.info("Permutation importance computed | top feature: %s", df.iloc[0]["feature"])
        return df

    # ------------------------------------------------------------------
    # Plot permutation importance
    # ------------------------------------------------------------------

    def plot_permutation_importance(
        self,
        importance_df: pd.DataFrame,
        save_path: Union[str, Path],
        top_n: int = 20,
    ) -> None:
        """
        Plot and save a horizontal bar chart of permutation feature importance.

        Parameters
        ----------
        importance_df : pd.DataFrame from ``compute_permutation_importance``
        save_path : path where PNG is saved
        top_n : int — how many top features to display (default 20)
        """
        if not _PLT_AVAILABLE:
            logger.error("matplotlib not installed; cannot plot permutation importance.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        df_top = importance_df.head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(4, len(df_top) * 0.45)))
        ax.barh(
            df_top["feature"][::-1],
            df_top["importance_mean"][::-1],
            xerr=df_top["importance_std"][::-1],
            color="steelblue",
            ecolor="black",
            capsize=3,
        )
        ax.set_xlabel("Permutation Importance (F1 decrease)", fontsize=12)
        ax.set_title(f"Top-{top_n} Permutation Feature Importance", fontsize=13)
        ax.grid(True, alpha=0.3, axis="x")
        ax.axvline(0, color="black", lw=0.8)
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("Permutation importance plot saved → %s", save_path)

    # ------------------------------------------------------------------
    # sklearn feature importance
    # ------------------------------------------------------------------

    def get_feature_importance_sklearn(
        self,
        model: Any,
        feature_names: List[str],
    ) -> pd.DataFrame:
        """
        Extract built-in feature importance from sklearn / XGBoost models.

        Supports models with ``feature_importances_`` (trees, XGBoost) or
        ``coef_`` (linear models).

        Parameters
        ----------
        model : fitted sklearn-compatible model
        feature_names : list of str

        Returns
        -------
        pd.DataFrame with columns: feature, importance
            sorted by importance descending.
        """
        if hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_, dtype=float)
            kind = "feature_importances_"
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float)
            importances = np.abs(coef).ravel()
            kind = "coef_"
        else:
            raise AttributeError(
                f"Model {type(model).__name__} has neither 'feature_importances_' "
                "nor 'coef_'. Use compute_permutation_importance() instead."
            )

        if len(importances) != len(feature_names):
            raise ValueError(
                f"Length mismatch: {len(importances)} importances vs "
                f"{len(feature_names)} feature names."
            )

        df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        logger.info(
            "Feature importance extracted from '%s' | top: %s (%.4f)",
            kind,
            df.iloc[0]["feature"],
            df.iloc[0]["importance"],
        )
        return df

    # ------------------------------------------------------------------
    # Attention visualisation
    # ------------------------------------------------------------------

    def visualize_attention(
        self,
        attention_weights: np.ndarray,
        save_path: Union[str, Path],
        feature_names: Optional[List[str]] = None,
        sample_idx: int = 0,
    ) -> None:
        """
        Visualise Transformer attention weights as a heat-map.

        Parameters
        ----------
        attention_weights : np.ndarray
            Shape (N, heads, seq_len, seq_len)  or  (N, seq_len, seq_len)
            or (seq_len, seq_len) for a single sample/head.
        save_path : path where PNG is saved
        feature_names : optional list of token / feature labels
        sample_idx : int — which sample to plot if batch dimension present
        """
        if not _PLT_AVAILABLE:
            logger.error("matplotlib not installed; cannot visualise attention.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        attn = np.asarray(attention_weights, dtype=float)

        # Normalise dimensions to (seq_len, seq_len)
        if attn.ndim == 4:
            # (batch, heads, seq, seq) → mean over heads, select sample
            attn = attn[sample_idx].mean(axis=0)
        elif attn.ndim == 3:
            # (batch, seq, seq) or (heads, seq, seq)
            attn = attn[sample_idx]
        # Now (seq_len, seq_len)

        seq_len = attn.shape[0]
        tick_labels = feature_names if feature_names and len(feature_names) == seq_len else [
            str(i) for i in range(seq_len)
        ]

        fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.5), max(5, seq_len * 0.45)))
        im = ax.imshow(attn, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_title(f"Attention Weights (sample {sample_idx})", fontsize=13)
        ax.set_xlabel("Key", fontsize=11)
        ax.set_ylabel("Query", fontsize=11)
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("Attention visualisation saved → %s", save_path)

    # ------------------------------------------------------------------
    # Single-prediction report
    # ------------------------------------------------------------------

    def generate_prediction_report(
        self,
        model: Any,
        X_single: np.ndarray,
        feature_names: List[str],
        model_type: Literal["tree", "linear", "deep", "kernel"] = "tree",
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate a complete explanation for a single sample.

        Parameters
        ----------
        model : fitted model
        X_single : np.ndarray of shape (1, F) or (F,)
        feature_names : list of str, length F
        model_type : SHAP explainer type
        top_k : int — number of top features to include

        Returns
        -------
        dict with keys:
            prediction (int), confidence (float), top_features (list of dicts),
            explanation_text (str)
        """
        X_single = np.asarray(X_single, dtype=np.float32)
        if X_single.ndim == 1:
            X_single = X_single[None, :]

        # Get prediction
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                t = torch.tensor(X_single, dtype=torch.float32).to(self.device)
                logit = model(t)
                if logit.ndim == 2 and logit.shape[1] == 1:
                    logit = logit.squeeze(1)
                prob = float(torch.sigmoid(logit).cpu().item())
        elif hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(X_single)[0, 1])
        else:
            raise TypeError(
                f"Cannot get predictions from {type(model).__name__}."
            )

        prediction = int(prob >= 0.5)
        confidence = prob if prediction == 1 else 1.0 - prob

        # SHAP values for the single sample
        try:
            shap_vals = self.explain_with_shap(
                model, X_single, X_single, model_type=model_type
            )
            abs_shap = np.abs(shap_vals[0])
            top_indices = np.argsort(abs_shap)[::-1][:top_k]
            top_features = [
                {
                    "feature": feature_names[i],
                    "shap_value": round(float(shap_vals[0][i]), 6),
                    "feature_value": round(float(X_single[0, i]), 6),
                    "direction": "increases" if shap_vals[0][i] > 0 else "decreases",
                }
                for i in top_indices
            ]
        except Exception as exc:
            logger.warning("SHAP failed for prediction report: %s", exc)
            top_features = []

        # Narrative explanation
        label_str = "THEFT" if prediction == 1 else "LEGITIMATE"
        lines = [
            f"Prediction: {label_str} (confidence={confidence:.2%})",
            "",
            "Key contributing features:",
        ]
        for rank, feat in enumerate(top_features, 1):
            lines.append(
                f"  {rank}. {feat['feature']} = {feat['feature_value']:.4f} "
                f"→ {feat['direction']} theft probability "
                f"(SHAP={feat['shap_value']:+.4f})"
            )
        explanation_text = "\n".join(lines)

        report: Dict[str, Any] = {
            "prediction": prediction,
            "probability": round(prob, 6),
            "confidence": round(confidence, 6),
            "top_features": top_features,
            "explanation_text": explanation_text,
        }
        logger.info(
            "Prediction report: %s @ %.2f%% confidence | top feature: %s",
            label_str,
            confidence * 100,
            top_features[0]["feature"] if top_features else "N/A",
        )
        return report

    # ------------------------------------------------------------------
    # Batch explanation with summary statistics
    # ------------------------------------------------------------------

    def explain_batch(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        model_type: Literal["tree", "linear", "deep", "kernel"] = "tree",
        n_samples: int = 50,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for a batch of samples and return summary statistics.

        Parameters
        ----------
        model : fitted model
        X : np.ndarray (N, F)
        y : np.ndarray (N,) — true labels
        feature_names : list of str
        model_type : SHAP explainer type
        n_samples : int — subsample size if N > n_samples
        random_state : int

        Returns
        -------
        dict with keys:
            shap_values (np.ndarray N×F),
            mean_abs_shap (pd.DataFrame sorted by mean |SHAP|),
            correct_shap (mean |SHAP| on correctly classified samples),
            incorrect_shap (mean |SHAP| on misclassified samples),
            n_explained (int),
            summary_text (str)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)

        # Subsample
        rng = np.random.default_rng(random_state)
        if len(X) > n_samples:
            idx = rng.choice(len(X), n_samples, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y

        n_explained = len(X_sub)
        logger.info(
            "Batch explanation | n_explained=%d | model_type=%s",
            n_explained,
            model_type,
        )

        shap_values = self.explain_with_shap(
            model, X_sub, X_sub, model_type=model_type
        )

        # Predicted classes
        if isinstance(model, nn.Module):
            model.eval()
            with torch.no_grad():
                t = torch.tensor(X_sub, dtype=torch.float32).to(self.device)
                logits = model(t)
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                y_pred_sub = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        elif hasattr(model, "predict"):
            y_pred_sub = model.predict(X_sub)
        else:
            y_pred_sub = y_sub  # fallback

        correct_mask = y_pred_sub == y_sub
        incorrect_mask = ~correct_mask

        # Mean |SHAP| per feature
        mean_abs = np.abs(shap_values).mean(axis=0)
        mean_abs_df = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        # Split by correct / incorrect
        correct_shap: Optional[pd.DataFrame] = None
        incorrect_shap: Optional[pd.DataFrame] = None

        if correct_mask.sum() > 0:
            c_mean = np.abs(shap_values[correct_mask]).mean(axis=0)
            correct_shap = (
                pd.DataFrame({"feature": feature_names, "mean_abs_shap": c_mean})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )

        if incorrect_mask.sum() > 0:
            i_mean = np.abs(shap_values[incorrect_mask]).mean(axis=0)
            incorrect_shap = (
                pd.DataFrame({"feature": feature_names, "mean_abs_shap": i_mean})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )

        # Summary text
        top5 = mean_abs_df.head(5)
        lines = [
            f"Batch SHAP Explanation Summary ({n_explained} samples explained)",
            f"Correctly classified: {correct_mask.sum()} / {n_explained}",
            "",
            "Top-5 most important features (mean |SHAP|):",
        ]
        for _, row in top5.iterrows():
            lines.append(f"  {row['feature']:30s}  {row['mean_abs_shap']:.5f}")

        summary_text = "\n".join(lines)
        logger.info(summary_text.split("\n")[0])

        return {
            "shap_values": shap_values,
            "mean_abs_shap": mean_abs_df,
            "correct_shap": correct_shap,
            "incorrect_shap": incorrect_shap,
            "n_explained": n_explained,
            "summary_text": summary_text,
        }

    def __repr__(self) -> str:
        return (
            f"Explainer(device={self.device}, "
            f"max_display={self.shap_max_display}, "
            f"background_samples={self.background_samples})"
        )
