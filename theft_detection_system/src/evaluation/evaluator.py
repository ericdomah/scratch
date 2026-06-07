"""
evaluator.py
============
Production-quality model evaluation for the theft detection system.

Provides the ``Evaluator`` class which computes a full suite of binary
classification metrics, generates publication-ready plots, finds an
optimal decision threshold, and writes structured result reports.
"""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    _PLT_AVAILABLE = True
except ImportError:
    _PLT_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


class Evaluator:
    """
    Comprehensive evaluator for binary classification models.

    Parameters
    ----------
    config : dict, optional
        Optional configuration (not currently required).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}
        logger.info("Evaluator initialised.")

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Compute a comprehensive set of binary classification metrics.

        Parameters
        ----------
        y_true : array-like of shape (N,) — ground-truth binary labels
        y_prob : array-like of shape (N,) — predicted probabilities for class 1
        threshold : float — decision threshold (default 0.5)

        Returns
        -------
        dict with keys:
            accuracy, precision, recall, f1, roc_auc, pr_auc,
            mcc, cohen_kappa, balanced_accuracy,
            confusion_matrix (2×2 list), TP, TN, FP, FN
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        if len(y_true) != len(y_prob):
            raise ValueError(
                f"y_true length ({len(y_true)}) != y_prob length ({len(y_prob)})."
            )

        y_pred = (y_prob >= threshold).astype(int)

        # Confusion matrix entries
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel().tolist()
        else:
            # Edge case: only one class present
            tn, fp, fn, tp = 0, 0, 0, 0
            logger.warning("Confusion matrix is not 2×2; TN/FP/FN/TP may be 0.")

        # ROC AUC
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except ValueError as exc:
            logger.warning("ROC AUC could not be computed: %s", exc)
            roc_auc = float("nan")

        # PR AUC
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = float(auc(recall_vals, precision_vals))
        except ValueError as exc:
            logger.warning("PR AUC could not be computed: %s", exc)
            pr_auc = float("nan")

        metrics: Dict[str, Any] = {
            "threshold": round(float(threshold), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(
                float(precision_score(y_true, y_pred, zero_division=0)), 4
            ),
            "recall": round(
                float(recall_score(y_true, y_pred, zero_division=0)), 4
            ),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
            "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(y_true, y_pred)), 4
            ),
            "confusion_matrix": cm.tolist(),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }
        logger.info(
            "Metrics @ thr=%.3f | acc=%.4f | f1=%.4f | roc_auc=%.4f | pr_auc=%.4f",
            threshold,
            metrics["accuracy"],
            metrics["f1"],
            metrics["roc_auc"],
            metrics["pr_auc"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Optimal threshold
    # ------------------------------------------------------------------

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = "f1",
    ) -> float:
        """
        Find the decision threshold that maximises the chosen metric.

        Parameters
        ----------
        y_true : array-like (N,)
        y_prob : array-like (N,)
        metric : ``"f1"``, ``"balanced_accuracy"``, ``"precision"``,
                 ``"recall"``, or ``"mcc"``

        Returns
        -------
        float — optimal threshold
        """
        y_true = np.asarray(y_true, dtype=int)
        y_prob = np.asarray(y_prob, dtype=float)

        _supported = ("f1", "balanced_accuracy", "precision", "recall", "mcc")
        if metric not in _supported:
            raise ValueError(f"metric must be one of {_supported}, got '{metric}'.")

        best_thr, best_score = 0.5, -np.inf
        thresholds = np.arange(0.05, 0.96, 0.05)

        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "balanced_accuracy":
                score = balanced_accuracy_score(y_true, y_pred)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "mcc":
                score = matthews_corrcoef(y_true, y_pred)
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_thr = float(thr)

        logger.info(
            "Optimal threshold for '%s': %.4f (score=%.4f)", metric, best_thr, best_score
        )
        return round(best_thr, 4)

    # ------------------------------------------------------------------
    # ROC curve
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        save_path: Union[str, Path],
    ) -> None:
        """
        Plot and save an ROC curve.

        Parameters
        ----------
        y_true : array-like (N,)
        y_prob : array-like (N,)
        model_name : str — used in the title and legend
        save_path : path where the PNG is saved
        """
        if not _PLT_AVAILABLE:
            logger.error("matplotlib not installed; cannot plot ROC curve.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fpr, tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_prob))
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, lw=2, color="royalblue", label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve — {model_name}", fontsize=13)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("ROC curve saved → %s", save_path)

    # ------------------------------------------------------------------
    # PR curve
    # ------------------------------------------------------------------

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        save_path: Union[str, Path],
    ) -> None:
        """
        Plot and save a Precision-Recall curve.

        Parameters
        ----------
        y_true : array-like (N,)
        y_prob : array-like (N,)
        model_name : str
        save_path : path where the PNG is saved
        """
        if not _PLT_AVAILABLE:
            logger.error("matplotlib not installed; cannot plot PR curve.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        precision_vals, recall_vals, _ = precision_recall_curve(
            np.asarray(y_true), np.asarray(y_prob)
        )
        pr_auc = auc(recall_vals, precision_vals)
        baseline = float(np.asarray(y_true).mean())

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(
            recall_vals,
            precision_vals,
            lw=2,
            color="darkorange",
            label=f"PR AUC = {pr_auc:.4f}",
        )
        ax.axhline(y=baseline, linestyle="--", color="grey", lw=1, label=f"Baseline = {baseline:.4f}")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=13)
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("PR curve saved → %s", save_path)

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        save_path: Union[str, Path],
    ) -> None:
        """
        Plot and save an annotated confusion matrix.

        Parameters
        ----------
        y_true : array-like (N,) — ground-truth labels
        y_pred : array-like (N,) — predicted labels (not probabilities)
        model_name : str
        save_path : path where the PNG is saved
        """
        if not _PLT_AVAILABLE:
            logger.error("matplotlib not installed; cannot plot confusion matrix.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        classes = ["Legitimate (0)", "Theft (1)"]
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, rotation=25, ha="right", fontsize=10)
        ax.set_yticklabels(classes, fontsize=10)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=13,
                )

        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", save_path)

    # ------------------------------------------------------------------
    # Loss curves
    # ------------------------------------------------------------------

    def plot_loss_curves(
        self,
        history: Dict[str, List[float]],
        model_name: str,
        save_path: Union[str, Path],
    ) -> None:
        """
        Plot training and validation loss curves from a history dict.

        Parameters
        ----------
        history : dict with keys ``"train_loss"`` and optionally ``"val_loss"``
        model_name : str
        save_path : path where the PNG is saved
        """
        if not _PLT_AVAILABLE:
            logger.error("matplotlib not installed; cannot plot loss curves.")
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 5))
        if "train_loss" in history:
            ax.plot(history["train_loss"], label="Train Loss", color="steelblue", lw=2)
        if "val_loss" in history:
            ax.plot(
                history["val_loss"],
                label="Val Loss",
                color="orange",
                linestyle="--",
                lw=2,
            )
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(f"Loss Curves — {model_name}", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        fig.savefig(str(save_path), dpi=150)
        plt.close(fig)
        logger.info("Loss curves saved → %s", save_path)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def generate_comparison_table(
        self,
        results_dict: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Build a comparison DataFrame from per-model metric dicts.

        Parameters
        ----------
        results_dict : {model_name: metrics_dict, ...}
            Each ``metrics_dict`` is the output of ``compute_metrics()``.

        Returns
        -------
        pd.DataFrame sorted by F1 desc, then ROC-AUC desc.
        """
        if not results_dict:
            logger.warning("generate_comparison_table: empty results_dict.")
            return pd.DataFrame()

        _COLUMNS = [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "mcc",
            "cohen_kappa",
            "balanced_accuracy",
            "TP",
            "TN",
            "FP",
            "FN",
        ]
        rows = []
        for model_name, metrics in results_dict.items():
            row: Dict[str, Any] = {"model": model_name}
            for col in _COLUMNS[1:]:
                row[col] = metrics.get(col, float("nan"))
            rows.append(row)

        df = (
            pd.DataFrame(rows, columns=_COLUMNS)
            .sort_values(["f1", "roc_auc", "pr_auc"], ascending=False)
            .reset_index(drop=True)
        )
        logger.info("Comparison table generated | %d models.", len(df))
        return df

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    def save_results(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        path: Union[str, Path],
    ) -> None:
        """
        Persist results as both JSON and CSV.

        Parameters
        ----------
        results_dict : {model_name: metrics_dict, ...}
        path : directory path (created if necessary)
        """
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON — full fidelity
        json_path = out_dir / "evaluation_results.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results_dict, fh, indent=2, default=str)
        logger.info("Results JSON saved → %s", json_path)

        # CSV — comparison table
        csv_path = out_dir / "evaluation_results.csv"
        df = self.generate_comparison_table(results_dict)
        df.to_csv(str(csv_path), index=False)
        logger.info("Results CSV  saved → %s", csv_path)

    # ------------------------------------------------------------------
    # Full text report
    # ------------------------------------------------------------------

    def generate_full_report(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        save_path: Union[str, Path],
    ) -> str:
        """
        Generate a structured text summary report.

        Parameters
        ----------
        results_dict : {model_name: metrics_dict, ...}
        save_path : path for the ``.txt`` report

        Returns
        -------
        str — the full report text
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        sep = "=" * 70
        lines: List[str] = [
            sep,
            "THEFT DETECTION SYSTEM — MODEL EVALUATION REPORT",
            sep,
            f"Number of models evaluated: {len(results_dict)}",
            "",
        ]

        # Per-model section
        for model_name, m in results_dict.items():
            lines += [
                f"  MODEL: {model_name}",
                "  " + "-" * 50,
                f"    Threshold          : {m.get('threshold', 0.5):.4f}",
                f"    Accuracy           : {m.get('accuracy', 'N/A')}",
                f"    Precision          : {m.get('precision', 'N/A')}",
                f"    Recall             : {m.get('recall', 'N/A')}",
                f"    F1 Score           : {m.get('f1', 'N/A')}",
                f"    ROC AUC            : {m.get('roc_auc', 'N/A')}",
                f"    PR AUC             : {m.get('pr_auc', 'N/A')}",
                f"    MCC                : {m.get('mcc', 'N/A')}",
                f"    Cohen's Kappa      : {m.get('cohen_kappa', 'N/A')}",
                f"    Balanced Accuracy  : {m.get('balanced_accuracy', 'N/A')}",
                f"    TP / TN / FP / FN  : "
                f"{m.get('TP','?')} / {m.get('TN','?')} / {m.get('FP','?')} / {m.get('FN','?')}",
                "",
            ]

        # Ranking table
        df = self.generate_comparison_table(results_dict)
        if not df.empty:
            lines += [
                sep,
                "RANKING (sorted by F1 ↓, ROC-AUC ↓)",
                sep,
                df[["model", "f1", "roc_auc", "accuracy", "mcc"]]
                .to_string(index=False),
                "",
            ]

            # Best model highlight
            best = df.iloc[0]
            lines += [
                sep,
                f"BEST MODEL: {best['model']}",
                f"  F1={best['f1']:.4f}  ROC-AUC={best['roc_auc']:.4f}  "
                f"Accuracy={best['accuracy']:.4f}  MCC={best['mcc']:.4f}",
                sep,
            ]

        report_text = "\n".join(lines)

        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(report_text)
        logger.info("Full report saved → %s", save_path)
        return report_text

    def __repr__(self) -> str:
        return "Evaluator()"
