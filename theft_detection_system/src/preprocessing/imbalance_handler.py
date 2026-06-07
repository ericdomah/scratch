"""
imbalance_handler.py
====================
Production-quality ImbalanceHandler class for the SGCC Electricity Theft
Detection system.

The SGCC dataset is strongly imbalanced (~20 % theft, ~80 % normal).  This
module centralises all strategies for dealing with that imbalance:

  * **weighted_loss / focal_loss / none** – no resampling; class weights or
    focal-loss parameters are computed for use in the loss function.
  * **smote** – Synthetic Minority Over-sampling Technique (imblearn).
  * **adasyn** – Adaptive Synthetic Sampling (imblearn).
  * **oversample** – Random over-sampling (imblearn).
  * **undersample** – Random under-sampling (imblearn).

Typical usage
-------------
>>> from theft_detection_system.src.preprocessing.imbalance_handler import ImbalanceHandler
>>> handler = ImbalanceHandler(method="smote", config={"random_state": 42, "k_neighbors": 5})
>>> X_res, y_res = handler.resample(X_train, y_train)
>>> class_weights = handler.compute_class_weights(y_train)
>>> pos_weight    = handler.compute_pos_weight(y_train)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

_handler_log = logging.StreamHandler()
_handler_log.setLevel(logging.INFO)
_fmt = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_handler_log.setFormatter(_fmt)
if not logger.handlers:
    logger.addHandler(_handler_log)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_METHODS = frozenset(
    {"weighted_loss", "smote", "borderlinesmote", "smoteenn", "smotetomek", "adasyn", "oversample", "undersample", "focal_loss", "none"}
)


# ---------------------------------------------------------------------------
# ImbalanceHandler
# ---------------------------------------------------------------------------
class ImbalanceHandler:
    """
    Centralised handler for class imbalance in binary classification.

    Parameters
    ----------
    method : str
        Imbalance strategy.  Must be one of:
        ``"weighted_loss"``, ``"smote"``, ``"borderlinesmote"``, ``"smoteenn"``,
        ``"smotetomek"``, ``"adasyn"``, ``"oversample"``,
        ``"undersample"``, ``"focal_loss"``, ``"none"``.
    config : dict, optional
        Strategy-specific hyper-parameters.  Recognised keys:

        ======================== ============= ================================
        Key                      Default       Description
        ======================== ============= ================================
        ``random_state``         ``42``        RNG seed for all samplers.
        ``k_neighbors``          ``5``         *k* for SMOTE / ADASYN.
        ``sampling_strategy``    ``"auto"``    Passed to all imblearn samplers.
        ``focal_alpha``          ``0.25``      α for focal loss.
        ``focal_gamma``          ``2.0``       γ for focal loss.
        ``n_jobs``               ``-1``        Parallelism for imblearn.
        ======================== ============= ================================

    Raises
    ------
    ValueError
        If *method* is not one of the accepted values.

    Examples
    --------
    >>> ih = ImbalanceHandler(method="smote", config={"k_neighbors": 7})
    >>> X_res, y_res = ih.resample(X_train, y_train)
    """

    def __init__(
        self,
        method: str = "weighted_loss",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        method = method.lower().strip()
        if method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown imbalance method '{method}'.  "
                f"Valid options: {sorted(_VALID_METHODS)}"
            )

        self.method: str = method
        self.config: Dict[str, Any] = config or {}

        # Merge defaults into config (without overwriting user-provided values)
        _defaults: Dict[str, Any] = {
            "random_state": 42,
            "k_neighbors": 5,
            "sampling_strategy": "auto",
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
            "n_jobs": -1,
        }
        for key, default_val in _defaults.items():
            self.config.setdefault(key, default_val)

        logger.info(
            "ImbalanceHandler initialised: method='%s', config=%s",
            self.method, self.config,
        )

    # ------------------------------------------------------------------
    # resample
    # ------------------------------------------------------------------

    def resample(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the configured resampling strategy and return (X_res, y_res).

        For strategies that do not resample the data
        (``"weighted_loss"``, ``"focal_loss"``, ``"none"``) the original
        arrays are returned unchanged.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
            Training feature matrix.
        y_train : np.ndarray, shape (n_samples,)
            Binary label vector (0 = normal, 1 = theft).

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Resampled (or unchanged) ``(X, y)`` pair.

        Raises
        ------
        RuntimeError
            If an imblearn dependency is missing.
        """
        if not isinstance(X_train, np.ndarray):
            X_train = np.asarray(X_train, dtype=np.float32)
        if not isinstance(y_train, np.ndarray):
            y_train = np.asarray(y_train, dtype=np.int64)

        logger.info(
            "resample() called with method='%s'  input: X=%s, y=%s",
            self.method, X_train.shape, y_train.shape,
        )
        self.print_class_distribution(y_train, label="Before resampling")

        if self.method in {"weighted_loss", "focal_loss", "none"}:
            logger.info(
                "Method '%s' does not resample data; returning original arrays.",
                self.method,
            )
            return X_train, y_train

        if self.method == "smote":
            X_res, y_res = self._apply_smote(X_train, y_train)
        elif self.method == "borderlinesmote":
            X_res, y_res = self._apply_borderline_smote(X_train, y_train)
        elif self.method == "smoteenn":
            X_res, y_res = self._apply_smoteenn(X_train, y_train)
        elif self.method == "smotetomek":
            X_res, y_res = self._apply_smotetomek(X_train, y_train)
        elif self.method == "adasyn":
            X_res, y_res = self._apply_adasyn(X_train, y_train)
        elif self.method == "oversample":
            X_res, y_res = self._apply_random_oversample(X_train, y_train)
        elif self.method == "undersample":
            X_res, y_res = self._apply_random_undersample(X_train, y_train)
        else:
            # Should be unreachable due to __init__ validation
            raise ValueError(f"Unhandled method: '{self.method}'")

        self.print_class_distribution(y_res, label="After resampling")
        return X_res, y_res

    # ------------------------------------------------------------------
    # Private resampling implementations
    # ------------------------------------------------------------------

    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE over-sampling."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise RuntimeError(
                "imblearn is required for SMOTE.  "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        logger.info(
            "Applying SMOTE (k_neighbors=%d, sampling_strategy=%s) …",
            self.config["k_neighbors"], self.config["sampling_strategy"],
        )
        sampler = SMOTE(
            k_neighbors=int(self.config["k_neighbors"]),
            sampling_strategy=self.config["sampling_strategy"],
            random_state=int(self.config["random_state"]),
        )
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info(
            "SMOTE complete: %d → %d samples.", len(y), len(y_res)
        )
        return X_res.astype(np.float32), y_res.astype(np.int64)

    def _apply_borderline_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply BorderlineSMOTE over-sampling."""
        try:
            from imblearn.over_sampling import BorderlineSMOTE
        except ImportError as exc:
            raise RuntimeError(
                "imblearn is required for BorderlineSMOTE.  "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        logger.info(
            "Applying BorderlineSMOTE (k_neighbors=%d, sampling_strategy=%s) …",
            self.config["k_neighbors"], self.config["sampling_strategy"],
        )
        sampler = BorderlineSMOTE(
            k_neighbors=int(self.config["k_neighbors"]),
            sampling_strategy=self.config["sampling_strategy"],
            random_state=int(self.config["random_state"]),
        )
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info(
            "BorderlineSMOTE complete: %d → %d samples.", len(y), len(y_res)
        )
        return X_res.astype(np.float32), y_res.astype(np.int64)

    def _apply_smoteenn(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTEENN over- and under-sampling."""
        try:
            from imblearn.combine import SMOTEENN
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise RuntimeError(
                "imblearn is required for SMOTEENN.  "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        logger.info(
            "Applying SMOTEENN (k_neighbors=%d, sampling_strategy=%s) …",
            self.config["k_neighbors"], self.config["sampling_strategy"],
        )
        smote = SMOTE(
            k_neighbors=int(self.config["k_neighbors"]),
            sampling_strategy=self.config["sampling_strategy"],
            random_state=int(self.config["random_state"]),
        )
        sampler = SMOTEENN(
            smote=smote,
            random_state=int(self.config["random_state"]),
        )
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info(
            "SMOTEENN complete: %d → %d samples.", len(y), len(y_res)
        )
        return X_res.astype(np.float32), y_res.astype(np.int64)

    def _apply_smotetomek(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTETomek over- and under-sampling."""
        try:
            from imblearn.combine import SMOTETomek
            from imblearn.over_sampling import SMOTE
        except ImportError as exc:
            raise RuntimeError(
                "imblearn is required for SMOTETomek.  "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        logger.info(
            "Applying SMOTETomek (k_neighbors=%d, sampling_strategy=%s) …",
            self.config["k_neighbors"], self.config["sampling_strategy"],
        )
        smote = SMOTE(
            k_neighbors=int(self.config["k_neighbors"]),
            sampling_strategy=self.config["sampling_strategy"],
            random_state=int(self.config["random_state"]),
        )
        sampler = SMOTETomek(
            smote=smote,
            random_state=int(self.config["random_state"]),
        )
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info(
            "SMOTETomek complete: %d → %d samples.", len(y), len(y_res)
        )
        return X_res.astype(np.float32), y_res.astype(np.int64)

    def _apply_adasyn(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply ADASYN adaptive over-sampling."""
        try:
            from imblearn.over_sampling import ADASYN
        except ImportError as exc:
            raise RuntimeError(
                "imblearn is required for ADASYN.  "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        logger.info(
            "Applying ADASYN (n_neighbors=%d, sampling_strategy=%s) …",
            self.config["k_neighbors"], self.config["sampling_strategy"],
        )
        sampler = ADASYN(
            n_neighbors=int(self.config["k_neighbors"]),
            sampling_strategy=self.config["sampling_strategy"],
            random_state=int(self.config["random_state"]),
            n_jobs=int(self.config["n_jobs"]),
        )
        try:
            X_res, y_res = sampler.fit_resample(X, y)
        except ValueError as exc:
            logger.warning(
                "ADASYN raised ValueError ('%s'); falling back to SMOTE.", exc
            )
            return self._apply_smote(X, y)

        logger.info(
            "ADASYN complete: %d → %d samples.", len(y), len(y_res)
        )
        return X_res.astype(np.float32), y_res.astype(np.int64)

    def _apply_random_oversample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random over-sampling (duplication of minority samples)."""
        try:
            from imblearn.over_sampling import RandomOverSampler
        except ImportError as exc:
            raise RuntimeError(
                "imblearn is required for RandomOverSampler.  "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        logger.info(
            "Applying RandomOverSampler (sampling_strategy=%s) …",
            self.config["sampling_strategy"],
        )
        sampler = RandomOverSampler(
            sampling_strategy=self.config["sampling_strategy"],
            random_state=int(self.config["random_state"]),
        )
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info(
            "RandomOverSampler complete: %d → %d samples.", len(y), len(y_res)
        )
        return X_res.astype(np.float32), y_res.astype(np.int64)

    def _apply_random_undersample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random under-sampling (remove majority samples)."""
        try:
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:
            raise RuntimeError(
                "imblearn is required for RandomUnderSampler.  "
                "Install it with: pip install imbalanced-learn"
            ) from exc

        logger.info(
            "Applying RandomUnderSampler (sampling_strategy=%s) …",
            self.config["sampling_strategy"],
        )
        sampler = RandomUnderSampler(
            sampling_strategy=self.config["sampling_strategy"],
            random_state=int(self.config["random_state"]),
        )
        X_res, y_res = sampler.fit_resample(X, y)
        logger.info(
            "RandomUnderSampler complete: %d → %d samples.", len(y), len(y_res)
        )
        return X_res.astype(np.float32), y_res.astype(np.int64)

    # ------------------------------------------------------------------
    # compute_class_weights
    # ------------------------------------------------------------------

    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute balanced class weights for use in sklearn estimators or
        custom loss functions.

        The weight for class *c* is::

            n_samples / (n_classes * count_c)

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            Binary label vector.

        Returns
        -------
        dict
            Mapping ``{class_index: weight}``, e.g. ``{0: 0.625, 1: 2.5}``.
        """
        y = np.asarray(y, dtype=np.int64)
        classes = np.unique(y)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y,
        )
        weight_dict: Dict[int, float] = {
            int(cls): float(w) for cls, w in zip(classes, weights)
        }
        logger.info("Class weights (balanced): %s", weight_dict)
        return weight_dict

    # ------------------------------------------------------------------
    # compute_pos_weight
    # ------------------------------------------------------------------

    def compute_pos_weight(self, y: np.ndarray) -> float:
        """
        Compute the ``pos_weight`` scalar for PyTorch's
        :class:`~torch.nn.BCEWithLogitsLoss`.

        The scalar is defined as::

            pos_weight = n_negative / n_positive

        A value > 1 up-weights the positive (theft) class in the loss.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            Binary label vector.

        Returns
        -------
        float
            Positive-class weight scalar.

        Raises
        ------
        ValueError
            If there are no positive or no negative samples.
        """
        y = np.asarray(y, dtype=np.int64)
        n_positive = int(np.sum(y == 1))
        n_negative = int(np.sum(y == 0))

        if n_positive == 0:
            raise ValueError("No positive (theft) samples found in y.")
        if n_negative == 0:
            raise ValueError("No negative (normal) samples found in y.")

        pos_weight = float(n_negative / n_positive)
        logger.info(
            "pos_weight = %.4f  (n_neg=%d / n_pos=%d)",
            pos_weight, n_negative, n_positive,
        )
        return pos_weight

    # ------------------------------------------------------------------
    # get_focal_loss_params
    # ------------------------------------------------------------------

    def get_focal_loss_params(self) -> Dict[str, float]:
        """
        Return the focal loss hyper-parameters ``alpha`` and ``gamma``
        as configured in :attr:`config`.

        Returns
        -------
        dict
            ``{"alpha": float, "gamma": float}``

        Notes
        -----
        These parameters are meaningful only when :attr:`method` is
        ``"focal_loss"``, but can be retrieved regardless.

        References
        ----------
        Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
        *Focal Loss for Dense Object Detection*. ICCV.
        """
        alpha = float(self.config.get("focal_alpha", 0.25))
        gamma = float(self.config.get("focal_gamma", 2.0))
        params = {"alpha": alpha, "gamma": gamma}
        logger.info("Focal loss params: alpha=%.4f, gamma=%.4f", alpha, gamma)
        return params

    # ------------------------------------------------------------------
    # print_class_distribution
    # ------------------------------------------------------------------

    def print_class_distribution(
        self,
        y: np.ndarray,
        label: str = "Before",
    ) -> None:
        """
        Print class counts and class ratios for the given label vector.

        Parameters
        ----------
        y : np.ndarray, shape (n_samples,)
            Binary label vector.
        label : str
            Descriptive prefix printed in the header.

        Returns
        -------
        None
        """
        y = np.asarray(y, dtype=np.int64)
        total = len(y)
        classes, counts = np.unique(y, return_counts=True)

        print(f"\n{'=' * 50}")
        print(f"  Class Distribution  [{label}]")
        print(f"{'=' * 50}")
        print(f"  Total samples : {total}")
        for cls, cnt in zip(classes, counts):
            ratio = cnt / total if total > 0 else 0.0
            class_name = "Normal (0)" if int(cls) == 0 else "Theft  (1)"
            print(f"  {class_name} : {cnt:>7d}  ({ratio:.2%})")
        print(f"{'=' * 50}\n")

        logger.info(
            "[%s] Total=%d  %s",
            label, total,
            "  ".join(
                f"class {int(c)}={cnt}({cnt/total:.2%})"
                for c, cnt in zip(classes, counts)
            ),
        )

    # ------------------------------------------------------------------
    # __repr__
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ImbalanceHandler("
            f"method='{self.method}', "
            f"config={self.config})"
        )
