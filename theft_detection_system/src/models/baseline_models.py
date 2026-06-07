"""
baseline_models.py
==================
Production-quality BaselineModelSuite that trains, evaluates, and tunes
8 classical/ensemble machine-learning models for theft detection.

Models
------
1. LogisticRegression
2. DecisionTreeClassifier
3. RandomForestClassifier
4. ExtraTreesClassifier
5. XGBClassifier
6. LGBMClassifier
7. CatBoostClassifier
8. SVC

Author : ML Engineering Team
Created: 2026-06-07
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Suppress verbose third-party loggers unless debugging
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]
ModelDict = Dict[str, Any]
MetricsDict = Dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute area under the precision-recall curve via trapezoidal rule."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    # recall is decreasing; reverse for np.trapz
    return float(np.trapz(precision[::-1], recall[::-1]))


def _safe_predict_proba(
    model: Any,
    X: ArrayLike,
) -> Optional[np.ndarray]:
    """
    Return positive-class probability scores or None if the model cannot
    produce probability estimates.
    """
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:  # noqa: BLE001
            return None
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
            # Normalise to [0, 1] via sigmoid
            return 1.0 / (1.0 + np.exp(-scores))
        except Exception:  # noqa: BLE001
            return None
    return None


# ---------------------------------------------------------------------------
# Default configuration used when no YAML is found / key is absent
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG: Dict[str, Any] = {
    "models": {
        "logistic_regression": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "decision_tree": {
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "extra_trees": {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "xgboost": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        },
        "lightgbm": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        "catboost": {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 6,
            "eval_metric": "F1",
            "random_seed": 42,
            "verbose": 0,
            "auto_class_weights": "Balanced",
        },
        "svm": {
            "C": 1.0,
            "kernel": "rbf",
            "gamma": "scale",
            "probability": True,
            "class_weight": "balanced",
            "random_state": 42,
        },
    },
    "cross_validation": {"n_splits": 5, "shuffle": True, "random_state": 42},
    "optuna": {"n_trials": 50, "direction": "maximize", "metric": "f1"},
}


# ---------------------------------------------------------------------------
# BaselineModelSuite
# ---------------------------------------------------------------------------

class BaselineModelSuite:
    """
    Train, evaluate, and tune 8 baseline ML models for theft detection.

    Parameters
    ----------
    config_path : str | Path | None
        Path to a ``model_config.yaml`` file.  When *None* or the file is
        missing, built-in defaults are used.
    output_dir : str | Path
        Root directory for persisting models and artefacts.
        Sub-directory ``models/`` is created automatically.
    """

    MODEL_NAMES: List[str] = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
        "catboost",
        "svm",
    ]

    # ------------------------------------------------------------------
    # Construction / initialisation
    # ------------------------------------------------------------------

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "output",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.models_dir = self.output_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.config: Dict[str, Any] = self._load_config(config_path)

        # Populated by train_all / load_model
        self.trained_models: ModelDict = {}
        # Populated by evaluate_all / evaluate_single
        self.metrics: Dict[str, MetricsDict] = {}
        # Training wall-clock times in seconds
        self.train_times: Dict[str, float] = {}

        self.models: ModelDict = self._build_models()

        logger.info(
            "BaselineModelSuite initialised with %d models. "
            "Output directory: %s",
            len(self.models),
            self.output_dir,
        )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(
        self, config_path: Optional[Union[str, Path]]
    ) -> Dict[str, Any]:
        """Load YAML configuration, falling back to built-in defaults."""
        cfg = _DEFAULT_CONFIG.copy()

        if config_path is None:
            logger.info("No config_path supplied – using built-in defaults.")
            return cfg

        config_path = Path(config_path)
        if not config_path.is_file():
            logger.warning(
                "Config file not found at '%s'. Using built-in defaults.",
                config_path,
            )
            return cfg

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                user_cfg = yaml.safe_load(fh)
            if user_cfg and isinstance(user_cfg, dict):
                # Deep-merge: user values override defaults
                for section, params in user_cfg.items():
                    if section in cfg and isinstance(cfg[section], dict):
                        cfg[section].update(params)
                    else:
                        cfg[section] = params
            logger.info("Configuration loaded from '%s'.", config_path)
        except yaml.YAMLError as exc:
            logger.error(
                "Failed to parse YAML config '%s': %s. Using defaults.",
                config_path,
                exc,
            )
        return cfg

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_models(self) -> ModelDict:
        """
        Instantiate all 8 models using parameters from the loaded config.

        Returns
        -------
        dict
            Mapping of ``model_name -> sklearn-compatible estimator``.
        """
        mc = self.config.get("models", {})

        def _p(key: str) -> Dict[str, Any]:
            """Return params for the given model key, empty dict if absent."""
            return mc.get(key, {})

        models: ModelDict = {}

        # 1. Logistic Regression
        lr_params = _p("logistic_regression")
        models["logistic_regression"] = LogisticRegression(**lr_params)
        logger.debug("Built LogisticRegression with params: %s", lr_params)

        # 2. Decision Tree
        dt_params = _p("decision_tree")
        models["decision_tree"] = DecisionTreeClassifier(**dt_params)
        logger.debug("Built DecisionTreeClassifier with params: %s", dt_params)

        # 3. Random Forest
        rf_params = _p("random_forest")
        models["random_forest"] = RandomForestClassifier(**rf_params)
        logger.debug("Built RandomForestClassifier with params: %s", rf_params)

        # 4. Extra Trees
        et_params = _p("extra_trees")
        models["extra_trees"] = ExtraTreesClassifier(**et_params)
        logger.debug("Built ExtraTreesClassifier with params: %s", et_params)

        # 5. XGBoost
        xgb_params = {k: v for k, v in _p("xgboost").items()}
        # Remove deprecated key if present in older configs
        xgb_params.pop("use_label_encoder", None)
        models["xgboost"] = XGBClassifier(**xgb_params)
        logger.debug("Built XGBClassifier with params: %s", xgb_params)

        # 6. LightGBM
        lgbm_params = _p("lightgbm")
        models["lightgbm"] = LGBMClassifier(**lgbm_params)
        logger.debug("Built LGBMClassifier with params: %s", lgbm_params)

        # 7. CatBoost
        cb_params = _p("catboost")
        models["catboost"] = CatBoostClassifier(**cb_params)
        logger.debug("Built CatBoostClassifier with params: %s", cb_params)

        # 8. SVM
        svm_params = _p("svm")
        
        # Safely extract params supported by LinearSVC
        import inspect
        sig = inspect.signature(LinearSVC.__init__)
        safe_svm_params = {
            k: v for k, v in svm_params.items() 
            if k in sig.parameters
        }
        
        base_svm = LinearSVC(**safe_svm_params)
        models["svm"] = CalibratedClassifierCV(base_svm, method="sigmoid", cv=5)
        logger.debug("Built LinearSVC wrapped in CalibratedClassifierCV with params: %s", safe_svm_params)

        logger.info("All %d models built successfully.", len(models))
        return models

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_single(
        self,
        name: str,
        model: Any,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> Any:
        """
        Fit a single model, recording training wall-clock time.

        Parameters
        ----------
        name : str
            Human-readable model identifier used for logging / caching.
        model : estimator
            An unfitted sklearn-compatible estimator.
        X_train, y_train : array-like
            Training features and labels.
        X_val, y_val : array-like
            Validation features and labels (used for early stopping where
            supported; otherwise used for a post-fit validation log).
        class_weights : dict | None
            Optional ``{class_label: weight}`` mapping.  Converted to
            sample weights for models that do not accept ``class_weight``.

        Returns
        -------
        estimator
            The fitted model.
        """
        logger.info("Training model: '%s' …", name)

        X_train_arr = np.array(X_train)
        y_train_arr = np.array(y_train)
        X_val_arr = np.array(X_val)
        y_val_arr = np.array(y_val)

        # Build sample weights if class_weights are provided
        sample_weights: Optional[np.ndarray] = None
        if class_weights:
            sample_weights = compute_sample_weight(
                class_weight=class_weights, y=y_train_arr
            )

        t0 = time.perf_counter()

        try:
            if name == "xgboost" and sample_weights is not None:
                model.fit(
                    X_train_arr,
                    y_train_arr,
                    sample_weight=sample_weights,
                    eval_set=[(X_val_arr, y_val_arr)],
                    verbose=False,
                )
            elif name == "lightgbm" and sample_weights is not None:
                model.fit(
                    X_train_arr,
                    y_train_arr,
                    sample_weight=sample_weights,
                    eval_set=[(X_val_arr, y_val_arr)],
                )
            elif name == "catboost":
                # CatBoost accepts Pool objects or raw arrays; raw is fine here
                cb_fit_kwargs: Dict[str, Any] = {
                    "eval_set": (X_val_arr, y_val_arr),
                    "use_best_model": False,
                }
                if sample_weights is not None:
                    cb_fit_kwargs["sample_weight"] = sample_weights
                model.fit(X_train_arr, y_train_arr, **cb_fit_kwargs)
            elif sample_weights is not None:
                # Most sklearn estimators accept sample_weight in fit()
                try:
                    model.fit(
                        X_train_arr, y_train_arr, sample_weight=sample_weights
                    )
                except TypeError:
                    # Estimator does not support sample_weight (e.g. some SVC)
                    logger.warning(
                        "'%s' does not support sample_weight – fitting "
                        "without it.",
                        name,
                    )
                    model.fit(X_train_arr, y_train_arr)
            else:
                model.fit(X_train_arr, y_train_arr)

        except Exception as exc:
            logger.error("Error training model '%s': %s", name, exc, exc_info=True)
            raise

        elapsed = time.perf_counter() - t0
        self.train_times[name] = elapsed

        # Quick validation accuracy for the log
        val_pred = model.predict(X_val_arr)
        val_acc = accuracy_score(y_val_arr, val_pred)
        val_f1 = f1_score(y_val_arr, val_pred, average="binary", zero_division=0)

        logger.info(
            "Finished training '%s' in %.2fs | val_acc=%.4f | val_f1=%.4f",
            name,
            elapsed,
            val_acc,
            val_f1,
        )
        return model

    def train_all(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> ModelDict:
        """
        Train every model in ``self.models``, persist each with joblib.

        Parameters
        ----------
        X_train, y_train : array-like
            Training data.
        X_val, y_val : array-like
            Validation data.
        class_weights : dict | None
            Optional class-weight mapping.

        Returns
        -------
        dict
            Mapping of model name to fitted estimator.
        """
        logger.info(
            "Starting training of all %d models …", len(self.models)
        )
        total_start = time.perf_counter()

        for name, model in self.models.items():
            try:
                fitted = self.train_single(
                    name,
                    model,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    class_weights,
                )
                self.trained_models[name] = fitted
                self.save_model(name, fitted)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Skipping '%s' due to training error: %s", name, exc
                )

        total_elapsed = time.perf_counter() - total_start
        logger.info(
            "All models trained in %.2fs. Successfully trained: %s",
            total_elapsed,
            list(self.trained_models.keys()),
        )
        return self.trained_models

    # ------------------------------------------------------------------
    # Hyperparameter Tuning (Optuna)
    # ------------------------------------------------------------------

    def tune_model(
        self,
        name: str,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> Any:
        """
        Use Optuna to tune hyperparameters for the specified model.
        Optimizes for F1 score on the validation set.
        """
        if name not in ["xgboost", "lightgbm", "catboost", "logistic_regression"]:
            logger.info("Tuning not implemented/requested for '%s'. Skipping.", name)
            return self.models[name]

        logger.info("Starting Optuna tuning for '%s' …", name)
        
        X_train_arr = np.array(X_train)
        y_train_arr = np.array(y_train)
        X_val_arr = np.array(X_val)
        y_val_arr = np.array(y_val)
        
        sample_weights = None
        if class_weights:
            sample_weights = compute_sample_weight(class_weight=class_weights, y=y_train_arr)

        def objective(trial: optuna.Trial) -> float:
            if name == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "objective": "binary:logistic",
                    "eval_metric": "aucpr",
                    "tree_method": "hist",
                    "random_state": 42,
                    "n_jobs": -1
                }
                model = XGBClassifier(**params)
                if sample_weights is not None:
                    model.fit(X_train_arr, y_train_arr, sample_weight=sample_weights, eval_set=[(X_val_arr, y_val_arr)], verbose=False)
                else:
                    model.fit(X_train_arr, y_train_arr, eval_set=[(X_val_arr, y_val_arr)], verbose=False)
                    
            elif name == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1
                }
                model = LGBMClassifier(**params)
                if sample_weights is not None:
                    model.fit(X_train_arr, y_train_arr, sample_weight=sample_weights, eval_set=[(X_val_arr, y_val_arr)])
                else:
                    model.fit(X_train_arr, y_train_arr, eval_set=[(X_val_arr, y_val_arr)])
                    
            elif name == "catboost":
                params = {
                    "iterations": trial.suggest_int("iterations", 100, 1000),
                    "depth": trial.suggest_int("depth", 4, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 10.0, log=True),
                    "eval_metric": "F1",
                    "random_seed": 42,
                    "verbose": 0,
                    "auto_class_weights": "Balanced"
                }
                model = CatBoostClassifier(**params)
                cb_fit_kwargs = {"eval_set": (X_val_arr, y_val_arr), "use_best_model": True, "early_stopping_rounds": 50}
                if sample_weights is not None:
                    cb_fit_kwargs["sample_weight"] = sample_weights
                model.fit(X_train_arr, y_train_arr, **cb_fit_kwargs)
                
            elif name == "logistic_regression":
                params = {
                    "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                    "solver": trial.suggest_categorical("solver", ["lbfgs", "saga", "liblinear"]),
                    "max_iter": 5000,
                    "class_weight": "balanced",
                    "random_state": 42,
                }
                model = LogisticRegression(**params)
                if sample_weights is not None:
                    try:
                        model.fit(X_train_arr, y_train_arr, sample_weight=sample_weights)
                    except:
                        model.fit(X_train_arr, y_train_arr)
                else:
                    model.fit(X_train_arr, y_train_arr)
                    
            # Evaluate on validation set
            y_val_pred = model.predict(X_val_arr)
            score = f1_score(y_val_arr, y_val_pred, average="binary", zero_division=0)
            return score

        optuna_cfg = self.config.get("optuna_tuning", {"n_trials": 100, "direction": "maximize"})
        n_trials = optuna_cfg.get("n_trials", 100)
        direction = optuna_cfg.get("direction", "maximize")
        
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        logger.info("Best trial for '%s': %s", name, study.best_trial.params)
        
        # Re-build and re-train best model
        best_params = study.best_trial.params
        
        if name == "xgboost":
            best_params.update({"objective": "binary:logistic", "eval_metric": "aucpr", "tree_method": "hist", "random_state": 42, "n_jobs": -1})
            best_model = XGBClassifier(**best_params)
        elif name == "lightgbm":
            best_params.update({"class_weight": "balanced", "random_state": 42, "n_jobs": -1, "verbose": -1})
            best_model = LGBMClassifier(**best_params)
        elif name == "catboost":
            best_params.update({"eval_metric": "F1", "random_seed": 42, "verbose": 0, "auto_class_weights": "Balanced"})
            best_model = CatBoostClassifier(**best_params)
        elif name == "logistic_regression":
            best_params.update({"max_iter": 5000, "class_weight": "balanced", "random_state": 42})
            best_model = LogisticRegression(**best_params)

        self.models[name] = best_model
        return self.models[name]

    # ------------------------------------------------------------------
    # Feature Selection
    # ------------------------------------------------------------------

    def select_features(
        self, 
        X_train: ArrayLike, 
        y_train: ArrayLike, 
        method: str = "mutual_info", 
        k: Union[int, str] = 50
    ) -> Any:
        """
        Fit and return a feature selector.
        
        Parameters
        ----------
        method : "mutual_info", "select_k_best", "rfe", "xgboost"
        k : int or "all"
        """
        X_arr = np.array(X_train)
        y_arr = np.array(y_train)
        
        if k == "all":
            k = X_arr.shape[1]
            
        logger.info("Selecting top %s features using %s …", k, method)
        
        if method in ["mutual_info", "select_k_best"]:
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
            selector.fit(X_arr, y_arr)
        elif method == "rfe":
            estimator = DecisionTreeClassifier(max_depth=5, random_state=42)
            selector = RFE(estimator, n_features_to_select=k, step=0.1)
            selector.fit(X_arr, y_arr)
        elif method == "xgboost":
            # Using SelectFromModel as XGBoost feature selection
            from sklearn.feature_selection import SelectFromModel
            estimator = XGBClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
            estimator.fit(X_arr, y_arr)
            selector = SelectFromModel(estimator, max_features=k, prefit=True)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
            
        return selector

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        name: str,
        model: Any,
        X_test: ArrayLike,
        y_test: ArrayLike,
        threshold: float = 0.5,
    ) -> MetricsDict:
        """
        Compute a comprehensive set of classification metrics for one model.

        Metrics computed
        ----------------
        accuracy, precision, recall, f1, roc_auc, pr_auc, mcc,
        cohen_kappa, balanced_accuracy, confusion_matrix,
        inference_time_ms (per-sample, milliseconds)

        Parameters
        ----------
        name : str
            Model identifier (for logging).
        model : fitted estimator
        X_test, y_test : array-like
            Hold-out test data.
        threshold : float
            Decision threshold applied to probability scores.

        Returns
        -------
        dict
            Dictionary of metric names to values.
        """
        logger.info("Evaluating model '%s' on test set …", name)

        X_arr = np.array(X_test)
        y_arr = np.array(y_test)
        n_samples = len(y_arr)

        # -- Inference timing ------------------------------------------
        t0 = time.perf_counter()
        y_prob = _safe_predict_proba(model, X_arr)
        infer_elapsed = time.perf_counter() - t0
        inference_time_ms = (infer_elapsed / n_samples) * 1_000

        # -- Binary predictions ----------------------------------------
        if y_prob is not None:
            y_pred = (y_prob >= threshold).astype(int)
        else:
            t0 = time.perf_counter()
            y_pred = model.predict(X_arr)
            infer_elapsed = time.perf_counter() - t0
            inference_time_ms = (infer_elapsed / n_samples) * 1_000

        # -- Core metrics ----------------------------------------------
        acc = accuracy_score(y_arr, y_pred)
        prec = precision_score(y_arr, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_arr, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_arr, y_pred, average="binary", zero_division=0)
        bal_acc = balanced_accuracy_score(y_arr, y_pred)
        mcc = matthews_corrcoef(y_arr, y_pred)
        kappa = cohen_kappa_score(y_arr, y_pred)
        cm = confusion_matrix(y_arr, y_pred).tolist()

        # -- Probability-based metrics ---------------------------------
        roc_auc: Optional[float] = None
        pr_auc: Optional[float] = None
        if y_prob is not None:
            try:
                roc_auc = float(roc_auc_score(y_arr, y_prob))
            except ValueError:
                roc_auc = None
            try:
                pr_auc = _auc_pr(y_arr, y_prob)
            except ValueError:
                pr_auc = None

        metrics: MetricsDict = {
            "model": name,
            "accuracy": round(acc, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "roc_auc": round(roc_auc, 6) if roc_auc is not None else None,
            "pr_auc": round(pr_auc, 6) if pr_auc is not None else None,
            "mcc": round(mcc, 6),
            "cohen_kappa": round(kappa, 6),
            "balanced_accuracy": round(bal_acc, 6),
            "confusion_matrix": cm,
            "inference_time_ms_per_sample": round(inference_time_ms, 6),
            "train_time_s": round(self.train_times.get(name, float("nan")), 4),
            "threshold": threshold,
        }

        self.metrics[name] = metrics

        logger.info(
            "Results for '%s': acc=%.4f | prec=%.4f | rec=%.4f | "
            "f1=%.4f | roc_auc=%s | pr_auc=%s | mcc=%.4f",
            name,
            acc,
            prec,
            rec,
            f1,
            f"{roc_auc:.4f}" if roc_auc is not None else "N/A",
            f"{pr_auc:.4f}" if pr_auc is not None else "N/A",
            mcc,
        )
        return metrics

    def evaluate_all(
        self,
        X_test: ArrayLike,
        y_test: ArrayLike,
        threshold: float = 0.5,
    ) -> pd.DataFrame:
        """
        Evaluate every trained model and return a comparison DataFrame.

        Parameters
        ----------
        X_test, y_test : array-like
            Hold-out test data.
        threshold : float
            Decision threshold applied to probability scores.

        Returns
        -------
        pd.DataFrame
            One row per model, sorted descending by F1 score.
        """
        if not self.trained_models:
            raise RuntimeError(
                "No trained models found. Call train_all() or load_model() first."
            )

        logger.info(
            "Evaluating all %d trained models …", len(self.trained_models)
        )

        for name, model in self.trained_models.items():
            try:
                self.evaluate_single(name, model, X_test, y_test, threshold)
            except Exception as exc:  # noqa: BLE001
                logger.error("Evaluation failed for '%s': %s", name, exc)

        return self.get_comparison_table()

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(
        self,
        name: str,
        feature_names: List[str],
        X_val: Optional[ArrayLike] = None,
        y_val: Optional[ArrayLike] = None,
        n_repeats: int = 10,
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Return a sorted feature-importance DataFrame for the named model.

        Strategy per model family
        -------------------------
        * Tree-based (RF, ET, DT, XGB, LGBM, CB) → ``feature_importances_``
        * Logistic Regression                     → ``|coef_[0]|``
        * SVM                                     → permutation importance
          (requires X_val, y_val)

        Parameters
        ----------
        name : str
            Model key.
        feature_names : list[str]
            Ordered list of feature names matching training columns.
        X_val, y_val : array-like | None
            Validation data needed only for SVM permutation importance.
        n_repeats : int
            Repetitions for permutation importance (SVM only).
        random_state : int
            RNG seed for permutation importance.

        Returns
        -------
        pd.DataFrame
            Columns: ``feature``, ``importance`` – sorted descending.
        """
        if name not in self.trained_models:
            raise KeyError(
                f"Model '{name}' is not in trained_models. "
                "Train or load it first."
            )

        model = self.trained_models[name]
        logger.info("Computing feature importance for '%s' …", name)

        importances: Optional[np.ndarray] = None

        # Tree-based models
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_

        # Linear models (coefficients)
        elif hasattr(model, "coef_"):
            coef = np.array(model.coef_)
            importances = np.abs(coef).mean(axis=0)

        # SVM / other: permutation importance
        else:
            if X_val is None or y_val is None:
                raise ValueError(
                    "X_val and y_val must be provided to compute permutation "
                    f"importance for model '{name}'."
                )
            logger.info(
                "Using permutation importance for '%s' (n_repeats=%d) …",
                name,
                n_repeats,
            )
            perm_result = permutation_importance(
                model,
                np.array(X_val),
                np.array(y_val),
                n_repeats=n_repeats,
                random_state=random_state,
                scoring="f1",
                n_jobs=-1,
            )
            importances = perm_result.importances_mean

        if importances is None:
            raise RuntimeError(
                f"Could not determine feature importances for '{name}'."
            )

        df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info(
            "Top-5 features for '%s': %s",
            name,
            df["feature"].head(5).tolist(),
        )
        return df

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate_model(
        self,
        name: str,
        model: Any,
        X: ArrayLike,
        y: ArrayLike,
        cv: int = 5,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> Dict[str, float]:
        """
        Stratified K-Fold cross-validation for a single model.

        Metrics averaged across folds: accuracy, precision, recall, f1,
        roc_auc, mcc, balanced_accuracy.

        Parameters
        ----------
        name : str
            Human-readable model identifier for logging.
        model : estimator
            An *unfitted* sklearn-compatible estimator (or a fresh clone).
        X, y : array-like
            Full dataset (features and labels).
        cv : int
            Number of folds.
        class_weights : dict | None
            Optional class-weight mapping.

        Returns
        -------
        dict
            Keys: ``{metric}_mean`` and ``{metric}_std`` for each metric.
        """
        logger.info(
            "Cross-validating '%s' with %d-fold StratifiedKFold …", name, cv
        )

        X_arr = np.array(X)
        y_arr = np.array(y)

        cv_cfg = self.config.get("cross_validation", {})
        skf = StratifiedKFold(
            n_splits=cv,
            shuffle=cv_cfg.get("shuffle", True),
            random_state=cv_cfg.get("random_state", 42),
        )

        fold_metrics: Dict[str, List[float]] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
            "mcc": [],
            "balanced_accuracy": [],
        }

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_arr, y_arr), start=1
        ):
            X_tr, X_vl = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_vl = y_arr[train_idx], y_arr[val_idx]

            sample_weights: Optional[np.ndarray] = None
            if class_weights:
                sample_weights = compute_sample_weight(
                    class_weight=class_weights, y=y_tr
                )

            from sklearn.base import clone  # local to avoid circular

            fold_model = clone(model)

            try:
                if name == "catboost":
                    fit_kwargs: Dict[str, Any] = {}
                    if sample_weights is not None:
                        fit_kwargs["sample_weight"] = sample_weights
                    fold_model.fit(X_tr, y_tr, **fit_kwargs)
                elif sample_weights is not None:
                    try:
                        fold_model.fit(
                            X_tr, y_tr, sample_weight=sample_weights
                        )
                    except TypeError:
                        fold_model.fit(X_tr, y_tr)
                else:
                    fold_model.fit(X_tr, y_tr)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Fold %d training failed for '%s': %s", fold_idx, name, exc
                )
                continue

            y_pred = fold_model.predict(X_vl)
            y_prob = _safe_predict_proba(fold_model, X_vl)

            fold_metrics["accuracy"].append(accuracy_score(y_vl, y_pred))
            fold_metrics["precision"].append(
                precision_score(y_vl, y_pred, average="binary", zero_division=0)
            )
            fold_metrics["recall"].append(
                recall_score(y_vl, y_pred, average="binary", zero_division=0)
            )
            fold_metrics["f1"].append(
                f1_score(y_vl, y_pred, average="binary", zero_division=0)
            )
            fold_metrics["mcc"].append(matthews_corrcoef(y_vl, y_pred))
            fold_metrics["balanced_accuracy"].append(
                balanced_accuracy_score(y_vl, y_pred)
            )

            if y_prob is not None:
                try:
                    fold_metrics["roc_auc"].append(
                        roc_auc_score(y_vl, y_prob)
                    )
                except ValueError:
                    pass
            else:
                fold_metrics["roc_auc"].append(float("nan"))

            logger.debug(
                "Fold %d/%d | '%s' | f1=%.4f | roc_auc=%s",
                fold_idx,
                cv,
                name,
                fold_metrics["f1"][-1],
                (
                    f"{fold_metrics['roc_auc'][-1]:.4f}"
                    if fold_metrics["roc_auc"]
                    else "N/A"
                ),
            )

        summary: Dict[str, float] = {}
        for metric, values in fold_metrics.items():
            arr = np.array(values, dtype=float)
            summary[f"{metric}_mean"] = float(np.nanmean(arr))
            summary[f"{metric}_std"] = float(np.nanstd(arr))

        logger.info(
            "CV results for '%s': f1_mean=%.4f±%.4f | "
            "roc_auc_mean=%.4f±%.4f",
            name,
            summary["f1_mean"],
            summary["f1_std"],
            summary["roc_auc_mean"],
            summary["roc_auc_std"],
        )
        return summary

    # ------------------------------------------------------------------
    # Hyperparameter tuning (Optuna)
    # ------------------------------------------------------------------

    def hyperparameter_tune(
        self,
        name: str,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
        n_trials: int = 50,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run Optuna hyperparameter search for the specified model.

        Each trial trains the model on ``X_train`` / ``y_train`` and
        evaluates F1 on ``X_val`` / ``y_val``.

        Parameters
        ----------
        name : str
            One of the eight model keys.
        X_train, y_train : array-like
            Training data.
        X_val, y_val : array-like
            Validation data used for objective evaluation.
        n_trials : int
            Number of Optuna trials.
        class_weights : dict | None
            Optional class-weight mapping.

        Returns
        -------
        (fitted_model, best_params)
            The best model retrained on the full training set, and the
            dictionary of best hyperparameters.
        """
        logger.info(
            "Starting Optuna hyperparameter search for '%s' "
            "(%d trials) …",
            name,
            n_trials,
        )

        X_tr = np.array(X_train)
        y_tr = np.array(y_train)
        X_vl = np.array(X_val)
        y_vl = np.array(y_val)

        sample_weights: Optional[np.ndarray] = None
        if class_weights:
            sample_weights = compute_sample_weight(
                class_weight=class_weights, y=y_tr
            )

        # ── Search space per model ────────────────────────────────────
        def _suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
            if name == "logistic_regression":
                return {
                    "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
                    "solver": trial.suggest_categorical(
                        "solver", ["lbfgs", "saga"]
                    ),
                    "max_iter": 1000,
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1,
                }
            elif name == "decision_tree":
                return {
                    "max_depth": trial.suggest_int("max_depth", 2, 30),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split", 2, 50
                    ),
                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf", 1, 30
                    ),
                    "class_weight": "balanced",
                    "random_state": 42,
                }
            elif name == "random_forest":
                return {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 50, 500
                    ),
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split", 2, 20
                    ),
                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf", 1, 10
                    ),
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1,
                }
            elif name == "extra_trees":
                return {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 50, 500
                    ),
                    "max_depth": trial.suggest_int("max_depth", 3, 30),
                    "min_samples_split": trial.suggest_int(
                        "min_samples_split", 2, 20
                    ),
                    "min_samples_leaf": trial.suggest_int(
                        "min_samples_leaf", 1, 10
                    ),
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1,
                }
            elif name == "xgboost":
                return {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 50, 500
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-3, 0.3, log=True
                    ),
                    "max_depth": trial.suggest_int("max_depth", 2, 12),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                    "reg_alpha": trial.suggest_float(
                        "reg_alpha", 1e-8, 10.0, log=True
                    ),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 1e-8, 10.0, log=True
                    ),
                    "eval_metric": "logloss",
                    "random_state": 42,
                    "n_jobs": -1,
                }
            elif name == "lightgbm":
                return {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", 50, 500
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-3, 0.3, log=True
                    ),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "max_depth": trial.suggest_int("max_depth", -1, 20),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float(
                        "colsample_bytree", 0.5, 1.0
                    ),
                    "reg_alpha": trial.suggest_float(
                        "reg_alpha", 1e-8, 10.0, log=True
                    ),
                    "reg_lambda": trial.suggest_float(
                        "reg_lambda", 1e-8, 10.0, log=True
                    ),
                    "class_weight": "balanced",
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1,
                }
            elif name == "catboost":
                return {
                    "iterations": trial.suggest_int("iterations", 50, 500),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-3, 0.3, log=True
                    ),
                    "depth": trial.suggest_int("depth", 3, 10),
                    "l2_leaf_reg": trial.suggest_float(
                        "l2_leaf_reg", 1e-3, 10.0, log=True
                    ),
                    "random_seed": 42,
                    "verbose": 0,
                    "auto_class_weights": "Balanced",
                }
            elif name == "svm":
                kernel = trial.suggest_categorical(
                    "kernel", ["rbf", "poly", "sigmoid"]
                )
                params: Dict[str, Any] = {
                    "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
                    "kernel": kernel,
                    "class_weight": "balanced",
                    "probability": True,
                    "random_state": 42,
                }
                if kernel == "rbf":
                    params["gamma"] = trial.suggest_categorical(
                        "gamma", ["scale", "auto"]
                    )
                elif kernel == "poly":
                    params["degree"] = trial.suggest_int("degree", 2, 5)
                return params
            else:
                raise ValueError(f"Unknown model name for tuning: '{name}'")

        # ── Model factory ─────────────────────────────────────────────
        _FACTORY = {
            "logistic_regression": LogisticRegression,
            "decision_tree": DecisionTreeClassifier,
            "random_forest": RandomForestClassifier,
            "extra_trees": ExtraTreesClassifier,
            "xgboost": XGBClassifier,
            "lightgbm": LGBMClassifier,
            "catboost": CatBoostClassifier,
            "svm": SVC,
        }

        if name not in _FACTORY:
            raise ValueError(
                f"Hyperparameter tuning is not defined for model '{name}'. "
                f"Supported: {list(_FACTORY.keys())}"
            )

        # ── Objective ─────────────────────────────────────────────────
        def objective(trial: optuna.Trial) -> float:
            params = _suggest_params(trial)
            trial_model = _FACTORY[name](**params)

            try:
                if name == "catboost":
                    fit_kw: Dict[str, Any] = {}
                    if sample_weights is not None:
                        fit_kw["sample_weight"] = sample_weights
                    trial_model.fit(X_tr, y_tr, **fit_kw)
                elif sample_weights is not None:
                    try:
                        trial_model.fit(
                            X_tr, y_tr, sample_weight=sample_weights
                        )
                    except TypeError:
                        trial_model.fit(X_tr, y_tr)
                else:
                    trial_model.fit(X_tr, y_tr)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "Trial %d failed: %s", trial.number, exc
                )
                raise optuna.exceptions.TrialPruned()

            y_pred = trial_model.predict(X_vl)
            score = f1_score(y_vl, y_pred, average="binary", zero_division=0)
            return score

        # ── Study ─────────────────────────────────────────────────────
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
        )

        best_params = study.best_params
        logger.info(
            "Best params for '%s': %s | best_f1=%.4f",
            name,
            best_params,
            study.best_value,
        )

        # Retrain best model on training data
        best_model = _FACTORY[name](**best_params)
        try:
            if name == "catboost":
                fit_kw = {}
                if sample_weights is not None:
                    fit_kw["sample_weight"] = sample_weights
                best_model.fit(X_tr, y_tr, **fit_kw)
            elif sample_weights is not None:
                try:
                    best_model.fit(X_tr, y_tr, sample_weight=sample_weights)
                except TypeError:
                    best_model.fit(X_tr, y_tr)
            else:
                best_model.fit(X_tr, y_tr)
        except Exception as exc:
            logger.error(
                "Failed to refit best model '%s': %s", name, exc, exc_info=True
            )
            raise

        self.trained_models[name] = best_model
        self.save_model(name, best_model)

        return best_model, best_params

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_model(self, name: str, model: Any) -> Path:
        """
        Serialise ``model`` to ``{output_dir}/models/{name}.pkl``.

        Parameters
        ----------
        name : str
            Model key used as the file stem.
        model : fitted estimator
            The object to serialise.

        Returns
        -------
        Path
            Absolute path of the saved file.
        """
        save_path = self.models_dir / f"{name}.pkl"
        try:
            joblib.dump(model, save_path)
            logger.info("Saved model '%s' to '%s'.", name, save_path)
        except Exception as exc:
            logger.error(
                "Failed to save model '%s': %s", name, exc, exc_info=True
            )
            raise
        return save_path

    def load_model(self, name: str) -> Any:
        """
        Deserialise model from ``{output_dir}/models/{name}.pkl``.

        The loaded model is stored in ``self.trained_models[name]``.

        Parameters
        ----------
        name : str
            Model key / file stem.

        Returns
        -------
        estimator
            The deserialised fitted model.

        Raises
        ------
        FileNotFoundError
            If no ``.pkl`` file exists for ``name``.
        """
        load_path = self.models_dir / f"{name}.pkl"
        if not load_path.is_file():
            raise FileNotFoundError(
                f"No saved model found for '{name}' at '{load_path}'."
            )
        try:
            model = joblib.load(load_path)
            self.trained_models[name] = model
            logger.info("Loaded model '%s' from '%s'.", name, load_path)
        except Exception as exc:
            logger.error(
                "Failed to load model '%s': %s", name, exc, exc_info=True
            )
            raise
        return model

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Return a DataFrame comparing all evaluated models.

        Columns: model, accuracy, precision, recall, f1, roc_auc, pr_auc,
                 mcc, cohen_kappa, balanced_accuracy,
                 inference_time_ms_per_sample, train_time_s.

        Rows are sorted descending by ``f1``.

        Returns
        -------
        pd.DataFrame
        """
        if not self.metrics:
            raise RuntimeError(
                "No evaluation results found. "
                "Call evaluate_all() or evaluate_single() first."
            )

        display_cols = [
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
            "inference_time_ms_per_sample",
            "train_time_s",
        ]

        rows = []
        for name, m in self.metrics.items():
            row = {col: m.get(col) for col in display_cols}
            row["model"] = name
            rows.append(row)

        df = (
            pd.DataFrame(rows, columns=display_cols)
            .sort_values("f1", ascending=False)
            .reset_index(drop=True)
        )

        logger.info(
            "Comparison table built (%d models). Best F1: %.4f (%s)",
            len(df),
            df["f1"].iloc[0] if len(df) > 0 else float("nan"),
            df["model"].iloc[0] if len(df) > 0 else "N/A",
        )
        return df

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        trained = list(self.trained_models.keys())
        return (
            f"BaselineModelSuite("
            f"output_dir='{self.output_dir}', "
            f"trained={trained})"
        )

    def __len__(self) -> int:
        return len(self.models)


# ---------------------------------------------------------------------------
# Script entry-point (smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test BaselineModelSuite on synthetic data."
    )
    parser.add_argument("--config", default=None, help="Path to model_config.yaml")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--samples", type=int, default=2_000, help="Synthetic sample count")
    parser.add_argument("--features", type=int, default=20, help="Feature count")
    args = parser.parse_args()

    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    logging.getLogger().setLevel(logging.INFO)

    logger.info("Generating synthetic dataset …")
    X_all, y_all = make_classification(
        n_samples=args.samples,
        n_features=args.features,
        n_informative=10,
        n_redundant=5,
        weights=[0.85, 0.15],  # Imbalanced
        random_state=42,
    )
    feat_names = [f"feature_{i}" for i in range(args.features)]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.15, stratify=y_tmp, random_state=42
    )

    suite = BaselineModelSuite(config_path=args.config, output_dir=args.output)

    logger.info("Training all models …")
    suite.train_all(X_train, y_train, X_val, y_val)

    logger.info("Evaluating all models …")
    comparison = suite.evaluate_all(X_test, y_test)
    print("\n" + comparison.to_string(index=False))

    logger.info("Feature importance for random_forest …")
    fi = suite.get_feature_importance("random_forest", feat_names)
    print("\nTop-10 features (Random Forest):")
    print(fi.head(10).to_string(index=False))

    logger.info("Smoke test completed successfully.")
