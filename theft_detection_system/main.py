"""
Electricity Theft Detection System — Main Entry Point
======================================================
Usage:
  python main.py --mode baseline          # train all baseline models
  python main.py --mode deep --model cnn  # train a single DL model
  python main.py --mode all              # run full pipeline
  python main.py --mode tune --model lstm # Optuna hyperparameter search
  python main.py --mode explain           # run explainability on best model
  python main.py --mode evaluate          # evaluate all saved models

CLI args override YAML config values.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.data_loader import DataLoader
from src.preprocessing.feature_engineering import FeatureEngineer
from src.preprocessing.imbalance_handler import ImbalanceHandler
from src.models.baseline_models import BaselineModelSuite
from src.evaluation.evaluator import Evaluator


# =============================================================================
#  Logging Setup
# =============================================================================

def setup_logging(config: dict) -> logging.Logger:
    log_cfg   = config.get("logging", {})
    log_level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
    log_file  = log_cfg.get("log_file", "outputs/logs/experiment.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    handlers: list[logging.Handler] = []
    if log_cfg.get("console", True):
        handlers.append(logging.StreamHandler(sys.stdout))
    handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s  %(name)-25s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("main")


# =============================================================================
#  Config Loader
# =============================================================================

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# =============================================================================
#  Device Setup
# =============================================================================

def get_device(config: dict) -> torch.device:
    device_cfg = config.get("project", {}).get("device", "auto")
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_cfg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # A100 optimizations
    a100 = config.get("training_config", {})
    if device.type == "cuda":
        allow_tf32 = a100.get("allow_tf32", True)
        torch.backends.cuda.matmul.allow_tf32  = allow_tf32
        torch.backends.cudnn.allow_tf32        = allow_tf32
        torch.backends.cudnn.benchmark         = a100.get("cudnn_benchmark", True)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU     : {gpu_name}")
            print(f"  VRAM    : {vram_gb:.1f} GB")

    return device


# =============================================================================
#  Seed
# =============================================================================

def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
#  Data Pipeline
# =============================================================================

def run_data_pipeline(
    config: dict,
    logger: logging.Logger,
    feature_engineering: bool = True,
) -> tuple:
    """
    Load, preprocess, feature-engineer, and split the SGCC dataset.
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, feature_names).
    """
    data_cfg  = config["data"]
    prep_cfg  = config["preprocessing"]
    fe_cfg    = config["feature_engineering"]

    logger.info("=" * 60)
    logger.info("  DATA PIPELINE")
    logger.info("=" * 60)

    # Load and preprocess
    loader = DataLoader(random_state=data_cfg.get("random_state", 42))

    # Check cache
    cache_path = data_cfg.get("cache_path")
    if data_cfg.get("cache_processed") and cache_path and os.path.isfile(cache_path):
        logger.info(f"Loading cached processed data from {cache_path}")
        loader.load_processed(cache_path)
    else:
        loader.load_data(data_cfg["path"])
        loader.validate_data()
        if prep_cfg.get("remove_duplicates", True):
            loader.remove_duplicates()
        loader.handle_missing_values(prep_cfg.get("missing_value_strategy", "interpolate"))
        loader.detect_and_handle_outliers(
            method    = prep_cfg.get("outlier_method", "iqr"),
            threshold = prep_cfg.get("outlier_threshold", 3.0),
        )
        loader.split_data(
            test_size    = data_cfg.get("test_size", 0.15),
            val_size     = data_cfg.get("val_size", 0.15),
            stratify     = data_cfg.get("stratify", True),
            random_state = data_cfg.get("random_state", 42),
        )
        loader.scale_features(prep_cfg.get("scaler", "standard"))

        if cache_path and data_cfg.get("cache_processed"):
            loader.save_processed(cache_path)

    loader.get_data_summary()

    X_train, X_val, X_test, y_train, y_val, y_test = loader.get_numpy_arrays()
    feature_names = loader.get_feature_columns()

    # Feature Engineering
    if feature_engineering:
        logger.info("Running feature engineering...")
        fe = FeatureEngineer(config=fe_cfg)
        X_train_fe = fe.fit_transform(X_train)
        X_val_fe   = fe.transform(X_val)
        X_test_fe  = fe.transform(X_test)
        feature_names_fe = fe.get_feature_names()
        logger.info(f"Feature engineering: {X_train.shape[1]} -> {X_train_fe.shape[1]} features")
        return X_train_fe, X_val_fe, X_test_fe, y_train, y_val, y_test, feature_names_fe

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names


# =============================================================================
#  Imbalance Handling
# =============================================================================

def run_imbalance_handling(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, dict]:
    imb_cfg = config["imbalance"]
    handler = ImbalanceHandler(method=imb_cfg["method"], config=imb_cfg)

    handler.print_class_distribution(y_train, label="Before resampling")
    X_train_r, y_train_r = handler.resample(X_train, y_train)
    handler.print_class_distribution(y_train_r, label="After resampling")

    class_weights = handler.compute_class_weights(y_train)
    pos_weight    = handler.compute_pos_weight(y_train)

    logger.info(f"Imbalance method : {imb_cfg['method']}")
    logger.info(f"pos_weight       : {pos_weight:.4f}")
    logger.info(f"class_weights    : {class_weights}")

    return X_train_r, y_train_r, {"class_weights": class_weights, "pos_weight": pos_weight}


# =============================================================================
#  Baseline Mode
# =============================================================================

def run_baseline_mode(
    config: dict,
    logger: logging.Logger,
    output_dir: str,
) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("  MODE: BASELINE ML MODELS")
    logger.info("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = run_data_pipeline(
        config, logger, feature_engineering=True
    )
    X_train_r, y_train_r, weights = run_imbalance_handling(
        X_train, y_train, config, logger
    )

    suite = BaselineModelSuite(
        config_path = "config/model_config.yaml",
        output_dir  = output_dir,
    )

    logger.info("Training all baseline models...")
    suite.train_all(
        X_train_r, y_train_r,
        X_val, y_val,
        class_weights=weights["class_weights"],
    )

    logger.info("Evaluating all baseline models on test set...")
    comparison = suite.evaluate_all(X_test, y_test)

    logger.info("\n" + "=" * 60)
    logger.info("  BASELINE MODEL COMPARISON TABLE")
    logger.info("=" * 60)
    logger.info("\n" + comparison.to_string())

    # Save comparison table
    os.makedirs(f"{output_dir}/reports", exist_ok=True)
    comparison.to_csv(f"{output_dir}/reports/baseline_comparison.csv", index=False)
    logger.info(f"Saved: {output_dir}/reports/baseline_comparison.csv")

    # Feature importance for tree models
    evaluator = Evaluator(config=config["evaluation"])
    for name in ["random_forest", "xgboost", "lightgbm", "catboost"]:
        try:
            model = suite.load_model(name)
            fi_df = suite.get_feature_importance(name, feature_names)
            if fi_df is not None:
                plot_path = f"{output_dir}/plots/{name}_feature_importance.png"
                os.makedirs(f"{output_dir}/plots", exist_ok=True)
                fi_df.head(20).plot(
                    kind="barh", x="feature", y="importance",
                    title=f"{name} Feature Importance (Top 20)",
                    figsize=(10, 8),
                ).figure.savefig(plot_path, bbox_inches="tight", dpi=150)
                logger.info(f"Saved feature importance: {plot_path}")
        except Exception as e:
            logger.warning(f"Could not compute feature importance for {name}: {e}")


# =============================================================================
#  Deep Learning Mode
# =============================================================================

def run_deep_mode(
    model_name: str,
    config: dict,
    logger: logging.Logger,
    output_dir: str,
    device: torch.device,
) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(f"  MODE: DEEP LEARNING — {model_name.upper()}")
    logger.info(f"{'='*60}")

    from src.training.trainer import Trainer
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = run_data_pipeline(
        config, logger, feature_engineering=False  # Use raw sequence for DL
    )
    X_train_r, y_train_r, weights = run_imbalance_handling(
        X_train, y_train, config, logger
    )

    input_dim = X_train.shape[1]
    seq_len   = 1  # tabular input treated as single-step sequence

    # Build model
    model = _build_dl_model(model_name, input_dim, seq_len, config)
    logger.info(f"Model: {model_name}  |  Params: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device)

    # DataLoaders
    train_cfg = config["training"]
    batch_size = train_cfg.get("batch_size", 256)

    def _make_loader(X, y, shuffle):
        Xt = torch.FloatTensor(X)
        yt = torch.FloatTensor(y)
        if len(Xt.shape) == 2:
            Xt = Xt.unsqueeze(1)  # (B, 1, features) for sequence models
        return DataLoader(
            TensorDataset(Xt, yt),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
        )

    train_loader = _make_loader(X_train_r, y_train_r, shuffle=True)
    val_loader   = _make_loader(X_val, y_val, shuffle=False)
    test_loader  = _make_loader(X_test, y_test, shuffle=False)

    # Loss
    pos_weight = torch.tensor([weights["pos_weight"]], device=device)
    imb_method = config["imbalance"]["method"]
    if imb_method == "focal_loss":
        from src.training.trainer import FocalLoss
        alpha = config["imbalance"].get("focal_alpha", 0.25)
        gamma = config["imbalance"].get("focal_gamma", 2.0)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
    else:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight if imb_method == "weighted_loss" else None
        )

    # Train
    trainer = Trainer(
        model      = model,
        config     = config,
        device     = device,
        output_dir = output_dir,
        model_name = model_name,
    )
    history = trainer.train(
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = train_cfg.get("epochs", 100),
        criterion    = criterion,
    )

    # Evaluate
    evaluator = Evaluator(config=config["evaluation"])
    model.eval()
    all_probs = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            out = model(Xb.to(device)).squeeze().cpu().numpy()
            if out.ndim == 0:
                out = out.reshape(1)
            all_probs.extend(out.tolist())

    y_prob = np.array(all_probs)
    metrics = evaluator.compute_metrics(y_test, y_prob)

    logger.info(f"\n{model_name} Test Results:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            logger.info(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save plots
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    evaluator.plot_roc_curve(y_test, y_prob, model_name,
                             f"{output_dir}/plots/{model_name}_roc.png")
    evaluator.plot_pr_curve(y_test, y_prob, model_name,
                            f"{output_dir}/plots/{model_name}_pr.png")
    evaluator.plot_confusion_matrix(y_test, (y_prob >= 0.5).astype(int), model_name,
                                    f"{output_dir}/plots/{model_name}_cm.png")
    evaluator.plot_loss_curves(history, model_name,
                               f"{output_dir}/plots/{model_name}_loss.png")

    logger.info(f"Plots saved to {output_dir}/plots/")


def _build_dl_model(
    name: str,
    input_dim: int,
    seq_len: int,
    config: dict,
) -> torch.nn.Module:
    mc = config.get("model_config", {}).get("deep_learning", {})

    if name == "cnn":
        from src.models.deep_learning.cnn import CNN1D
        cfg = mc.get("cnn", {})
        return CNN1D(input_dim=input_dim, seq_len=seq_len,
                     channels=cfg.get("channels", [64, 128, 256]),
                     kernel_sizes=cfg.get("kernel_sizes", [3, 3, 3]),
                     dropout=cfg.get("dropout", 0.3),
                     batch_norm=cfg.get("batch_norm", True))
    elif name == "lstm":
        from src.models.deep_learning.lstm import LSTMClassifier
        cfg = mc.get("lstm", {})
        return LSTMClassifier(input_dim=input_dim,
                              hidden_dim=cfg.get("hidden_dim", 128),
                              num_layers=cfg.get("num_layers", 2),
                              dropout=cfg.get("dropout", 0.3))
    elif name == "gru":
        from src.models.deep_learning.gru import GRUClassifier
        cfg = mc.get("gru", {})
        return GRUClassifier(input_dim=input_dim,
                             hidden_dim=cfg.get("hidden_dim", 128),
                             num_layers=cfg.get("num_layers", 2),
                             dropout=cfg.get("dropout", 0.3))
    elif name == "bilstm":
        from src.models.deep_learning.bilstm import BiLSTMClassifier
        cfg = mc.get("bilstm", {})
        return BiLSTMClassifier(input_dim=input_dim,
                                hidden_dim=cfg.get("hidden_dim", 128),
                                num_layers=cfg.get("num_layers", 2),
                                dropout=cfg.get("dropout", 0.3))
    elif name == "cnn_lstm":
        from src.models.deep_learning.cnn_lstm import CNNLSTMClassifier
        cfg = mc.get("cnn_lstm", {})
        return CNNLSTMClassifier(input_dim=input_dim,
                                 cnn_channels=cfg.get("cnn_channels", [64, 128]),
                                 kernel_size=cfg.get("kernel_size", 3),
                                 lstm_hidden=cfg.get("lstm_hidden", 128),
                                 lstm_layers=cfg.get("lstm_layers", 2),
                                 dropout=cfg.get("dropout", 0.3))
    elif name == "cnn_gru":
        from src.models.deep_learning.cnn_gru import CNNGRUClassifier
        cfg = mc.get("cnn_gru", {})
        return CNNGRUClassifier(input_dim=input_dim,
                                cnn_channels=cfg.get("cnn_channels", [64, 128]),
                                kernel_size=cfg.get("kernel_size", 3),
                                gru_hidden=cfg.get("gru_hidden", 128),
                                gru_layers=cfg.get("gru_layers", 2),
                                dropout=cfg.get("dropout", 0.3))
    elif name == "transformer":
        from src.models.deep_learning.transformer import TransformerClassifier
        cfg = mc.get("transformer", {})
        return TransformerClassifier(input_dim=input_dim,
                                     d_model=cfg.get("d_model", 128),
                                     nhead=cfg.get("nhead", 8),
                                     num_encoder_layers=cfg.get("num_encoder_layers", 4),
                                     dim_feedforward=cfg.get("dim_feedforward", 512),
                                     dropout=cfg.get("dropout", 0.1))
    elif name == "tcn":
        from src.models.deep_learning.tcn import TCNClassifier
        cfg = mc.get("tcn", {})
        return TCNClassifier(input_dim=input_dim,
                             num_channels=cfg.get("num_channels", [64, 128, 256]),
                             kernel_size=cfg.get("kernel_size", 3),
                             dropout=cfg.get("dropout", 0.2))
    elif name == "autoencoder":
        from src.models.deep_learning.autoencoder import AutoencoderDetector
        cfg = mc.get("autoencoder", {})
        return AutoencoderDetector(input_dim=input_dim,
                                   encoder_dims=cfg.get("encoder_dims", [256, 128, 64]),
                                   latent_dim=cfg.get("latent_dim", 32),
                                   dropout=cfg.get("dropout", 0.2))
    else:
        raise ValueError(f"Unknown deep learning model: {name}")


# =============================================================================
#  Tune Mode
# =============================================================================

def run_tune_mode(
    model_name: str,
    config: dict,
    logger: logging.Logger,
    output_dir: str,
    device: torch.device,
) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(f"  MODE: HYPERPARAMETER TUNING — {model_name.upper()}")
    logger.info(f"{'='*60}")

    from src.training.hyperparameter_tuning import HyperparameterTuner
    import importlib

    X_train, X_val, X_test, y_train, y_val, y_test, _ = run_data_pipeline(
        config, logger, feature_engineering=False
    )
    X_train_r, y_train_r, _ = run_imbalance_handling(X_train, y_train, config, logger)

    tune_cfg = config["hyperparameter_tuning"]
    model_cls_map = {
        "cnn":         "src.models.deep_learning.cnn.CNN1D",
        "lstm":        "src.models.deep_learning.lstm.LSTMClassifier",
        "gru":         "src.models.deep_learning.gru.GRUClassifier",
        "bilstm":      "src.models.deep_learning.bilstm.BiLSTMClassifier",
        "cnn_lstm":    "src.models.deep_learning.cnn_lstm.CNNLSTMClassifier",
        "cnn_gru":     "src.models.deep_learning.cnn_gru.CNNGRUClassifier",
        "transformer": "src.models.deep_learning.transformer.TransformerClassifier",
        "tcn":         "src.models.deep_learning.tcn.TCNClassifier",
    }
    if model_name not in model_cls_map:
        raise ValueError(f"Cannot tune model: {model_name}")

    mod_path, cls_name = model_cls_map[model_name].rsplit(".", 1)
    module    = importlib.import_module(mod_path)
    model_cls = getattr(module, cls_name)

    tuner = HyperparameterTuner(
        model_class = model_cls,
        X_train     = X_train_r,
        y_train     = y_train_r,
        X_val       = X_val,
        y_val       = y_val,
        config      = config,
        device      = device,
        n_trials    = tune_cfg.get("n_trials", 100),
        direction   = tune_cfg.get("direction", "maximize"),
        study_name  = tune_cfg.get("study_name", "study"),
        storage     = tune_cfg.get("storage", None),
    )

    study = tuner.tune()
    best  = tuner.get_best_params()

    logger.info(f"Best params: {best}")
    logger.info(f"Best val F1: {study.best_value:.4f}")

    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    tuner.plot_optimization_history(f"{output_dir}/plots/{model_name}_optuna_history.png")
    tuner.plot_param_importances(f"{output_dir}/plots/{model_name}_optuna_params.png")


# =============================================================================
#  Explain Mode
# =============================================================================

def run_explain_mode(
    config: dict,
    logger: logging.Logger,
    output_dir: str,
) -> None:
    logger.info(f"\n{'='*60}")
    logger.info(f"  MODE: EXPLAINABILITY")
    logger.info(f"{'='*60}")

    from src.explainability.explainer import Explainer
    import joblib

    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = run_data_pipeline(
        config, logger, feature_engineering=True
    )

    # Load best baseline model (Random Forest as default)
    model_path = os.path.join(output_dir, "models", "random_forest.pkl")
    if not os.path.isfile(model_path):
        logger.warning(f"Model not found: {model_path}. Run --mode baseline first.")
        return

    model   = joblib.load(model_path)
    explain = Explainer(config=config.get("explainability", {}))

    os.makedirs(f"{output_dir}/plots/explanations", exist_ok=True)

    # SHAP
    shap_vals = explain.explain_with_shap(
        model, X_train[:200], X_test[:100], model_type="tree"
    )
    explain.plot_shap_summary(
        shap_vals, feature_names,
        f"{output_dir}/plots/explanations/shap_summary.png"
    )
    explain.plot_shap_waterfall(
        shap_vals, idx=0, feature_names=feature_names,
        save_path=f"{output_dir}/plots/explanations/shap_waterfall_0.png"
    )

    # Permutation importance
    pi_df = explain.compute_permutation_importance(model, X_val, y_val, feature_names)
    explain.plot_permutation_importance(
        pi_df, f"{output_dir}/plots/explanations/permutation_importance.png"
    )

    # Batch explanation report
    explain.explain_batch(
        model, X_test[:50], y_test[:50], feature_names,
        model_type="tree", n_samples=50
    )

    logger.info(f"Explanation plots saved to {output_dir}/plots/explanations/")


# =============================================================================
#  CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Electricity Theft Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", type=str, default="baseline",
        choices=["baseline", "deep", "all", "tune", "explain", "evaluate"],
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--model", type=str, default="cnn_lstm",
        choices=["cnn", "lstm", "gru", "bilstm", "cnn_lstm", "cnn_gru",
                 "transformer", "tcn", "autoencoder"],
        help="Deep learning model to use (for --mode deep or tune)",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Override data path from config",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Directory for all outputs",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--imbalance", type=str, default=None,
        choices=["weighted_loss", "smote", "adasyn", "oversample",
                 "undersample", "focal_loss", "none"],
        help="Override imbalance handling method",
    )
    return parser.parse_args()


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI argument overrides to loaded config."""
    if args.data:
        config["data"]["path"] = args.data
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.imbalance:
        config["imbalance"]["method"] = args.imbalance
    return config


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    args   = parse_args()
    config = load_config(args.config)

    # Also load model_config and training_config and merge
    with open("config/model_config.yaml")   as f: config["model_config"]    = yaml.safe_load(f)
    with open("config/training_config.yaml") as f: config["training_config"] = yaml.safe_load(f)

    config = apply_cli_overrides(config, args)

    # Create output directories
    for d in ["outputs/models", "outputs/plots", "outputs/reports", "outputs/logs"]:
        os.makedirs(d, exist_ok=True)

    logger = setup_logging(config)
    set_global_seed(config["project"].get("seed", 42))
    device = get_device(config)

    logger.info("#" * 60)
    logger.info("  ELECTRICITY THEFT DETECTION SYSTEM")
    logger.info(f"  Mode   : {args.mode}")
    logger.info(f"  Device : {device}")
    logger.info("#" * 60)

    t0 = time.time()

    if args.mode == "baseline":
        run_baseline_mode(config, logger, args.output_dir)

    elif args.mode == "deep":
        run_deep_mode(args.model, config, logger, args.output_dir, device)

    elif args.mode == "tune":
        run_tune_mode(args.model, config, logger, args.output_dir, device)

    elif args.mode == "explain":
        run_explain_mode(config, logger, args.output_dir)

    elif args.mode == "all":
        logger.info("Running FULL pipeline: baseline + all deep models + ensemble + explain")
        run_baseline_mode(config, logger, args.output_dir)
        for mdl in ["cnn", "lstm", "gru", "bilstm", "cnn_lstm",
                    "cnn_gru", "transformer", "tcn"]:
            try:
                run_deep_mode(mdl, config, logger, args.output_dir, device)
            except Exception as e:
                logger.error(f"DL model {mdl} failed: {e}", exc_info=True)
        run_explain_mode(config, logger, args.output_dir)

    elif args.mode == "evaluate":
        from src.evaluation.evaluator import Evaluator
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = run_data_pipeline(
            config, logger, feature_engineering=True
        )
        evaluator = Evaluator(config=config["evaluation"])
        import joblib, glob
        results = {}
        for model_path in glob.glob(f"{args.output_dir}/models/*.pkl"):
            name  = Path(model_path).stem
            model = joblib.load(model_path)
            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.predict(X_test).astype(float)
                results[name] = evaluator.compute_metrics(y_test, y_prob)
            except Exception as e:
                logger.warning(f"Could not evaluate {name}: {e}")

        table = evaluator.generate_comparison_table(results)
        logger.info("\n" + table.to_string())
        evaluator.save_results(results, f"{args.output_dir}/reports/full_evaluation.json")

    elapsed = time.time() - t0
    logger.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    logger.info("Done.")


if __name__ == "__main__":
    main()
