import os
import json
import logging
import random
import yaml
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = "c:/Users/User/Downloads/scratch-main/gridguard/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Set seeds globally
def set_seed(seed=None):
    if seed is None:
        seed = config["system"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to: {seed}")
    return seed

seed = set_seed()

class MetricsEngine:
    """
    Unified performance evaluation engine for the GridGuard AI system.
    Resolves Fix 1: Single Source of Truth for classification metrics.
    """
    
    @staticmethod
    def evaluate_and_save(model_name, y_true, y_proba, threshold=0.5, seed_used=42):
        """
        Computes all classification metrics and saves them to a structured JSON file.
        """
        logger.info(f"Starting metrics evaluation for model: {model_name}")
        
        # Binary prediction generation based on threshold
        y_pred = (y_proba >= threshold).astype(int)
        y_true = y_true.astype(int)
        
        # Calculate base classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate curves
        try:
            auroc = roc_auc_score(y_true, y_proba)
        except Exception as e:
            logger.warning(f"Failed to calculate AUROC: {e}. Setting to 0.5")
            auroc = 0.5
            
        precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall_pts, precision_pts)
        
        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Per-class metrics
        precision_class0 = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        recall_class0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        f1_class0 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        results = {
            "model_name": model_name,
            "seed": seed_used,
            "threshold": threshold,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auroc": float(auroc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp)
            },
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "per_class": {
                "normal": {
                    "precision": float(precision_class0),
                    "recall": float(recall_class0),
                    "f1": float(f1_class0)
                },
                "theft": {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1)
                }
            }
        }
        
        # Save to JSON
        output_dir = config["data"]["evaluation_results_dir"]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}.json")
        
        with open(output_path, "w") as out_f:
            json.dump(results, out_f, indent=4)
            
        logger.info(f"Results successfully saved to {output_path}")
        
        # Generate LaTeX row
        MetricsEngine.print_latex_row(results)
        
        return results

    @staticmethod
    def print_latex_row(results):
        """
        Prints LaTeX-formatted table row dynamically.
        """
        model_name_formatted = results["model_name"].replace("_", " ")
        latex_str = (
            f"\\textbf{{{model_name_formatted}}} & "
            f"{results['accuracy']:.3f} & "
            f"{results['precision']:.3f} & "
            f"{results['recall']:.3f} & "
            f"\\mathbf{{{results['f1_score']:.3f}}} & "
            f"{results['auroc']:.3f} \\\\"
        )
        print(f"\n--- LaTeX Row for {results['model_name']} ---")
        print(latex_str)
        print("-------------------------------------------\n")

    @staticmethod
    def load_results(model_name):
        """
        Loads metrics results for a given model from its saved JSON.
        """
        output_dir = config["data"]["evaluation_results_dir"]
        path = os.path.join(output_dir, f"{model_name}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No results found for model: {model_name} at {path}")
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def compare_models(model_names_list):
        """
        Loads saved JSON files and prints a formatted terminal comparison table,
        highlighting differences relative to the top-performing model.
        """
        results_list = []
        for name in model_names_list:
            try:
                results_list.append(MetricsEngine.load_results(name))
            except Exception as e:
                logger.error(f"Failed to load metrics for {name}: {e}")
                
        if not results_list:
            logger.error("No models loaded for comparison.")
            return
            
        # Sort by F1-Score desc
        results_list.sort(key=lambda x: x["f1_score"], reverse=True)
        
        print("\n" + "="*80)
        print(f"{'GRIDGUARD AI SYSTEM COMPARATIVE ANALYSIS MATRIX':^80}")
        print("="*80)
        print(f"{'Model Name':<28} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8} | {'AUROC':<6}")
        print("-"*80)
        
        best_f1 = results_list[0]["f1_score"]
        
        for idx, res in enumerate(results_list):
            marker = "*" if idx == 0 else " "
            f1_diff = res["f1_score"] - best_f1
            diff_str = f" ({f1_diff:+.3f})" if idx > 0 else " (Best)"
            
            print(f"{marker} {res['model_name'][:25]:<25} | "
                  f"{res['accuracy']:.4f} | "
                  f"{res['precision']:.4f} | "
                  f"{res['recall']:.4f} | "
                  f"{res['f1_score']:.4f}{diff_str:<8} | "
                  f"{res['auroc']:.4f}")
        print("="*80 + "\n")
