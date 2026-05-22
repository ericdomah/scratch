import os
import json
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Add parent dir to path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gridguard')))

from backend.evaluation.metrics_engine import MetricsEngine

def test_metrics_consistency_and_persistence(tmp_path):
    """
    Asserts that MetricsEngine computes accurate metrics and that loading the saved JSON
    yields identical values to a fresh sklearn verification compute pass.
    """
    # 1. Generate realistic test predictions and ground truth
    np.random.seed(42)
    y_true = np.random.choice([0, 1], size=1000, p=[0.85, 0.15])
    
    # Add noise to probabilities so we get a good ROC curve
    y_proba = np.clip(y_true * 0.8 + np.random.normal(0.0, 0.25, 1000), 0.0, 1.0)
    
    model_name = "TestGuard_Model"
    threshold = 0.5
    
    # 2. Run metrics engine evaluation
    res = MetricsEngine.evaluate_and_save(model_name, y_true, y_proba, threshold=threshold, seed_used=42)
    
    # 3. Reload persisted JSON
    loaded_res = MetricsEngine.load_results(model_name)
    
    # 4. Compute direct verification calculations with scikit-learn
    y_pred = (y_proba >= threshold).astype(int)
    val_acc = accuracy_score(y_true, y_pred)
    val_prec = precision_score(y_true, y_pred, zero_division=0)
    val_rec = recall_score(y_true, y_pred, zero_division=0)
    val_f1 = f1_score(y_true, y_pred, zero_division=0)
    val_auroc = roc_auc_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # 5. Assert perfect equality between metrics engine return and reloaded JSON
    assert res["accuracy"] == loaded_res["accuracy"]
    assert res["precision"] == loaded_res["precision"]
    assert res["recall"] == loaded_res["recall"]
    assert res["f1_score"] == loaded_res["f1_score"]
    assert res["auroc"] == loaded_res["auroc"]
    
    # 6. Assert parity against sklearn reference calculations
    assert np.isclose(loaded_res["accuracy"], val_acc)
    assert np.isclose(loaded_res["precision"], val_prec)
    assert np.isclose(loaded_res["recall"], val_rec)
    assert np.isclose(loaded_res["f1_score"], val_f1)
    assert np.isclose(loaded_res["auroc"], val_auroc)
    
    # 7. Assert confusion matrix integer equality
    assert loaded_res["confusion_matrix"]["tn"] == int(tn)
    assert loaded_res["confusion_matrix"]["fp"] == int(fp)
    assert loaded_res["confusion_matrix"]["fn"] == int(fn)
    assert loaded_res["confusion_matrix"]["tp"] == int(tp)
    
    logger_msg = "MetricsEngine unit tests verified successfully with absolute scikit-learn parity."
    print(f"\n[SUCCESS] {logger_msg}")
