import pytest
import numpy as np
import warnings
from theft_detection_system.src.preprocessing.imbalance_handler import ImbalanceHandler

def create_dummy_data():
    """Create a highly imbalanced dataset."""
    X_train = np.random.rand(100, 10).astype(np.float32)
    y_train = np.array([0] * 90 + [1] * 10, dtype=np.int64)
    return X_train, y_train

@pytest.mark.parametrize("method", [
    "smote",
    "borderlinesmote",
    "adasyn",
    "smoteenn",
    "smotetomek"
])
def test_resampling_methods(method):
    X_train, y_train = create_dummy_data()
    
    # Check if the code runs without exceptions
    handler = ImbalanceHandler(method=method)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_resampled, y_resampled = handler.resample(X_train, y_train)
        
    assert X_resampled.shape[0] > 100, f"{method} did not oversample."
    assert len(np.unique(y_resampled)) == 2, "Should have 2 classes."
    
def test_sample_code():
    """Ensure the sample code from requirements runs."""
    X_train, y_train = create_dummy_data()
    
    handler = ImbalanceHandler(method="smoteenn")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_resampled, y_resampled = handler.resample(X_train, y_train)
        
    assert X_resampled is not None
    assert y_resampled is not None
