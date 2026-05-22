import os
import torch
import numpy as np
import yaml

# Add parent dir to path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gridguard')))

from backend.data.theft_injector import TheftInjector
from backend.data.dataset_loader import GridGuardDataset

def test_theft_injector_ramp_rate():
    """
    Asserts that the TheftInjector's physical ramp rate limiter strictly enforces
    that the absolute change between consecutive steps does not exceed 0.5 kWh
    across all 5 injected anomaly profiles.
    """
    injector = TheftInjector()
    
    # 26-week base normal consumption sequence
    np.random.seed(42)
    base_seq = torch.tensor(np.random.uniform(0.3, 0.9, 26), dtype=torch.float32)
    
    # Test all 5 active theft types (1 to 5)
    for ttype in range(1, 6):
        injected, label = injector.inject_theft(base_seq, theft_type=ttype)
        assert label == ttype
        
        # Calculate absolute transitions between consecutive weeks
        transitions = torch.abs(injected[1:] - injected[:-1])
        max_transition = torch.max(transitions).item()
        
        # Verify the 0.5 kWh physical grid ramp rate constraint
        assert max_transition <= 0.5, (
            f"Theft type {ttype} violated ramp rate constraint! "
            f"Observed max transition = {max_transition:.4f} (limit = 0.5)"
        )
        
    print("\n[SUCCESS] TheftInjector physical grid constraints verified (max ramp rate <= 0.5 kWh).")

def test_dataset_class_ratio():
    """
    Asserts that GridGuardDataset loads and balances the smart meter active cohort,
    preserving the 85:15 class ratio within the allowable ±2% drift boundary.
    """
    dataset = GridGuardDataset(inject_ratio=0.15)
    
    # Extract all final labels
    labels = np.array(dataset.final_labels)
    total_samples = len(labels)
    thefts = np.sum(labels > 0)
    theft_ratio = thefts / total_samples
    
    # Target is 15% theft (prevalence = 0.15)
    target_ratio = 0.15
    drift = abs(theft_ratio - target_ratio)
    
    # Assert drift is strictly within ±2%
    assert drift <= 0.02, (
        f"Smart grid dataset class ratio drifted too far! "
        f"Observed ratio = {theft_ratio:.4f} (target = {target_ratio:.2f}, drift limit = 0.02)"
    )
    
    print(f"[SUCCESS] Dataset class ratio is {theft_ratio:.2%} (drift = {drift:.2%} <= 2.0% limit).")
