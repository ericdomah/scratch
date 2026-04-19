import numpy as np
import torch

class TheftInjector:
    """
    Simulates common electricity theft patterns for thesis benchmarking.
    Patterns derived from: 'Detecting electricity theft in smart grids' (academic standards).
    """
    
    @staticmethod
    def inject_constant_reduction(window, alpha=0.5):
        """Pattern 1: Multiply all readings by a constant factor (alpha < 1.0)"""
        return window * alpha

    @staticmethod
    def inject_partial_bypass(window, start_idx, end_idx, alpha=0.1):
        """Pattern 2: Reduce readings only for a specific period"""
        injected = window.clone()
        injected[start_idx:end_idx] = injected[start_idx:end_idx] * alpha
        return injected

    @staticmethod
    def inject_on_off_bypass(window, probability=0.5, alpha=0.1):
        """Pattern 3: Randomly bypass the meter on certain days"""
        injected = window.clone()
        mask = torch.rand(window.shape[0]) < probability
        injected[mask] = injected[mask] * alpha
        return injected

    @staticmethod
    def inject_constant_value(window, value=0.1):
        """Pattern 4: Set the reading to a constant low value"""
        return torch.full_like(window, value)

if __name__ == "__main__":
    # Test injection
    sample_window = torch.ones(20, 1) * 10
    injector = TheftInjector()
    
    reduced = injector.inject_constant_reduction(sample_window, 0.3)
    print(f"Original first element: {sample_window[0].item()}, Reduced: {reduced[0].item()}")
