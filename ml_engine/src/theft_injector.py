import numpy as np
import torch

class TheftInjector:
    """
    Simulates common electricity theft patterns for thesis benchmarking.
    Patterns derived from: 'Detecting electricity theft in smart grids' (academic standards).
    
    METHODOLOGICAL SAFEGUARD (Defensible & Realistic):
    Since the input 'window' is already normalized to [0, 1] based on the consumer's
    historical baseline (MinMax fit on normal behavior), any theft pattern that reduces
    consumption will naturally remain within [0, 1].
    
    CRITICAL BUG FIX: We DO NOT renormalize the window post-theft injection. 
    Dynamic min-max renormalization after theft injection cancels out any constant scale 
    reduction (e.g. multiplying by 0.5 and then normalizing rescales it back to the exact 
    same [0, 1] values, completely obliterating the theft signature and making constant 
    reduction 100% invisible to the ML model). We preserve physical drop magnitude.
    """
    
    @staticmethod
    def inject_constant_reduction(window, alpha=0.5):
        """Pattern 1: Multiply all readings by a constant factor (alpha < 1.0)"""
        injected = window * alpha
        return torch.clamp(injected, 0.0, 1.0)
 
    @staticmethod
    def inject_partial_bypass(window, start_idx, end_idx, alpha=0.1):
        """Pattern 2: Reduce readings only for a specific period"""
        injected = window.clone()
        injected[start_idx:end_idx] = injected[start_idx:end_idx] * alpha
        return torch.clamp(injected, 0.0, 1.0)
 
    @staticmethod
    def inject_on_off_bypass(window, probability=0.5, alpha=0.1):
        """Pattern 3: Randomly bypass the meter on certain days"""
        injected = window.clone()
        mask = torch.rand(window.shape[0]) < probability
        if window.dim() > 1:
            mask = mask.unsqueeze(-1).expand_as(window)
        injected[mask] = injected[mask] * alpha
        return torch.clamp(injected, 0.0, 1.0)
 
    @staticmethod
    def inject_constant_value(window, value=0.1):
        """Pattern 4: Set the reading to a constant low value"""
        injected = torch.full_like(window, value)
        return torch.clamp(injected, 0.0, 1.0)
 
    @staticmethod
    def inject_stealthy_drift(window, slope=-0.01):
        """Pattern 5: Subtle gradual drift in meter readings (hard to detect)"""
        seq_len = window.shape[0]
        drift = torch.linspace(1.0, 1.0 + (seq_len * slope), seq_len)
        if window.dim() > 1:
            drift = drift.unsqueeze(-1)
        injected = window * drift
        return torch.clamp(injected, 0.0, 1.0)
 
if __name__ == "__main__":
    # Test injection
    sample_window = torch.ones(20, 1)
    injector = TheftInjector()
    
    reduced = injector.inject_constant_reduction(sample_window, 0.3)
    print(f"Original first element: {sample_window[0].item():.2f}, Reduced: {reduced[0].item():.2f} (Should be 0.30)")
