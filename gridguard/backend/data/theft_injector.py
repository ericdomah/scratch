import os
import random
import yaml
import numpy as np
import torch

# Configure Logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = "c:/Users/User/Downloads/scratch-main/gridguard/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

class TheftInjector:
    """
    Advanced Theft Ingestion Engine for smart grids.
    Resolves Fix 3 (Sequence Window Contradiction - weekly aggregations) and
    Fix 8 (Per-Theft-Type breakdown validation with labeled anomaly profiles).
    Enforces a strict 0.5 kWh max ramp rate constraint on injected consumption.
    """
    
    def __init__(self):
        self.ramp_limit = config["data"]["ramp_rate_constraint"] # 0.5 kWh
        
    def enforce_ramp_rate(self, original_seq, injected_seq):
        """
        Enforces that the change between consecutive timesteps in the injected
        consumption sequence does not exceed the configured ramp rate limit (0.5).
        """
        seq = injected_seq.clone()
        for t in range(1, len(seq)):
            diff = seq[t] - seq[t-1]
            if abs(diff) > self.ramp_limit:
                # Clamp the step transition to the limit
                sign = 1.0 if diff > 0 else -1.0
                seq[t] = seq[t-1] + sign * self.ramp_limit
        return torch.clamp(seq, 0.0, 1.0)

    def inject_theft(self, sequence, theft_type=1):
        """
        Injects a specific theft signature into a normal sequence.
        Returns:
            Tuple[torch.Tensor, int]: Injected sequence tensor, and its theft type label.
        
        Labels:
            0 = Normal
            1 = Constant Reduction
            2 = Partial Phase Bypass
            3 = High-Resistance Shunt (gradual drift)
            4 = Load-Shifting Attack
            5 = Direct Hook (abrupt step)
        """
        # Ensure sequence is a PyTorch tensor of shape (seq_len)
        if isinstance(sequence, np.ndarray):
            window = torch.tensor(sequence, dtype=torch.float32)
        else:
            window = sequence.clone()
            
        seq_len = len(window)
        injected = window.clone()
        
        if theft_type == 1:
            # 1 = Constant Reduction (alpha factor)
            alpha = np.random.uniform(0.4, 0.7)
            injected = window * alpha
            
        elif theft_type == 2:
            # 2 = Partial Phase Bypass (drop during a specific interval window)
            # Table 3.1 Weekly aggregates duration maps 4-26 weekly intervals (Fix 3)
            start_idx = np.random.randint(4, 12)
            end_idx = np.random.randint(14, seq_len)
            alpha = np.random.uniform(0.1, 0.3)
            injected[start_idx:end_idx] = window[start_idx:end_idx] * alpha
            
        elif theft_type == 3:
            # 3 = High-Resistance Shunt (Subtle gradual downward drift)
            slope = np.random.uniform(-0.02, -0.01)
            drift = torch.linspace(1.0, 1.0 + (seq_len * slope), seq_len)
            injected = window * drift
            
        elif theft_type == 4:
            # 4 = Load-Shifting Attack (Shift consumption from peak weeks to low weeks)
            # Reverses consumption pattern or shifts peak values backwards
            injected = torch.flip(window, dims=[0]) * 0.9
            
        elif theft_type == 5:
            # 5 = Direct Hook (Abrupt step drop to near-zero)
            hook_start = np.random.randint(8, 16)
            injected[hook_start:] = np.random.uniform(0.02, 0.08)
            
        else:
            # 0 = Normal (no-op)
            return torch.clamp(window, 0.0, 1.0), 0
            
        # Apply strict physical ramp rate limiter
        clamped_injected = self.enforce_ramp_rate(window, injected)
        return clamped_injected, theft_type

if __name__ == "__main__":
    print("--- Theft Injector Functional Verification ---")
    injector = TheftInjector()
    
    # Generate mock uniform normal sequence
    normal_seq = torch.ones(26) * 0.8
    print(f"Normal Base Consumption: {normal_seq.numpy()}")
    
    for ttype in range(1, 6):
        injected, label = injector.inject_theft(normal_seq, theft_type=ttype)
        # Verify ramp rate constraint
        diffs = torch.abs(injected[1:] - injected[:-1])
        max_diff = torch.max(diffs).item()
        print(f"Type {ttype}: Max Ramp Rate={max_diff:.4f} | Label={label} | Status={'[OK]' if max_diff <= 0.5 else '[FAIL]'}")
