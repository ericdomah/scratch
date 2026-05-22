import os
import sys

# Configure pathing
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'tests')))

from tests.test_metrics_engine import test_metrics_consistency_and_persistence
from tests.test_gli_manager import test_gli_fallback_modes
from tests.test_theft_injector import test_theft_injector_ramp_rate, test_dataset_class_ratio
from tests.test_financial_model import test_financial_projections_and_bounds

def main():
    print("="*60)
    print(f"{'GRIDGUARD AI INTEGRATED SYSTEM TEST RUNNER':^60}")
    print("="*60)
    
    suite_status = True
    
    # Test 1: Metrics Engine
    try:
        print(">> Running Test Suite 1/4: Metrics Engine Consistency...")
        test_metrics_consistency_and_persistence(None)
        print("[PASS] Metrics Engine persistent calculations match verification benchmarks.")
    except Exception as e:
        print(f"[FAIL] Metrics Engine verification failed: {e}")
        suite_status = False
        
    print("-" * 60)
    
    # Test 2: GLI Fallback Manager
    try:
        print(">> Running Test Suite 2/4: GLI Fallback Degradation...")
        test_gli_fallback_modes()
        print("[PASS] GLI fallback tiers LIVE, STALE, ESTIMATED, and ABSENT operate perfectly.")
    except Exception as e:
        print(f"[FAIL] GLI fallback validation failed: {e}")
        suite_status = False
        
    print("-" * 60)
    
    # Test 3: Theft Injector & Dataset loader
    try:
        print(">> Running Test Suite 3/4: Theft Ingestion & Dataset balancing...")
        test_theft_injector_ramp_rate()
        test_dataset_class_ratio()
        print("[PASS] Physical ramp rate limit (0.5 kWh) and 85:15 class ratios preserved.")
    except Exception as e:
        print(f"[FAIL] Theft ingestion validation failed: {e}")
        suite_status = False
        
    print("-" * 60)
    
    # Test 4: Financial projections
    try:
        print(">> Running Test Suite 4/4: Financial range bounds...")
        test_financial_projections_and_bounds()
        print("[PASS] Financial range bounds and Abbas et al. (2024) deterrence multipliers verified.")
    except Exception as e:
        print(f"[FAIL] Financial model verification failed: {e}")
        suite_status = False
        
    print("="*60)
    if suite_status:
        print(f"{'ALL SYSTEM CORES FULLY COMPLIANT [100% PASS]':^60}")
    else:
        print(f"{'SYSTEM AUDIT ENCOUNTERED ERRORS [COMPLIANCE BLOCKED]':^60}")
    print("="*60)
    
    if not suite_status:
        sys.exit(1)

if __name__ == "__main__":
    main()
