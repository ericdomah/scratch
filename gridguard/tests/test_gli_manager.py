import os
import time
import yaml

# Add parent dir to path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gridguard')))

from backend.infrastructure.gli_manager import GLIManager, GLIStatus, PredictionRequest

def test_gli_fallback_modes():
    """
    Asserts correct behavior for all 4 GLI fallback channels:
    LIVE, STALE, ESTIMATED, and ABSENT.
    """
    manager = GLIManager()
    
    # ----------------------------------------------------
    # Test Mode 1: LIVE
    # ----------------------------------------------------
    req_live = PredictionRequest(
        meter_id="MTR_TEST_LIVE",
        kwh_sequence=[1.0] * 26,
        live_gli=0.85,
        live_gli_timestamp=time.time() - 10, # 10s old (fresh)
        hour_of_day=14,
        day_of_week=2
    )
    val, status = manager.process_gli(req_live)
    assert status == GLIStatus.LIVE
    assert val == 0.85
    # Verify cache got updated
    assert manager._cached_gli == 0.85
    
    # ----------------------------------------------------
    # Test Mode 2: STALE
    # ----------------------------------------------------
    # Request missing live data, should read from cache (10s old < 30 min limit)
    req_stale = PredictionRequest(
        meter_id="MTR_TEST_STALE",
        kwh_sequence=[1.0] * 26,
        live_gli=None,
        live_gli_timestamp=None,
        hour_of_day=14,
        day_of_week=2
    )
    val, status = manager.process_gli(req_stale)
    assert status == GLIStatus.STALE
    assert val == 0.85
    
    # ----------------------------------------------------
    # Test Mode 3: ESTIMATED
    # ----------------------------------------------------
    # Clear cache to force historical fallback
    manager._cached_gli = None
    manager._cached_timestamp = None
    
    req_est = PredictionRequest(
        meter_id="MTR_TEST_EST",
        kwh_sequence=[1.0] * 26,
        live_gli=None,
        live_gli_timestamp=None,
        hour_of_day=14,
        day_of_week=2
    )
    val, status = manager.process_gli(req_est)
    assert status == GLIStatus.ESTIMATED
    # Expected historical mock around peak hour is 0.65
    assert np.isclose(val, 0.65)
    
    # ----------------------------------------------------
    # Test Mode 4: ABSENT
    # ----------------------------------------------------
    # Mock database failure to trigger final degradation tier
    def broken_query(h, d):
        raise RuntimeError("TimescaleDB unavailable")
    manager.get_historical_gli_timescaledb = broken_query
    
    val, status = manager.process_gli(req_est)
    assert status == GLIStatus.ABSENT
    assert val == 0.5 # Population mean configured in config.yaml
    
    # Verify ABSENT triggers mandatory operator review
    report = manager.generate_forensic_report(req_est, 0.15) # Low theft probability
    assert report.requires_mandatory_human_review is True
    assert "LOW CONFIDENCE" in report.forensic_message
    
    print("\n[SUCCESS] GLI Manager fallback verification tests completed successfully.")

import numpy as np
