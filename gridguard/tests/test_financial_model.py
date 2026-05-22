import os
import yaml

# Add parent dir to path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gridguard')))

from backend.finance.financial_impact_model import (
    FinancialImpactModel, 
    TRNC_TARIFF_TL_PER_KWH,
    SIMULATED_METERS,
    NTL_RATE_PROXY,
    AVG_DAILY_KWH_PER_METER,
    DAYS_PER_MONTH,
    MODEL_PRECISION,
    MODEL_RECALL,
    DETERRENCE_MULTIPLIER_LOW,
    DETERRENCE_MULTIPLIER_HIGH,
    DETERRENCE_SOURCE
)

def approx_equal(a, b, tol=1.0):
    return abs(a - b) < tol

def test_financial_projections_and_bounds():
    """
    Asserts that the financial impact calculations match the exact math equations,
    and that the deterrence bounds are strictly consistent with the 1.05 and 1.25 bounds
    of the cited academic source (Abbas et al., 2024).
    """
    # 1. Run financial analysis
    res = FinancialImpactModel.run_financial_analysis()
    
    # 2. Verify constants extraction
    constants = res["constants"]
    assert constants["trnc_tariff_tl_per_kwh"] == TRNC_TARIFF_TL_PER_KWH
    assert constants["simulated_meters"] == SIMULATED_METERS
    assert constants["ntl_rate_proxy"] == NTL_RATE_PROXY
    assert constants["deterrence_multiplier_low"] == 1.05
    assert constants["deterrence_multiplier_high"] == 1.25
    assert "Abbas et al." in constants["deterrence_source"]
    
    # 3. Direct verify monthly equations
    monthly_model = res["base_monthly_model"]
    expected_energy = SIMULATED_METERS * AVG_DAILY_KWH_PER_METER * DAYS_PER_MONTH
    expected_ntl_kwh = expected_energy * NTL_RATE_PROXY
    expected_loss_tl = expected_ntl_kwh * TRNC_TARIFF_TL_PER_KWH
    expected_recovery = expected_loss_tl * MODEL_PRECISION * MODEL_RECALL
    
    assert approx_equal(monthly_model["energy_dispatched_kwh"], expected_energy)
    assert approx_equal(monthly_model["ntl_loss_kwh"], expected_ntl_kwh)
    assert approx_equal(monthly_model["monthly_loss_tl"], expected_loss_tl)
    assert approx_equal(monthly_model["direct_recovery_tl"], expected_recovery)
    
    # 4. Verify deterrence bounds
    expected_low = expected_recovery * 1.05
    expected_high = expected_recovery * 1.25
    
    assert approx_equal(monthly_model["deterrence_recovery_range"]["low"], expected_low)
    assert approx_equal(monthly_model["deterrence_recovery_range"]["high"], expected_high)
    
    # 5. Verify Lefkoşa scaled regional figures
    regional = res["regional_scaled_model"]
    scaling_factor = regional["scaling_factor"]
    
    expected_scaled_direct = expected_recovery * scaling_factor
    expected_scaled_low = expected_scaled_direct * 1.05
    expected_scaled_high = expected_scaled_direct * 1.25
    
    assert approx_equal(regional["direct_recovery_tl"], expected_scaled_direct)
    assert approx_equal(regional["deterrence_recovery_range"]["low"], expected_scaled_low)
    assert approx_equal(regional["deterrence_recovery_range"]["high"], expected_scaled_high)
    
    # Verify the defense string contains the exact bounds
    defense_str = regional["honest_range_defense_string"]
    assert f"TL {expected_scaled_low:,.0f}" in defense_str
    assert f"TL {expected_scaled_high:,.0f}" in defense_str
    
    print("\n[SUCCESS] Financial impact range model and academic citation bounds verified.")
