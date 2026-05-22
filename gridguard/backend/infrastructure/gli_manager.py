import os
import time
import logging
import yaml
from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load config
CONFIG_PATH = "c:/Users/User/Downloads/scratch-main/gridguard/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

class GLIStatus(str, Enum):
    LIVE = "LIVE"
    STALE = "STALE"
    ESTIMATED = "ESTIMATED"
    ABSENT = "ABSENT"

class PredictionRequest(BaseModel):
    meter_id: str = Field(..., description="Unique smart meter identifier")
    kwh_sequence: List[float] = Field(..., min_items=26, max_items=26, description="Weekly aggregated kWh sequence")
    live_gli: Optional[float] = Field(None, description="Current Grid Load Index from master meter")
    live_gli_timestamp: Optional[float] = Field(None, description="Unix timestamp of the live GLI reading")
    hour_of_day: int = Field(12, ge=0, le=23, description="Target hour for inference (for temporal matching)")
    day_of_week: int = Field(0, ge=0, le=6, description="Target day for inference (0=Monday, 6=Sunday)")

class ForensicReport(BaseModel):
    meter_id: str
    gli_status: GLIStatus
    gli_value_used: float
    confidence_score: float
    is_theft_suspected: bool
    forensic_message: str
    requires_mandatory_human_review: bool

class GLIManager:
    """
    Manages the Context-Aware Grid Load Index (GLI) ingestion pipeline.
    Resolves Fix 7: Multi-tiered fallback logic for delayed, missing, or corrupted master meter data.
    """
    
    def __init__(self):
        self.cache_timeout_seconds = config["gli"]["cache_timeout_minutes"] * 60
        self.population_mean = config["gli"]["population_mean"]
        self.historical_days = config["gli"]["historical_days"]
        
        # In-memory cache for live GLI values
        self._cached_gli: Optional[float] = None
        self._cached_timestamp: Optional[float] = None
        
    def update_cache(self, value: float, timestamp: float):
        """Updates the cache with a fresh GLI reading."""
        self._cached_gli = value
        self._cached_timestamp = timestamp
        logger.info(f"GLI Cache updated: {value:.4f} at timestamp {timestamp}")

    def get_historical_gli_timescaledb(self, hour: int, day_of_week: int) -> float:
        """
        Queries historical GLI baseline from TimescaleDB hypertable.
        Implements the rolling historical estimate over the last 7 days for same hour and weekday.
        """
        # Under production: executes the following TimescaleDB raw SQL query:
        # SELECT AVG(gli_value) FROM substation_metrics 
        # WHERE hour_of_day = :hour AND day_of_week = :day_of_week 
        # AND timestamp >= NOW() - INTERVAL '7 days';
        
        # Standard mathematical mock representing historical rolling mean
        # Averages around the base target but incorporates stable cyclic patterns
        base_gli = 0.5 + 0.15 * (1.0 if (8 <= hour <= 20) else -1.0) # peak/off-peak cycle
        logger.debug(f"TimescaleDB Query: fetched historical baseline for hour {hour}, day {day_of_week} -> {base_gli:.4f}")
        return float(base_gli)

    def process_gli(self, request: PredictionRequest) -> Tuple[float, GLIStatus]:
        """
        Evaluates the 4 degradation modes for GLI ingestion.
        Returns:
            Tuple[float, GLIStatus]: The GLI value to inject, and its active status.
        """
        current_time = time.time()
        
        # MODE 1: GLI Available (Normal Operation)
        if request.live_gli is not None and request.live_gli_timestamp is not None:
            age = current_time - request.live_gli_timestamp
            if age >= 0 and age < self.cache_timeout_seconds:
                # Cache fresh reading
                self.update_cache(request.live_gli, request.live_gli_timestamp)
                return request.live_gli, GLIStatus.LIVE
                
        # MODE 2: GLI Delayed (Stale Data, < 30 minutes old)
        if self._cached_gli is not None and self._cached_timestamp is not None:
            cached_age = current_time - self._cached_timestamp
            if cached_age < self.cache_timeout_seconds:
                logger.warning(f"GLI Delayed (Stale data: {cached_age/60:.1f}m old). Operating under cache fallback.")
                return self._cached_gli, GLIStatus.STALE
                
        # MODE 3: GLI Unavailable (Missing or > 30 minutes old)
        # Attempt to retrieve historical estimate from TimescaleDB
        try:
            historical_val = self.get_historical_gli_timescaledb(request.hour_of_day, request.day_of_week)
            logger.warning(f"GLI Unavailable (>30m old or missing). Falling back to TimescaleDB rolling historical baseline.")
            return historical_val, GLIStatus.ESTIMATED
        except Exception as e:
            logger.error(f"TimescaleDB historical query failed: {e}. Degrading to Tier 4.")
            
        # MODE 4: GLI Completely Absent (No historical data or database offline)
        logger.error("GLI Completely Absent. Using population mean (0.5). Low spatial confidence flag raised.")
        return self.population_mean, GLIStatus.ABSENT

    def generate_forensic_report(self, request: PredictionRequest, predicted_proba: float) -> ForensicReport:
        """
        Runs model inference preprocessing, retrieves the appropriate GLI level,
        and generates a detailed, legally defensible forensic NLG brief.
        """
        gli_val, status = self.process_gli(request)
        is_theft = predicted_proba >= 0.50
        
        # Mandatory Human Review triggered by physical bypass suspicion or low spatial context (ABSENT mode)
        requires_review = is_theft or (status == GLIStatus.ABSENT)
        
        # Build NLG messages based on fallback tiers
        if status == GLIStatus.LIVE:
            brief = "GLI: live telemetry fully synchronized."
        elif status == GLIStatus.STALE:
            age_m = int((time.time() - (request.live_gli_timestamp or time.time())) / 60)
            brief = f"GLI: stale ({age_m}m old cached value used)."
        elif status == GLIStatus.ESTIMATED:
            brief = "GLI: estimated from historical baseline — context awareness degraded."
        else:
            brief = "LOW CONFIDENCE — no spatial context available. Mandatory human review required before field dispatch."
            
        final_message = (
            f"GRIDGUARD AUTOMATED FORENSIC REPORT | METER ID: {request.meter_id}\n"
            f"DIAGNOSTIC STATUS: {status.value} | VALUE USED: {gli_val:.4f}\n"
            f"THEFT PROBABILITY: {predicted_proba:.2%}\n"
            f"STATUS BRIEF: {brief}"
        )
        
        return ForensicReport(
            meter_id=request.meter_id,
            gli_status=status,
            gli_value_used=gli_val,
            confidence_score=predicted_proba,
            is_theft_suspected=is_theft,
            forensic_message=final_message,
            requires_mandatory_human_review=requires_review
        )

if __name__ == "__main__":
    # Self-test trace verifying all 4 degradation channels
    print("--- GLI Fallback Manager Verification Script ---")
    manager = GLIManager()
    
    # 1. LIVE Mode
    req1 = PredictionRequest(
        meter_id="MTR_TEST_01",
        kwh_sequence=[1.0] * 26,
        live_gli=0.74,
        live_gli_timestamp=time.time() - 10, # 10s old
        hour_of_day=14,
        day_of_week=2
    )
    val, status = manager.process_gli(req1)
    print(f"Test 1 (LIVE): Value={val}, Status={status} (Expected: LIVE)")
    
    # 2. STALE Mode
    # Request missing live data, should read from cache (10s old < 30 mins)
    req2 = PredictionRequest(
        meter_id="MTR_TEST_02",
        kwh_sequence=[1.0] * 26,
        live_gli=None,
        live_gli_timestamp=None,
        hour_of_day=14,
        day_of_week=2
    )
    val, status = manager.process_gli(req2)
    print(f"Test 2 (STALE): Value={val}, Status={status} (Expected: STALE)")
    
    # 3. ESTIMATED Mode (Clear cache to force historical)
    manager._cached_gli = None
    manager._cached_timestamp = None
    val, status = manager.process_gli(req2)
    print(f"Test 3 (ESTIMATED): Value={val:.4f}, Status={status} (Expected: ESTIMATED)")
    
    # 4. ABSENT Mode (Simulate TimescaleDB failure by disabling query method)
    def broken_db_query(h, d): raise RuntimeError("Database hypertable offline!")
    manager.get_historical_gli_timescaledb = broken_db_query
    val, status = manager.process_gli(req2)
    print(f"Test 4 (ABSENT): Value={val}, Status={status} (Expected: ABSENT)")
    
    # Generate report under ABSENT mode
    report = manager.generate_forensic_report(req2, 0.12)
    print(f"\nGenerated Report under Tier 4 Fallback:\n{report.forensic_message}")
    print(f"Mandatory Review Flag: {report.requires_mandatory_human_review}")
