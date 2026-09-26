"""
Unit tests for cost-sensitive optimization and dispatch ranking
"""
import pytest
import numpy as np
import pandas as pd
from src.models.cost_optimizer import CostSensitiveInspectionOptimizer
from config.settings import TariffAndCostConfig

def test_cost_optimizer_expected_value():
    config = TariffAndCostConfig(
        tariff_per_kwh=10.0,
        inspection_cost_c_inspect=1000.0,
        false_accusation_cost_c_false=2000.0,
        theft_penalty_multiplier=2.0
    )
    optimizer = CostSensitiveInspectionOptimizer(config=config)
    
    # Case 1: High probability theft on big consumer (e.g. 500 kWh lost, P=0.95)
    # Gross recovery = 500 * 10 * 2 = 10,000
    # Expected value = 0.95 * 10,000 - 1000 - 0.05 * 2000 = 9500 - 1000 - 100 = 8400 > 0
    ev_high = optimizer.calculate_expected_net_value(
        calibrated_probs=np.array([0.95]),
        estimated_loss_kwh_per_month=np.array([500.0]),
        consumer_types=np.array(["Residential"])
    )[0]
    assert ev_high > 8000.0
    
    # Case 2: Low probability honest consumer (P=0.02)
    # Expected value should be negative
    ev_low = optimizer.calculate_expected_net_value(
        calibrated_probs=np.array([0.02]),
        estimated_loss_kwh_per_month=np.array([50.0]),
        consumer_types=np.array(["Residential"])
    )[0]
    assert ev_low < 0.0

def test_rank_and_dispatch():
    optimizer = CostSensitiveInspectionOptimizer()
    df = pd.DataFrame({
        "meter_id": [f"MTR_{i}" for i in range(5)],
        "consumer_type": ["Residential"] * 5,
        "mean_consumption": [2.0] * 5,
        "mean_baseline_residual": [-1.5, -0.1, -2.0, -0.5, 0.2]
    })
    probs = np.array([0.80, 0.10, 0.95, 0.40, 0.05])
    
    ranked = optimizer.rank_and_dispatch(df, probs, budget_limit=2)
    assert len(ranked) == 5
    assert ranked["rank"].iloc[0] == 1
    # Top rank should be the meter with highest probability and highest stolen loss (MTR_2)
    assert ranked["meter_id"].iloc[0] == "MTR_2"
    # Only top 2 within budget should be marked for dispatch
    dispatched = ranked[ranked["dispatch_status"] == "DISPATCH_INSPECTION"]
    assert len(dispatched) <= 2
