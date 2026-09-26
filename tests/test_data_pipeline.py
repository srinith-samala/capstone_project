"""
Unit tests for data generation, outage handling, and feeder energy balance
"""
import pytest
import numpy as np
import pandas as pd
from src.data.generator import SmartMeterSimulator
from src.data.outage_handler import OutageHandler
from src.data.feeder_balancer import FeederBalancer
from config.settings import SimulationConfig

@pytest.fixture
def small_sim_config():
    return SimulationConfig(
        num_meters=10,
        num_days=7,
        frequency_minutes=60,
        num_feeders=2,
        theft_prevalence=0.30
    )

def test_simulator_generation(small_sim_config):
    sim = SmartMeterSimulator(config=small_sim_config, seed=42)
    readings_df, meters_df, feeders_df = sim.generate_fleet()
    
    assert len(meters_df) == 10
    assert len(readings_df) == 10 * 7 * 24
    assert len(feeders_df) == 2 * 7 * 24
    assert "reported_consumption_kwh" in readings_df.columns
    assert "is_tampered" in readings_df.columns
    assert readings_df["reported_consumption_kwh"].min() >= 0.0

def test_outage_handler():
    handler = OutageHandler(outage_threshold_ratio=0.50)
    timestamps = pd.date_range("2026-01-01", periods=10, freq="h")
    
    # 4 meters on FEEDER_A, 3 have zero consumption at t=0
    df = pd.DataFrame({
        "timestamp": list(timestamps[:4]) * 4,
        "meter_id": ["M1"]*4 + ["M2"]*4 + ["M3"]*4 + ["M4"]*4,
        "feeder_id": ["FEEDER_A"] * 16,
        "reported_consumption_kwh": [0.0, 1.0, 1.0, 1.0] + [0.0, 1.2, 1.1, 0.9] + [0.0, 0.8, 1.0, 1.1] + [1.5, 1.6, 1.4, 1.5]
    })
    
    flagged = handler.flag_grid_outages(df)
    assert "is_grid_outage" in flagged.columns
    # At t=0, 3 out of 4 (75%) meters are 0 -> should be flagged as outage
    t0_outages = flagged[flagged["timestamp"] == timestamps[0]]["is_grid_outage"].values
    assert all(t0_outages == 1)

def test_feeder_balancer():
    balancer = FeederBalancer(expected_technical_loss_rate=0.05)
    timestamps = pd.date_range("2026-01-01", periods=5, freq="h")
    
    feeders_df = pd.DataFrame({
        "timestamp": timestamps,
        "feeder_id": ["FEEDER_A"] * 5,
        "feeder_supply_kwh": [100.0] * 5,
        "feeder_reported_sum_kwh": [80.0] * 5
    })
    readings_df = pd.DataFrame()
    
    balance = balancer.calculate_feeder_energy_balance(readings_df, feeders_df)
    assert "unmetered_energy_loss_kwh" in balance.columns
    # 100 - 80 - 5 = 15 kWh unmetered loss
    assert np.isclose(balance["unmetered_energy_loss_kwh"].iloc[0], 15.0)
