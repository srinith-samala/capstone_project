"""
Unit tests for personalized baselines and statistical feature extraction
"""
import pytest
import numpy as np
import pandas as pd
from src.features.temporal_baseline import PersonalizedTemporalBaseline
from src.features.statistical_features import StatisticalFeatureExtractor

def test_temporal_baseline_residuals():
    baseline_model = PersonalizedTemporalBaseline(baseline_lookback_days=7)
    timestamps = pd.date_range("2026-01-01", periods=24 * 14, freq="h")
    
    # Synthetic meter with consistent diurnal profile + sudden 50% drop in week 2
    normal_load = 2.0 + np.sin(2 * np.pi * timestamps.hour.values / 24.0)
    reported = np.array(normal_load, copy=True)
    reported[24 * 7:] *= 0.5
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "meter_id": "MTR_TEST",
        "reported_consumption_kwh": reported,
        "temperature_c": [25.0] * len(timestamps)
    })
    
    residuals_df = baseline_model.compute_residuals(df)
    assert "baseline_residual_kwh" in residuals_df.columns
    assert "expected_baseline_kwh" in residuals_df.columns
    # Post-drop residuals should be significantly negative
    post_drop_mean = residuals_df.iloc[24 * 7:]["baseline_residual_kwh"].mean()
    assert post_drop_mean < -0.3

def test_statistical_features_extraction():
    extractor = StatisticalFeatureExtractor()
    timestamps = pd.date_range("2026-01-01", periods=24 * 10, freq="h")
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "meter_id": "MTR_0001",
        "feeder_id": "FEEDER_A",
        "consumer_type": "Residential",
        "reported_consumption_kwh": np.random.uniform(0.5, 3.5, len(timestamps)),
        "is_tampered": [0] * len(timestamps)
    })
    
    feats = extractor.extract_meter_features(df)
    assert feats["meter_id"] == "MTR_0001"
    assert "mean_consumption" in feats
    assert "peak_to_average_ratio" in feats
    assert "temporal_drop_ratio" in feats
    assert feats["load_factor"] <= 1.0
