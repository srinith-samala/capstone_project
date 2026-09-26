"""
Personalized Consumer Temporal Baseline Modeling
Decomposes individual consumer consumption into diurnal, weekly, and weather-adjusted expectations.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from config.logging_config import setup_logger

logger = setup_logger("PersonalizedTemporalBaseline")

class PersonalizedTemporalBaseline:
    def __init__(self, baseline_lookback_days: int = 28):
        """
        baseline_lookback_days: number of initial days used to learn the consumer's clean historical baseline
        """
        self.baseline_lookback_days = baseline_lookback_days
        self.profiles: Dict[str, pd.DataFrame] = {}
        self.temp_coefficients: Dict[str, float] = {}

    def fit_meter_profile(self, meter_df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        Learns personalized (hour, is_weekend) median consumption and temperature sensitivity.
        Uses the initial lookback window to prevent contaminated training on post-tamper data.
        """
        df = meter_df.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
        
        # Filter initial lookback period
        start_time = df["timestamp"].min()
        end_lookback = start_time + pd.Timedelta(days=self.baseline_lookback_days)
        train_window = df[df["timestamp"] <= end_lookback].dropna(subset=["reported_consumption_kwh"])
        
        if len(train_window) < 24 * 7:
            train_window = df.dropna(subset=["reported_consumption_kwh"])
            
        # Diurnal-weekly profile: median consumption per (is_weekend, hour)
        profile = train_window.groupby(["is_weekend", "hour"])["reported_consumption_kwh"].median().reset_index()
        profile.rename(columns={"reported_consumption_kwh": "baseline_diurnal_kwh"}, inplace=True)
        
        # Estimate weather sensitivity slope via simple covariance
        cdh = np.maximum(0, train_window["temperature_c"].values - 24.0)
        loads = train_window["reported_consumption_kwh"].values
        var_cdh = np.var(cdh)
        if var_cdh > 0.01:
            beta_weather = max(0.0, np.cov(loads, cdh)[0, 1] / var_cdh)
        else:
            beta_weather = 0.0
            
        return profile, beta_weather

    def compute_residuals(self, readings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes personalized baseline and residual time series for every meter in readings_df:
            - expected_baseline_kwh
            - baseline_residual_kwh = reported - expected
            - baseline_residual_ratio = reported / expected
            - rolling_residual_mean_7d
        """
        logger.info("Computing personalized temporal baselines across all meters...")
        df = readings_df.copy()
        df["hour"] = df["timestamp"].dt.hour
        df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
        df["cdh"] = np.maximum(0, df["temperature_c"] - 24.0)
        
        enriched_meters = []
        for meter_id, m_group in df.groupby("meter_id"):
            profile, beta = self.fit_meter_profile(m_group)
            self.profiles[meter_id] = profile
            self.temp_coefficients[meter_id] = beta
            
            merged = pd.merge(m_group, profile, on=["is_weekend", "hour"], how="left")
            merged["expected_baseline_kwh"] = np.maximum(
                0.01,
                merged["baseline_diurnal_kwh"] + beta * merged["cdh"]
            )
            
            # Compute residual metrics
            reported = merged["reported_consumption_kwh"].fillna(merged["expected_baseline_kwh"])
            merged["baseline_residual_kwh"] = reported - merged["expected_baseline_kwh"]
            merged["baseline_residual_ratio"] = reported / merged["expected_baseline_kwh"]
            
            # Rolling indicators (24h and 7 days)
            merged["rolling_residual_mean_24h"] = merged["baseline_residual_kwh"].rolling(24, min_periods=1).mean()
            merged["rolling_residual_mean_7d"] = merged["baseline_residual_kwh"].rolling(24 * 7, min_periods=1).mean()
            merged["rolling_ratio_mean_7d"] = merged["baseline_residual_ratio"].rolling(24 * 7, min_periods=1).mean()
            
            enriched_meters.append(merged)
            
        result_df = pd.concat(enriched_meters, ignore_index=True)
        logger.info("Personalized baseline residuals calculated successfully.")
        return result_df
