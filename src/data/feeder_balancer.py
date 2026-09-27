"""
Neighborhood and Feeder Energy Mass-Balance Module
Computes transformer supply vs aggregated consumer consumption to quantify localized non-technical losses.
"""
import numpy as np
import pandas as pd
from typing import Dict
from config.logging_config import setup_logger

logger = setup_logger("FeederBalancer")

class FeederBalancer:
    def __init__(self, expected_technical_loss_rate: float = 0.05):
        """
        expected_technical_loss_rate: baseline physical dissipation loss ratio (e.g. 5% line resistance/transformer loss)
        """
        self.expected_technical_loss_rate = expected_technical_loss_rate

    def calculate_feeder_energy_balance(self, 
                                        readings_df: pd.DataFrame, 
                                        feeders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates hourly and rolling mass-balance residuals at the distribution feeder level.
        Returns feeder-level dataframe enriched with:
            - unmetered_energy_loss_kwh
            - loss_percentage
            - rolling_unmetered_loss_7d
        """
        merged = feeders_df.copy()
        
        # Expected technical losses based on transformer supply
        merged["expected_tech_loss_kwh"] = merged["feeder_supply_kwh"] * self.expected_technical_loss_rate
        
        # Energy imbalance (unmetered leakage)
        merged["unmetered_energy_loss_kwh"] = (
            merged["feeder_supply_kwh"] - merged["feeder_reported_sum_kwh"] - merged["expected_tech_loss_kwh"]
        )
        
        # Protect against division by zero
        merged["loss_percentage"] = np.clip(
            (merged["unmetered_energy_loss_kwh"] / np.maximum(merged["feeder_supply_kwh"], 1e-4)) * 100.0,
            -10.0, 100.0
        )
        
        # Compute 24-hour and 7-day rolling average imbalance per feeder
        merged = merged.sort_values(["feeder_id", "timestamp"]).reset_index(drop=True)
        merged["rolling_loss_24h"] = merged.groupby("feeder_id")["unmetered_energy_loss_kwh"].transform(
            lambda s: s.rolling(24, min_periods=1).mean()
        )
        merged["rolling_loss_7d"] = merged.groupby("feeder_id")["unmetered_energy_loss_kwh"].transform(
            lambda s: s.rolling(24 * 7, min_periods=1).mean()
        )
        
        logger.info("Computed feeder-level energy balance and unmetered loss differentials.")
        return merged

    def attach_feeder_imbalance_to_meters(self, 
                                          readings_df: pd.DataFrame, 
                                          feeder_balance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges the local feeder unmetered loss signal directly into individual meter intervals.
        Provides meters with contextual neighborhood leakage indicators.
        """
        cols_to_merge = [
            "timestamp", "feeder_id", "unmetered_energy_loss_kwh", 
            "loss_percentage", "rolling_loss_24h", "rolling_loss_7d"
        ]
        enriched_df = pd.merge(
            readings_df,
            feeder_balance_df[cols_to_merge],
            on=["timestamp", "feeder_id"],
            how="left"
        )
        return enriched_df
