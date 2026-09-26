"""
Outage Detection and Disambiguation Module for Smart-Meter Telemetry
"""
import numpy as np
import pandas as pd
from typing import Tuple
from config.logging_config import setup_logger

logger = setup_logger("OutageHandler")

class OutageHandler:
    def __init__(self, outage_threshold_ratio: float = 0.55, min_consecutive_hours: int = 1):
        """
        outage_threshold_ratio: Fraction of meters on the same feeder with ~0 consumption to qualify as feeder-wide blackout
        """
        self.outage_threshold_ratio = outage_threshold_ratio
        self.min_consecutive_hours = min_consecutive_hours

    def flag_grid_outages(self, readings_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects synchronized zero-reading events across feeders and flags 'is_grid_outage'.
        Distinguishes common-mode grid blackouts from localized single-meter tampering.
        """
        df = readings_df.copy()
        
        # Zero or near-zero threshold (e.g. < 0.01 kWh)
        is_zero = (df["reported_consumption_kwh"] <= 0.01).astype(int)
        df["is_zero_reading"] = is_zero
        
        # Calculate fraction of zero readings per (feeder_id, timestamp)
        feeder_zero_fraction = df.groupby(["feeder_id", "timestamp"])["is_zero_reading"].transform("mean")
        df["feeder_zero_fraction"] = feeder_zero_fraction
        
        # A timestamp is marked as a grid outage if more than threshold ratio of meters are down simultaneously
        df["is_grid_outage"] = (df["feeder_zero_fraction"] >= self.outage_threshold_ratio).astype(int)
        
        num_outage_events = df[df["is_grid_outage"] == 1]["timestamp"].nunique()
        logger.info(f"Disambiguated grid outages: identified {num_outage_events} distinct blackout intervals across feeders.")
        
        return df

    def filter_or_impute_outages(self, df_flagged: pd.DataFrame, method: str = "mask") -> pd.DataFrame:
        """
        Cleans outage hours from individual meter profiles.
        method: 'mask' sets values during outages to NaN/masked so baseline algorithms ignore them.
        """
        clean_df = df_flagged.copy()
        if method == "mask":
            # For feature calculation, mask out grid blackout hours so they don't corrupt individual consumption averages
            clean_df.loc[clean_df["is_grid_outage"] == 1, "reported_consumption_kwh"] = np.nan
        return clean_df
