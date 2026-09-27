"""
End-to-End Feature Engineering Pipeline linking telemetry to model-ready features
"""
import pandas as pd
from typing import Tuple, Dict, Any
from src.data.outage_handler import OutageHandler
from src.data.feeder_balancer import FeederBalancer
from src.features.temporal_baseline import PersonalizedTemporalBaseline
from src.features.statistical_features import StatisticalFeatureExtractor
from config.logging_config import setup_logger

logger = setup_logger("FeaturePipeline")

class FeaturePipeline:
    def __init__(self, baseline_days: int = 28):
        self.outage_handler = OutageHandler()
        self.feeder_balancer = FeederBalancer()
        self.baseline_model = PersonalizedTemporalBaseline(baseline_lookback_days=baseline_days)
        self.feature_extractor = StatisticalFeatureExtractor()

    def run(self, 
            readings_df: pd.DataFrame, 
            feeders_df: pd.DataFrame, 
            meters_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes end-to-end transformation:
        1. Flags and filters grid blackout events.
        2. Computes feeder mass-balance unmetered losses.
        3. Generates personalized temporal baselines & residuals.
        4. Extracts tsfresh-style shape and drop statistical features.
        
        Returns:
            features_df: tabular ML dataset (one row per meter)
            enriched_readings_df: interval telemetry with all residuals attached
        """
        logger.info("Step 1/4: Flagging grid blackouts vs localized meter zero readings...")
        df_outage = self.outage_handler.flag_grid_outages(readings_df)
        
        logger.info("Step 2/4: Computing feeder energy mass-balance residuals...")
        feeder_balance = self.feeder_balancer.calculate_feeder_energy_balance(df_outage, feeders_df)
        df_with_feeder = self.feeder_balancer.attach_feeder_imbalance_to_meters(df_outage, feeder_balance)
        
        # Merge consumer type metadata if present
        if "consumer_type" in meters_df.columns and "consumer_type" not in df_with_feeder.columns:
            df_with_feeder = pd.merge(
                df_with_feeder, 
                meters_df[["meter_id", "consumer_type"]], 
                on="meter_id", 
                how="left"
            )
            
        logger.info("Step 3/4: Estimating personalized temporal baselines and residual signals...")
        enriched_readings = self.baseline_model.compute_residuals(df_with_feeder)
        
        logger.info("Step 4/4: Extracting comprehensive tabular features per meter...")
        features_df = self.feature_extractor.transform_fleet(enriched_readings)
        
        logger.info("Feature pipeline completed successfully.")
        return features_df, enriched_readings
