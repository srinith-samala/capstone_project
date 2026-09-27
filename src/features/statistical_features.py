"""
Tsfresh-inspired Statistical and Temporal Shape Feature Extractor
Transforms high-frequency interval meter telemetry into tabular ML features.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from config.logging_config import setup_logger

logger = setup_logger("StatisticalFeatureExtractor")

class StatisticalFeatureExtractor:
    def __init__(self):
        pass

    @staticmethod
    def _safe_autocorr(series: np.ndarray, lag: int) -> float:
        if len(series) <= lag:
            return 0.0
        s1 = series[:-lag]
        s2 = series[lag:]
        std1 = np.std(s1)
        std2 = np.std(s2)
        if std1 < 1e-6 or std2 < 1e-6:
            return 0.0
        corr = np.corrcoef(s1, s2)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def extract_meter_features(self, meter_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extracts comprehensive time-series shape, baseline residual, and feeder correlation features for a single meter.
        """
        df = meter_df.sort_values("timestamp").reset_index(drop=True)
        readings = df["reported_consumption_kwh"].fillna(0.0).values
        n_obs = len(readings)
        
        if n_obs < 48:
            raise ValueError(f"Insufficient time steps ({n_obs}) for feature extraction.")
            
        meter_id = df["meter_id"].iloc[0]
        feeder_id = df["feeder_id"].iloc[0]
        is_tampered_label = int(df["is_tampered"].max())
        consumer_type = df["consumer_type"].iloc[0] if "consumer_type" in df.columns else "Residential"
        
        # 1. Distributional & Moment features
        mean_val = float(np.mean(readings))
        std_val = float(np.std(readings))
        median_val = float(np.median(readings))
        q25, q75 = np.percentile(readings, [25, 75])
        iqr = float(q75 - q25)
        skew_val = float(stats.skew(readings)) if std_val > 1e-5 else 0.0
        kurt_val = float(stats.kurtosis(readings)) if std_val > 1e-5 else 0.0
        
        # 2. Shape, Peaks & Load Factor
        max_val = float(np.max(readings))
        min_val = float(np.min(readings))
        par = max_val / (mean_val + 1e-4)
        load_factor = mean_val / (max_val + 1e-4)
        zero_fraction = float(np.mean(readings <= 0.01))
        negative_count = int(np.sum(readings < 0.0))
        
        # 3. Temporal degradation & Trend dynamics
        mid_point = n_obs // 2
        first_half_mean = float(np.mean(readings[:mid_point]))
        second_half_mean = float(np.mean(readings[mid_point:]))
        temporal_drop_ratio = (second_half_mean + 1e-4) / (first_half_mean + 1e-4)
        
        # Rolling drop detection (last 14 days vs first 28 days)
        baseline_period_hours = min(n_obs // 3, 28 * 24)
        recent_period_hours = min(n_obs // 4, 14 * 24)
        baseline_ref_mean = float(np.mean(readings[:baseline_period_hours]))
        recent_mean = float(np.mean(readings[-recent_period_hours:]))
        recent_drop_ratio = (recent_mean + 1e-4) / (baseline_ref_mean + 1e-4)
        
        # 4. Periodicity & Autocorrelations
        autocorr_lag24 = self._safe_autocorr(readings, 24)
        autocorr_lag168 = self._safe_autocorr(readings, 168)
        
        # 5. Baseline Residual Features
        if "baseline_residual_kwh" in df.columns:
            residuals = df["baseline_residual_kwh"].fillna(0.0).values
            mean_residual = float(np.mean(residuals))
            std_residual = float(np.std(residuals))
            min_residual = float(np.min(residuals))
            # Fraction of hours severely below baseline (< 50% expected)
            if "baseline_residual_ratio" in df.columns:
                ratios = df["baseline_residual_ratio"].fillna(1.0).values
                severe_deficit_rate = float(np.mean(ratios < 0.50))
                min_ratio_7d = float(np.min(df["rolling_ratio_mean_7d"].dropna().values)) if "rolling_ratio_mean_7d" in df.columns and len(df["rolling_ratio_mean_7d"].dropna()) > 0 else 1.0
            else:
                severe_deficit_rate = 0.0
                min_ratio_7d = 1.0
        else:
            mean_residual = 0.0
            std_residual = 0.0
            min_residual = 0.0
            severe_deficit_rate = 0.0
            min_ratio_7d = 1.0
            
        # 6. Neighborhood / Feeder Coupling
        if "unmetered_energy_loss_kwh" in df.columns:
            feeder_losses = df["unmetered_energy_loss_kwh"].fillna(0.0).values
            if np.std(feeder_losses) > 1e-4 and std_val > 1e-4:
                # In non-technical loss scenarios, meter deficit mirrors feeder unmetered loss
                corr_imbalance = float(np.corrcoef(-readings, feeder_losses)[0, 1])
                corr_imbalance = 0.0 if np.isnan(corr_imbalance) else corr_imbalance
            else:
                corr_imbalance = 0.0
            feeder_avg_loss = float(np.mean(feeder_losses))
        else:
            corr_imbalance = 0.0
            feeder_avg_loss = 0.0
            
        return {
            "meter_id": meter_id,
            "feeder_id": feeder_id,
            "consumer_type": consumer_type,
            "mean_consumption": mean_val,
            "std_consumption": std_val,
            "median_consumption": median_val,
            "iqr_consumption": iqr,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "max_consumption": max_val,
            "min_consumption": min_val,
            "peak_to_average_ratio": par,
            "load_factor": load_factor,
            "zero_reading_frequency": zero_fraction,
            "negative_reading_count": negative_count,
            "temporal_drop_ratio": temporal_drop_ratio,
            "recent_drop_ratio": recent_drop_ratio,
            "autocorr_lag24": autocorr_lag24,
            "autocorr_lag168": autocorr_lag168,
            "mean_baseline_residual": mean_residual,
            "std_baseline_residual": std_residual,
            "min_baseline_residual": min_residual,
            "severe_deficit_rate": severe_deficit_rate,
            "min_ratio_7d": min_ratio_7d,
            "corr_feeder_imbalance": corr_imbalance,
            "feeder_avg_loss": feeder_avg_loss,
            "is_tampered": is_tampered_label
        }

    def transform_fleet(self, fleet_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms entire fleet intervals into a unified tabular feature matrix.
        """
        logger.info(f"Extracting statistical and shape features for {fleet_df['meter_id'].nunique()} meters...")
        features = []
        for _, m_group in fleet_df.groupby("meter_id"):
            feat_dict = self.extract_meter_features(m_group)
            features.append(feat_dict)
            
        features_df = pd.DataFrame(features)
        logger.info(f"Engineered feature table generated with shape: {features_df.shape}")
        return features_df
