"""
Data and Concept Drift Monitoring Module (PSI and Kolmogorov-Smirnov Tests)
Monitors feature shifts and model score stability over seasonal and operational timeframes.
"""
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List
from config.logging_config import setup_logger

logger = setup_logger("DriftMonitor")

class DriftMonitor:
    @staticmethod
    def calculate_psi(baseline_arr: np.ndarray, 
                      current_arr: np.ndarray, 
                      num_buckets: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI) between baseline and current distributions:
        PSI < 0.10: Stable (No shift)
        0.10 <= PSI < 0.25: Moderate shift (Monitor)
        PSI >= 0.25: Significant shift (Trigger baseline recalibration / retraining)
        """
        b = baseline_arr[~np.isnan(baseline_arr)]
        c = current_arr[~np.isnan(current_arr)]
        
        if len(b) < 10 or len(c) < 10:
            return 0.0
            
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(b, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        # Remove duplicate bin edges if values are constant
        bins = np.unique(bins)
        if len(bins) < 3:
            return 0.0
            
        b_counts = np.histogram(b, bins=bins)[0]
        c_counts = np.histogram(c, bins=bins)[0]
        
        b_pct = np.maximum(b_counts / len(b), 1e-4)
        c_pct = np.maximum(c_counts / len(c), 1e-4)
        
        psi = np.sum((c_pct - b_pct) * np.log(c_pct / b_pct))
        return float(psi)

    def monitor_feature_drift(self, 
                              baseline_df: pd.DataFrame, 
                              current_df: pd.DataFrame, 
                              feature_names: List[str]) -> Dict[str, Any]:
        """
        Runs PSI and 2-sample Kolmogorov-Smirnov test across key telemetry features.
        """
        drift_report = {}
        high_drift_features = []
        
        for feat in feature_names:
            if feat not in baseline_df.columns or feat not in current_df.columns:
                continue
            b_vals = baseline_df[feat].dropna().values
            c_vals = current_df[feat].dropna().values
            
            psi_val = self.calculate_psi(b_vals, c_vals)
            ks_stat, ks_pval = stats.ks_2samp(b_vals, c_vals)
            
            if psi_val >= 0.25:
                status = "CRITICAL_DRIFT"
                high_drift_features.append(feat)
            elif psi_val >= 0.10:
                status = "MODERATE_SHIFT"
            else:
                status = "STABLE"
                
            drift_report[feat] = {
                "psi": psi_val,
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "status": status
            }
            
        overall_status = "ALERT_RETRAIN" if len(high_drift_features) >= 2 else "HEALTHY"
        logger.info(f"Drift monitoring completed. Overall status: {overall_status} (High drift in: {high_drift_features})")
        
        return {
            "overall_status": overall_status,
            "high_drift_features": high_drift_features,
            "features": drift_report
        }
