"""
Comprehensive Model Evaluation Dossier for BDS-33
Computes PR-AUC, ROC-AUC, Recall@Budget, False Accusation Rate, and Detection Delay.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    roc_auc_score, 
    roc_curve, 
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score
)
from config.logging_config import setup_logger

logger = setup_logger("EvaluationMetrics")

class EvaluationMetrics:
    @staticmethod
    def calculate_all_metrics(y_true: np.ndarray, 
                              y_scores: np.ndarray, 
                              budget_fractions: List[float] = [0.05, 0.10, 0.15, 0.20],
                              threshold: float = 0.5) -> Dict[str, Any]:
        """
        Computes all mandatory portfolio metrics:
        1. PR-AUC (Precision-Recall Area Under Curve)
        2. ROC-AUC
        3. Recall @ Top-K Inspection Budget (5%, 10%, 15%, 20%)
        4. False Accusation Rate (FAR) = FP / (FP + TN)
        5. Precision, Recall, F1 at operating threshold
        6. Brier Score
        """
        # Ensure binary y_true
        y_true = np.asarray(y_true, dtype=int)
        y_scores = np.asarray(y_scores, dtype=float)
        
        # 1. PR-AUC
        precision_arr, recall_arr, pr_thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = float(auc(recall_arr, precision_arr))
        
        # 2. ROC-AUC
        try:
            roc_auc = float(roc_auc_score(y_true, y_scores))
            fpr_arr, tpr_arr, roc_thresholds = roc_curve(y_true, y_scores)
        except Exception:
            roc_auc = 0.5
            fpr_arr, tpr_arr = np.array([0, 1]), np.array([0, 1])
            
        # 3. Recall @ Inspection Budget (Ranking top K% highest risk meters)
        order = np.argsort(-y_scores)
        y_sorted = y_true[order]
        total_thefts = max(1, np.sum(y_true))
        n_total = len(y_true)
        
        recall_at_budget = {}
        for frac in budget_fractions:
            k = max(1, int(n_total * frac))
            thefts_caught = np.sum(y_sorted[:k])
            recall_val = float(thefts_caught / total_thefts)
            recall_at_budget[f"recall_at_{int(frac*100)}pct_budget"] = recall_val
            
        # 4. Standard threshold classification metrics
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        far = float(fp / max(1, fp + tn)) # False Accusation Rate
        prec = float(tp / max(1, tp + fp))
        rec = float(tp / max(1, tp + fn))
        f1 = float(2 * prec * rec / max(1e-6, prec + rec))
        brier = float(brier_score_loss(y_true, y_scores))
        
        return {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "false_accusation_rate": far,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "brier_score": brier,
            "recall_at_budget": recall_at_budget,
            "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
            "pr_curve": {"recall": recall_arr.tolist(), "precision": precision_arr.tolist()},
            "roc_curve": {"fpr": fpr_arr.tolist(), "tpr": tpr_arr.tolist()}
        }

    @staticmethod
    def calculate_detection_delay(readings_df: pd.DataFrame, 
                                  meter_metadata_df: pd.DataFrame, 
                                  alert_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Calculates Detection Delay: elapsed days from tamper onset until the meter is first flagged.
        """
        theft_meters = meter_metadata_df[meter_metadata_df["is_theft_meter"] == 1]
        delays_days = []
        
        for _, row in theft_meters.iterrows():
            m_id = row["meter_id"]
            raw_onset = row["tamper_start_timestamp"]
            if pd.isna(raw_onset):
                continue
            onset_time = pd.to_datetime(raw_onset)
                
            m_readings = readings_df[readings_df["meter_id"] == m_id].sort_values("timestamp")
            post_onset = m_readings[m_readings["timestamp"] >= onset_time]
            
            # An alert is triggered when rolling residual drops significantly below expectation
            if "rolling_ratio_mean_7d" in post_onset.columns:
                alerts = post_onset[post_onset["rolling_ratio_mean_7d"] < 0.60]
                if not alerts.empty:
                    first_alert_time = alerts["timestamp"].iloc[0]
                    delay_hours = (first_alert_time - onset_time).total_seconds() / 3600.0
                    delays_days.append(max(0.0, delay_hours / 24.0))
                else:
                    # Detection delayed to end of period
                    total_window = (m_readings["timestamp"].max() - onset_time).total_seconds() / 86400.0
                    delays_days.append(total_window)
                    
        if delays_days:
            mean_delay = float(np.mean(delays_days))
            median_delay = float(np.median(delays_days))
            min_delay = float(np.min(delays_days))
            max_delay = float(np.max(delays_days))
        else:
            mean_delay, median_delay, min_delay, max_delay = 0.0, 0.0, 0.0, 0.0
            
        return {
            "mean_detection_delay_days": mean_delay,
            "median_detection_delay_days": median_delay,
            "min_detection_delay_days": min_delay,
            "max_detection_delay_days": max_delay,
            "sample_size": len(delays_days)
        }
