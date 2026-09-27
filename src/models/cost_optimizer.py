"""
Cost-Sensitive Inspection Ranking and Operational Threshold Optimizer
Formulates field inspection dispatch as a financial decision problem under capacity constraints.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from config.settings import TariffAndCostConfig, TARIFF_CONFIG
from config.logging_config import setup_logger

logger = setup_logger("CostOptimizer")

class CostSensitiveInspectionOptimizer:
    def __init__(self, config: TariffAndCostConfig = TARIFF_CONFIG):
        self.config = config

    def calculate_expected_net_value(self, 
                                     calibrated_probs: np.ndarray, 
                                     estimated_loss_kwh_per_month: np.ndarray, 
                                     consumer_types: np.ndarray) -> np.ndarray:
        """
        Computes the expected net financial value of inspecting each candidate meter:
        E[V_i] = P_i * (Loss_kWh_i * Tariff * PenaltyMult) - C_inspect - (1 - P_i) * C_false
        """
        tariffs = np.array([
            self.config.industrial_tariff_per_kwh if c == "Industrial" else
            self.config.commercial_tariff_per_kwh if c == "Commercial" else
            self.config.tariff_per_kwh
            for c in consumer_types
        ])
        
        # Gross recovery if theft is verified (stolen revenue + penal recovery)
        gross_recovery = estimated_loss_kwh_per_month * tariffs * self.config.theft_penalty_multiplier
        
        # Expected value
        ev = (
            calibrated_probs * gross_recovery 
            - self.config.inspection_cost_c_inspect 
            - (1.0 - calibrated_probs) * self.config.false_accusation_cost_c_false
        )
        return ev

    def rank_and_dispatch(self, 
                          df_candidates: pd.DataFrame, 
                          calibrated_probs: np.ndarray, 
                          budget_limit: int = None) -> pd.DataFrame:
        """
        Ranks candidate meters by expected net return and applies the inspection budget constraint.
        """
        k = budget_limit if budget_limit is not None else self.config.weekly_inspection_capacity
        
        df = df_candidates.copy()
        df["calibrated_theft_prob"] = calibrated_probs
        
        # Estimate monthly unmetered energy loss (based on severe deficit or mean negative residual)
        if "mean_baseline_residual" in df.columns:
            # Negative residual indicates consumption below expectation
            daily_loss = np.maximum(0.0, -df["mean_baseline_residual"].values * 24.0)
            df["est_loss_kwh_month"] = daily_loss * 30.0
        else:
            df["est_loss_kwh_month"] = df["mean_consumption"].values * 24.0 * 30.0 * 0.40
            
        c_types = df["consumer_type"].values if "consumer_type" in df.columns else np.array(["Residential"] * len(df))
        df["expected_net_value_rs"] = self.calculate_expected_net_value(
            df["calibrated_theft_prob"].values,
            df["est_loss_kwh_month"].values,
            c_types
        )
        
        # Rank descending by Expected Net Value
        ranked = df.sort_values("expected_net_value_rs", ascending=False).reset_index(drop=True)
        ranked["rank"] = np.arange(1, len(ranked) + 1)
        
        # Dispatch decision: meters with positive expected net value within capacity limit
        ranked["dispatch_status"] = np.where(
            (ranked["rank"] <= k) & (ranked["expected_net_value_rs"] > 0),
            "DISPATCH_INSPECTION",
            "DEFER_MONITOR"
        )
        
        return ranked

    def simulate_cost_curve(self, 
                            calibrated_probs: np.ndarray, 
                            y_true: np.ndarray, 
                            estimated_loss_kwh: np.ndarray, 
                            consumer_types: np.ndarray) -> Dict[str, Any]:
        """
        Simulates total utility net financial yield as a function of inspection threshold or budget.
        Compares Cost-Sensitive Ranking vs Random Audit vs Uncalibrated Raw Score.
        """
        ev = self.calculate_expected_net_value(calibrated_probs, estimated_loss_kwh, consumer_types)
        
        tariffs = np.array([
            self.config.industrial_tariff_per_kwh if c == "Industrial" else
            self.config.commercial_tariff_per_kwh if c == "Commercial" else
            self.config.tariff_per_kwh
            for c in consumer_types
        ])
        
        realized_recovery_if_theft = estimated_loss_kwh * tariffs * self.config.theft_penalty_multiplier
        
        order = np.argsort(-ev)
        n = len(y_true)
        
        cumulative_inspections = np.arange(1, n + 1)
        cumulative_net_savings = []
        cumulative_theft_caught = []
        cumulative_false_alarms = []
        
        current_savings = 0.0
        current_theft = 0
        current_false = 0
        
        for idx in order:
            is_theft = y_true[idx]
            if is_theft == 1:
                # Realized positive recovery minus inspection cost
                net_gain = realized_recovery_if_theft[idx] - self.config.inspection_cost_c_inspect
                current_theft += 1
            else:
                # False accusation penalty minus inspection cost
                net_gain = - self.config.inspection_cost_c_inspect - self.config.false_accusation_cost_c_false
                current_false += 1
                
            current_savings += net_gain
            cumulative_net_savings.append(current_savings)
            cumulative_theft_caught.append(current_theft)
            cumulative_false_alarms.append(current_false)
            
        max_savings = max(cumulative_net_savings)
        optimal_budget_idx = int(np.argmax(cumulative_net_savings))
        
        return {
            "budget_k": cumulative_inspections.tolist(),
            "cumulative_net_savings": cumulative_net_savings,
            "cumulative_theft_caught": cumulative_theft_caught,
            "cumulative_false_alarms": cumulative_false_alarms,
            "max_net_savings_rs": float(max_savings),
            "optimal_inspection_count": optimal_budget_idx + 1,
            "optimal_theft_caught": cumulative_theft_caught[optimal_budget_idx],
            "optimal_false_alarms": cumulative_false_alarms[optimal_budget_idx]
        }
