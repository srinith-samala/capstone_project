"""
Probability Calibration Engine (Platt Scaling & Isotonic Regression)
Ensures raw classifier outputs represent true statistical probabilities of theft.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from config.logging_config import setup_logger

logger = setup_logger("ProbabilityCalibrator")

class ProbabilityCalibrator:
    def __init__(self, method: str = "isotonic"):
        """
        method: 'isotonic' (non-parametric, best for medium/large data) or 'sigmoid' (Platt scaling)
        """
        self.method = method
        self.calibrated_classifier: CalibratedClassifierCV = None

    def fit(self, base_estimator, X_val: pd.DataFrame, y_val: np.ndarray):
        """
        Fits calibration model on hold-out validation set.
        """
        logger.info(f"Calibrating probabilities using method='{self.method}'...")
        self.calibrated_classifier = CalibratedClassifierCV(
            estimator=base_estimator.model,
            method=self.method,
            cv="prefit"
        )
        X_mat = X_val[base_estimator.feature_names].fillna(0.0).values
        self.calibrated_classifier.fit(X_mat, y_val)
        self.feature_names = base_estimator.feature_names
        logger.info("Probability calibration complete.")
        return self

    def predict_calibrated_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns calibrated probability of theft P(Theft = 1 | X).
        """
        X_mat = X[self.feature_names].fillna(0.0).values
        probs = self.calibrated_classifier.predict_proba(X_mat)
        return probs[:, 1] if probs.shape[1] > 1 else np.zeros(len(X_mat))

    def evaluate_calibration(self, 
                             uncalibrated_probs: np.ndarray, 
                             calibrated_probs: np.ndarray, 
                             y_true: np.ndarray, 
                             n_bins: int = 8) -> Dict[str, Any]:
        """
        Computes calibration curves, Brier scores, and Expected Calibration Error (ECE).
        """
        # Brier score (lower is better)
        brier_uncal = brier_score_loss(y_true, uncalibrated_probs)
        brier_cal = brier_score_loss(y_true, calibrated_probs)
        
        # Reliability curves
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_true, uncalibrated_probs, n_bins=n_bins, strategy="uniform")
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, calibrated_probs, n_bins=n_bins, strategy="uniform")
        
        # ECE (Expected Calibration Error)
        ece_uncal = float(np.mean(np.abs(prob_true_uncal - prob_pred_uncal)))
        ece_cal = float(np.mean(np.abs(prob_true_cal - prob_pred_cal)))
        
        return {
            "brier_uncalibrated": float(brier_uncal),
            "brier_calibrated": float(brier_cal),
            "brier_improvement_pct": float((brier_uncal - brier_cal) / (brier_uncal + 1e-6) * 100.0),
            "ece_uncalibrated": ece_uncal,
            "ece_calibrated": ece_cal,
            "curve_uncalibrated": {"prob_pred": prob_pred_uncal.tolist(), "prob_true": prob_true_uncal.tolist()},
            "curve_calibrated": {"prob_pred": prob_pred_cal.tolist(), "prob_true": prob_true_cal.tolist()}
        }
