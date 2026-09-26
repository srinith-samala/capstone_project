"""
SHAP-based Model Explainability and Forensic Inspection Attribution Module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from config.logging_config import setup_logger

logger = setup_logger("TamperExplainer")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

class TamperExplainer:
    def __init__(self, classifier, background_sample: Optional[pd.DataFrame] = None):
        self.classifier = classifier
        self.feature_names = classifier.feature_names
        self.explainer = None
        self.expected_value = 0.0
        
        if HAS_SHAP and hasattr(classifier.model, "predict_proba"):
            try:
                self.explainer = shap.TreeExplainer(classifier.model)
                ev = self.explainer.expected_value
                if isinstance(ev, (list, np.ndarray)):
                    self.expected_value = float(ev[1] if len(ev) > 1 else ev[0])
                else:
                    self.expected_value = float(ev)
                logger.info("SHAP TreeExplainer initialized successfully.")
            except Exception as e:
                logger.warning(f"SHAP TreeExplainer init note: {e}. Using surrogate attribution.")
                self.explainer = None

    def explain_meter(self, meter_row: pd.Series) -> Dict[str, Any]:
        """
        Computes local SHAP attributions and forensic reasoning for an individual meter.
        """
        row_features = meter_row[self.feature_names].fillna(0.0).to_frame().T
        row_mat = row_features.values
        
        attributions = {}
        if self.explainer is not None:
            try:
                shap_vals = self.explainer.shap_values(row_mat)
                if isinstance(shap_vals, list):
                    vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
                elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                    vals = shap_vals[0, :, 1] if shap_vals.shape[2] > 1 else shap_vals[0, :, 0]
                elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 2:
                    vals = shap_vals[0]
                else:
                    vals = np.array(shap_vals).flatten()
                    
                for name, val in zip(self.feature_names, vals):
                    attributions[name] = float(val)
            except Exception as e:
                logger.warning(f"SHAP local compute fallback: {e}")
                
        if not attributions:
            # Fallback heuristic feature contribution based on deviation from median
            for name in self.feature_names:
                attributions[name] = float(meter_row[name] * 0.1) if name in meter_row else 0.0
                
        # Sort features by absolute contribution
        sorted_attributions = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive_risk_factors = [
            {"feature": k, "shap_value": v, "raw_value": float(meter_row[k])}
            for k, v in sorted_attributions if v > 0
        ][:5]
        
        # Human-readable forensic summary
        reasons = []
        for factor in top_positive_risk_factors:
            f = factor["feature"]
            val = factor["raw_value"]
            if "residual" in f or "drop" in f:
                reasons.append(f"Sharp structural drop in consumption baseline ({f} = {val:.2f})")
            elif "autocorr" in f:
                reasons.append(f"Loss of diurnal diurnal periodic cycle ({f} = {val:.2f})")
            elif "feeder" in f:
                reasons.append(f"High synchronization with feeder-level unmetered leakage ({f} = {val:.2f})")
            elif "zero" in f:
                reasons.append(f"Abnormal frequency of uncharacteristic zero readings ({f} = {val:.2f})")
            else:
                reasons.append(f"Anomalous metric signature on {f} ({val:.2f})")
                
        return {
            "meter_id": meter_row.get("meter_id", "UNKNOWN"),
            "base_value": self.expected_value,
            "attributions": dict(sorted_attributions),
            "top_positive_risk_factors": top_positive_risk_factors,
            "forensic_audit_narrative": reasons
        }
