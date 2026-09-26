"""
System Configuration & Operational Parameters for BDS-33:
Energy Theft and Meter Tamper Detection with Cost-Sensitive Temporal Anomalies
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "models_saved"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

@dataclass
class TariffAndCostConfig:
    """
    Financial and operational cost parameters for utility loss recovery.
    Default currency is INR (₹), but can represent any monetary unit.
    """
    tariff_per_kwh: float = 7.50          # Cost charged per kWh to consumers (₹/kWh)
    commercial_tariff_per_kwh: float = 11.0 # Commercial rate
    industrial_tariff_per_kwh: float = 14.5 # Industrial rate
    
    # Cost-sensitive parameters
    inspection_cost_c_inspect: float = 1200.0  # Fixed operational cost per physical field inspection (₹)
    false_accusation_cost_c_false: float = 2500.0 # Goodwill / legal / re-audit cost per false accusation (₹)
    theft_penalty_multiplier: float = 2.0      # Additional penal fine recovered on confirmed theft (multiplied by stolen energy)
    
    # Inspection operational constraints
    weekly_inspection_capacity: int = 15       # Max inspections the field squad can perform per week
    monthly_budget: float = 75000.0            # Max monthly inspection OPEX (₹)

@dataclass
class SimulationConfig:
    """
    Simulation parameters for physics-informed synthetic smart meter data generation.
    """
    num_meters: int = 120                      # Number of smart meters across feeders
    num_days: int = 120                        # Number of historical days
    frequency_minutes: int = 60                # Interval duration (60-min hourly data)
    num_feeders: int = 4                       # Number of distribution transformers/feeders
    
    # Tamper parameters
    theft_prevalence: float = 0.18             # ~18% meters exhibit non-technical loss tampering
    min_tamper_duration_days: int = 14         # Minimum duration a tamper attack persists
    
    # Attack injection types
    attack_types: List[str] = field(default_factory=lambda: [
        "scaling_bypass",                      # y_t = alpha * x_t (30-70% drop)
        "flatline_floor",                      # reports static minimal baseline
        "peak_selective_bypass",               # bypasses only during peak consumption hours
        "meter_reverse_flow",                  # reverse net-metering anomaly / negative flow
        "gradual_drift_attenuation"            # slow decline over weeks to evade delta thresholds
    ])

@dataclass
class ModelConfig:
    """
    Model training, calibration, and thresholding configurations.
    """
    random_state: int = 42
    test_size: float = 0.25
    cv_folds: int = 5
    
    # Probability calibration
    calibration_method: str = "isotonic"       # 'isotonic' or 'sigmoid'
    
    # Anomaly scoring
    isolation_forest_contamination: float = 0.15
    autoencoder_hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16, 32, 64])
    autoencoder_epochs: int = 40
    autoencoder_lr: float = 0.003

# Global singleton instances
TARIFF_CONFIG = TariffAndCostConfig()
SIMULATION_CONFIG = SimulationConfig()
MODEL_CONFIG = ModelConfig()
