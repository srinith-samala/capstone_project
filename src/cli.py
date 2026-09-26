"""
Command-Line Interface and Pipeline Coordinator for BDS-33 Capstone Project
"""
import argparse
import sys
import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split

from config.settings import (
    BASE_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SYNTHETIC_DATA_DIR, MODELS_DIR,
    TARIFF_CONFIG, SIMULATION_CONFIG, MODEL_CONFIG
)
from config.logging_config import setup_logger
from src.data.generator import SmartMeterSimulator
from src.features.feature_pipeline import FeaturePipeline
from src.models.baselines import IsolationForestBaseline, AutoencoderBaseline
from src.models.classifier import TamperClassifier
from src.models.calibrator import ProbabilityCalibrator
from src.models.cost_optimizer import CostSensitiveInspectionOptimizer
from src.models.explainability import TamperExplainer
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.drift_monitor import DriftMonitor

logger = setup_logger("BDS33_CLI")

def run_simulation() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates and saves synthetic fleet telemetry, metadata, and feeder reads."""
    sim = SmartMeterSimulator(config=SIMULATION_CONFIG, tariff_config=TARIFF_CONFIG)
    readings_df, meters_df, feeders_df = sim.generate_fleet()
    
    readings_df.to_parquet(SYNTHETIC_DATA_DIR / "readings.parquet", index=False)
    meters_df.to_csv(SYNTHETIC_DATA_DIR / "meters.csv", index=False)
    feeders_df.to_parquet(SYNTHETIC_DATA_DIR / "feeders.parquet", index=False)
    
    logger.info(f"Saved synthetic fleet datasets to {SYNTHETIC_DATA_DIR}")
    return readings_df, meters_df, feeders_df

def load_or_generate_data():
    readings_file = SYNTHETIC_DATA_DIR / "readings.parquet"
    meters_file = SYNTHETIC_DATA_DIR / "meters.csv"
    feeders_file = SYNTHETIC_DATA_DIR / "feeders.parquet"
    
    if readings_file.exists() and meters_file.exists() and feeders_file.exists():
        logger.info("Loading existing synthetic data from disk...")
        readings_df = pd.read_parquet(readings_file)
        meters_df = pd.read_csv(meters_file)
        feeders_df = pd.read_parquet(feeders_file)
    else:
        readings_df, meters_df, feeders_df = run_simulation()
    return readings_df, meters_df, feeders_df

def run_training_pipeline():
    """
    Executes end-to-end feature extraction, baseline modeling, supervised training,
    probability calibration, cost optimization, SHAP explainability, and evaluation.
    """
    logger.info("=== BDS-33 Capstone End-to-End Pipeline Started ===")
    readings_df, meters_df, feeders_df = load_or_generate_data()
    
    # 1. Feature Engineering
    pipeline = FeaturePipeline(baseline_days=28)
    features_df, enriched_readings = pipeline.run(readings_df, feeders_df, meters_df)
    
    # Save enriched data for dashboard
    features_df.to_parquet(PROCESSED_DATA_DIR / "features.parquet", index=False)
    enriched_readings.to_parquet(PROCESSED_DATA_DIR / "enriched_readings.parquet", index=False)
    
    # 2. Train / Val / Test Split
    feature_cols = [
        c for c in features_df.columns 
        if c not in ["meter_id", "feeder_id", "consumer_type", "is_tampered"]
    ]
    X = features_df[feature_cols]
    y = features_df["is_tampered"].values
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=MODEL_CONFIG.random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=MODEL_CONFIG.random_state, stratify=y_temp
    )
    logger.info(f"Splits -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 3. Unsupervised Baseline 1: Isolation Forest
    iso_model = IsolationForestBaseline(random_state=MODEL_CONFIG.random_state)
    iso_model.fit(X_train)
    iso_test_scores = iso_model.predict_score(X_test)
    iso_metrics = EvaluationMetrics.calculate_all_metrics(y_test, iso_test_scores)
    
    # 4. Unsupervised Baseline 2: PyTorch Autoencoder
    ae_model = AutoencoderBaseline(epochs=25, random_state=MODEL_CONFIG.random_state)
    ae_model.fit(X_train)
    ae_test_scores = ae_model.predict_score(X_test)
    ae_metrics = EvaluationMetrics.calculate_all_metrics(y_test, ae_test_scores)
    
    # 5. Supervised Model: Calibrated Classifier
    clf = TamperClassifier(random_state=MODEL_CONFIG.random_state, use_lightgbm=True)
    clf.fit(X_train, y_train)
    
    uncalibrated_test_probs = clf.predict_proba(X_test)[:, 1]
    
    # Calibrate on validation split
    calibrator = ProbabilityCalibrator(method=MODEL_CONFIG.calibration_method)
    calibrator.fit(clf, X_val, y_val)
    calibrated_test_probs = calibrator.predict_calibrated_proba(X_test)
    
    cal_eval = calibrator.evaluate_calibration(uncalibrated_test_probs, calibrated_test_probs, y_test)
    clf_metrics = EvaluationMetrics.calculate_all_metrics(y_test, calibrated_test_probs)
    
    # 6. Detection Delay Calculation
    delay_metrics = EvaluationMetrics.calculate_detection_delay(enriched_readings, meters_df)
    
    # 7. Cost-Sensitive Inspection Optimizer
    test_idx = X_test.index
    test_meters_info = features_df.loc[test_idx].copy()
    test_meters_info["y_true"] = y_test
    if "mean_baseline_residual" in test_meters_info.columns:
        test_meters_info["est_loss_kwh_month"] = np.maximum(0.0, -test_meters_info["mean_baseline_residual"].values * 24.0) * 30.0
    else:
        test_meters_info["est_loss_kwh_month"] = test_meters_info["mean_consumption"].values * 24.0 * 30.0 * 0.40
    
    cost_opt = CostSensitiveInspectionOptimizer(config=TARIFF_CONFIG)
    ranked_dispatch = cost_opt.rank_and_dispatch(
        test_meters_info, 
        calibrated_test_probs, 
        budget_limit=TARIFF_CONFIG.weekly_inspection_capacity
    )
    
    cost_curve_results = cost_opt.simulate_cost_curve(
        calibrated_test_probs, 
        y_test, 
        test_meters_info["est_loss_kwh_month"].values,
        test_meters_info["consumer_type"].values
    )
    
    # 8. Explainability Engine
    explainer = TamperExplainer(clf)
    sample_explanations = []
    for idx, row in ranked_dispatch.head(5).iterrows():
        sample_explanations.append(explainer.explain_meter(row))
        
    # 9. Drift Monitoring Setup
    drift_mon = DriftMonitor()
    drift_res = drift_mon.monitor_feature_drift(X_train, X_test, feature_cols[:8])
    
    # Save Model Artifacts
    with open(MODELS_DIR / "iso_model.pkl", "wb") as f:
        pickle.dump(iso_model, f)
    with open(MODELS_DIR / "ae_model.pkl", "wb") as f:
        pickle.dump(ae_model, f)
    with open(MODELS_DIR / "clf_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open(MODELS_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(calibrator, f)
        
    summary_results = {
        "baseline_isolation_forest": iso_metrics,
        "baseline_autoencoder": ae_metrics,
        "calibrated_classifier": clf_metrics,
        "calibration_metrics": cal_eval,
        "cost_curve": cost_curve_results,
        "detection_delay": delay_metrics,
        "drift_status": drift_res["overall_status"]
    }
    
    with open(MODELS_DIR / "evaluation_summary.json", "w") as f:
        json.dump(summary_results, f, indent=2)
        
    ranked_dispatch.to_parquet(PROCESSED_DATA_DIR / "ranked_dispatch.parquet", index=False)
    
    logger.info("=== Pipeline Run Finished Successfully ===")
    logger.info(f"Isolation Forest PR-AUC: {iso_metrics['pr_auc']:.4f} | Recall@10%: {iso_metrics['recall_at_budget']['recall_at_10pct_budget']:.2f}")
    logger.info(f"Autoencoder PR-AUC:      {ae_metrics['pr_auc']:.4f} | Recall@10%: {ae_metrics['recall_at_budget']['recall_at_10pct_budget']:.2f}")
    logger.info(f"Calibrated Model PR-AUC: {clf_metrics['pr_auc']:.4f} | Recall@10%: {clf_metrics['recall_at_budget']['recall_at_10pct_budget']:.2f}")
    logger.info(f"Max Net Projected Savings: INR {cost_curve_results['max_net_savings_rs']:,.2f}")
    return summary_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BDS-33 CLI")
    parser.add_argument("command", choices=["generate", "train", "run_all"], default="run_all", nargs="?")
    args = parser.parse_args()
    
    if args.command == "generate":
        run_simulation()
    elif args.command in ["train", "run_all"]:
        run_training_pipeline()
