"""
Unit tests for models: Isolation Forest, Autoencoder, Classifier, and Calibrator
"""
import pytest
import numpy as np
import pandas as pd
from src.models.baselines import IsolationForestBaseline, AutoencoderBaseline
from src.models.classifier import TamperClassifier
from src.models.calibrator import ProbabilityCalibrator

@pytest.fixture
def mock_dataset():
    np.random.seed(42)
    n_samples = 40
    X = pd.DataFrame({
        "feat1": np.random.randn(n_samples),
        "feat2": np.random.uniform(1, 10, n_samples),
        "feat3": np.random.exponential(1.5, n_samples)
    })
    y = np.array([0, 1] * 20)
    return X, y

def test_isolation_forest(mock_dataset):
    X, _ = mock_dataset
    iso = IsolationForestBaseline(contamination=0.2)
    iso.fit(X)
    scores = iso.predict_score(X)
    assert len(scores) == len(X)
    assert np.all((scores >= 0.0) & (scores <= 1.0))

def test_autoencoder_baseline(mock_dataset):
    X, _ = mock_dataset
    ae = AutoencoderBaseline(epochs=5, batch_size=8)
    ae.fit(X)
    scores = ae.predict_score(X)
    assert len(scores) == len(X)
    assert np.all((scores >= 0.0) & (scores <= 1.0))

def test_classifier_and_calibration(mock_dataset):
    X, y = mock_dataset
    clf = TamperClassifier(use_lightgbm=True)
    clf.fit(X[:25], y[:25])
    
    cal = ProbabilityCalibrator(method="isotonic")
    cal.fit(clf, X[25:35], y[25:35])
    
    probs = cal.predict_calibrated_proba(X[35:])
    assert len(probs) == len(X[35:])
    assert np.all((probs >= 0.0) & (probs <= 1.0))
