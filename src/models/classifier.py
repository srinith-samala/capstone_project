"""
Supervised Gradient Boosting Tamper Classifier
Supports LightGBM and Scikit-Learn HistGradientBoosting with class-imbalance weighting.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from config.logging_config import setup_logger

logger = setup_logger("TamperClassifier")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

class TamperClassifier:
    def __init__(self, random_state: int = 42, use_lightgbm: bool = True):
        self.random_state = random_state
        self.use_lightgbm = use_lightgbm and HAS_LIGHTGBM
        self.model = None
        self.feature_names: List[str] = []
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = list(X.columns)
        X_mat = X.fillna(0.0).values
        
        # Calculate positive class weight to handle class imbalance
        pos_weight = float((len(y) - np.sum(y)) / (np.sum(y) + 1e-4))
        
        if self.use_lightgbm:
            logger.info("Fitting supervised LightGBM Classifier...")
            self.model = lgb.LGBMClassifier(
                n_estimators=120,
                learning_rate=0.05,
                max_depth=5,
                scale_pos_weight=pos_weight,
                random_state=self.random_state,
                verbose=-1
            )
            self.model.fit(X_mat, y)
            self.feature_importances_ = self.model.feature_importances_ / np.sum(self.model.feature_importances_ + 1e-6)
        else:
            logger.info("Fitting Scikit-Learn HistGradientBoostingClassifier...")
            self.model = HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.05,
                max_depth=5,
                class_weight="balanced",
                random_state=self.random_state
            )
            self.model.fit(X_mat, y)
            # Permutation importance surrogate or uniform placeholder
            self.feature_importances_ = np.ones(len(self.feature_names)) / len(self.feature_names)
            
        logger.info("Supervised tamper classifier training complete.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns predicted class probabilities: shape (n_samples, 2)
        """
        X_mat = X[self.feature_names].fillna(0.0).values
        return self.model.predict_proba(X_mat)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)

    def get_feature_importance_df(self) -> pd.DataFrame:
        if self.feature_importances_ is None:
            return pd.DataFrame()
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.feature_importances_
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df
