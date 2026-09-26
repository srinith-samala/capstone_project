"""
Unsupervised Anomaly Detection Baselines: Isolation Forest and PyTorch Deep Autoencoder
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from config.logging_config import setup_logger

logger = setup_logger("AnomalyBaselines")

class IsolationForestBaseline:
    def __init__(self, contamination: float = 0.15, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_names = []

    def fit(self, X: pd.DataFrame):
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.fillna(0.0))
        self.model.fit(X_scaled)
        logger.info("Isolation Forest baseline trained successfully.")
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns anomaly scores normalized to [0, 1] where 1 is highest anomaly risk.
        """
        X_scaled = self.scaler.transform(X[self.feature_names].fillna(0.0))
        # decision_function: lower means more abnormal
        raw_scores = self.model.decision_function(X_scaled)
        # Invert and scale to [0, 1]
        norm_scores = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-6)
        return norm_scores


class PyTorchAutoencoderNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(32, 16, 8)):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderBaseline:
    def __init__(self, epochs: int = 35, lr: float = 0.005, batch_size: int = 16, random_state: int = 42):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state
        torch.manual_seed(random_state)
        self.scaler = StandardScaler()
        self.model: PyTorchAutoencoderNet = None
        self.feature_names = []

    def fit(self, X: pd.DataFrame):
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X.fillna(0.0))
        input_dim = X_scaled.shape[1]
        
        self.model = PyTorchAutoencoderNet(input_dim=input_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(tensor_x)
        loader = torch.utils.data.DataLoader(dataset, batch_size=min(self.batch_size, len(tensor_x)), shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                bx = batch[0]
                optimizer.zero_grad()
                reconstructed = self.model(bx)
                loss = criterion(reconstructed, bx)
                loss.backward()
                optimizer.step()
                
        logger.info(f"PyTorch Deep Autoencoder baseline trained across {self.epochs} epochs.")
        return self

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns normalized reconstruction error as anomaly score in [0, 1].
        """
        self.model.eval()
        X_scaled = self.scaler.transform(X[self.feature_names].fillna(0.0))
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            mse_errors = torch.mean((tensor_x - reconstructed) ** 2, dim=1).numpy()
            
        norm_scores = (mse_errors - mse_errors.min()) / (mse_errors.max() - mse_errors.min() + 1e-6)
        return norm_scores
