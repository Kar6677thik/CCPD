"""
Autoencoder Model (Scikit-learn based)
Neural network anomaly detection via reconstruction error.
Uses MLPRegressor as a simple autoencoder alternative - no TensorFlow needed.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "saved_models"
MODEL_PATH = MODELS_DIR / "autoencoder.joblib"


class AutoencoderModel:
    """
    Autoencoder-like model using MLPRegressor.
    Trains to reconstruct input features - anomalies have high reconstruction error.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = (22, 14, 22),
        max_iter: int = 200,
        random_state: int = 42
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False
        )
        self.threshold = None
        self.is_trained = False
        self._ensure_models_dir()
    
    def _ensure_models_dir(self):
        """Create models directory if it doesn't exist."""
        MODELS_DIR.mkdir(exist_ok=True)
    
    def train(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        **kwargs
    ) -> dict:
        """
        Train the autoencoder on normal transactions.
        The model learns to reconstruct input features.
        """
        # Train only on normal transactions
        if y is not None:
            X_train = X[y == 0]
        else:
            X_train = X
        
        # Train to reconstruct input
        self.model.fit(X_train, X_train)
        
        # Calculate threshold based on training reconstruction errors
        reconstructions = self.model.predict(X_train)
        train_errors = np.mean(np.square(X_train - reconstructions), axis=1)
        self.threshold = np.percentile(train_errors, 95)  # 95th percentile
        
        self.is_trained = True
        self.save()
        
        return {
            "status": "trained",
            "samples": len(X_train),
            "threshold": float(self.threshold),
            "iterations": self.model.n_iter_
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error.
        
        Returns:
            Binary predictions (1 = fraud, 0 = normal)
        """
        if not self.is_trained:
            self.load()
        
        reconstructions = self.model.predict(X)
        errors = np.mean(np.square(X - reconstructions), axis=1)
        return (errors > self.threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores based on reconstruction error.
        
        Returns:
            Array of scores where higher = more likely fraud
        """
        if not self.is_trained:
            self.load()
        
        reconstructions = self.model.predict(X)
        errors = np.mean(np.square(X - reconstructions), axis=1)
        
        # Normalize to 0-1 range
        scores = errors / (self.threshold * 2)
        scores = np.clip(scores, 0, 1)
        return scores
    
    def incremental_train(
        self, 
        X_new: np.ndarray, 
        y_new: Optional[np.ndarray] = None,
        **kwargs
    ) -> dict:
        """
        Perform incremental training (warm start) with new data.
        """
        if not self.is_trained:
            return self.train(X_new, y_new)
        
        # MLPRegressor supports warm_start
        self.model.set_params(warm_start=True, max_iter=50)
        
        if y_new is not None:
            X_train = X_new[y_new == 0]
        else:
            X_train = X_new
        
        if len(X_train) > 0:
            self.model.fit(X_train, X_train)
            
            # Update threshold
            reconstructions = self.model.predict(X_train)
            new_errors = np.mean(np.square(X_train - reconstructions), axis=1)
            self.threshold = (self.threshold + np.percentile(new_errors, 95)) / 2
        
        self.model.set_params(warm_start=False)
        self.save()
        
        return {
            "status": "incrementally_trained",
            "new_samples": len(X_train),
            "new_threshold": float(self.threshold)
        }
    
    def save(self):
        """Save model and threshold to disk."""
        joblib.dump({
            'model': self.model,
            'threshold': self.threshold,
            'is_trained': self.is_trained,
            'hidden_layer_sizes': self.hidden_layer_sizes
        }, MODEL_PATH)
    
    def load(self) -> bool:
        """Load model and threshold from disk."""
        if MODEL_PATH.exists():
            data = joblib.load(MODEL_PATH)
            self.model = data['model']
            self.threshold = data['threshold']
            self.is_trained = data['is_trained']
            self.hidden_layer_sizes = data.get('hidden_layer_sizes', (22, 14, 22))
            return True
        return False
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "name": "Autoencoder (MLP)",
            "type": "Neural Network Anomaly Detection",
            "hidden_layers": self.hidden_layer_sizes,
            "threshold": float(self.threshold) if self.threshold else None,
            "is_trained": self.is_trained
        }


# Singleton instance
autoencoder_model = AutoencoderModel()
