"""
Isolation Forest Model
Unsupervised anomaly detection for fraud identification.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "saved_models"
MODEL_PATH = MODELS_DIR / "isolation_forest.joblib"


class IsolationForestModel:
    """Isolation Forest for unsupervised fraud detection."""
    
    def __init__(
        self, 
        contamination: float = 0.001,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        random_state: int = 42
    ):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_trained = False
        self._ensure_models_dir()
    
    def _ensure_models_dir(self):
        """Create models directory if it doesn't exist."""
        MODELS_DIR.mkdir(exist_ok=True)
    
    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> dict:
        """
        Train the Isolation Forest model.
        
        Note: Isolation Forest is unsupervised, but we can use y for evaluation.
        """
        # Train only on normal transactions for better anomaly detection
        if y is not None:
            X_normal = X[y == 0]
            self.model.fit(X_normal)
        else:
            self.model.fit(X)
        
        self.is_trained = True
        self.save()
        
        return {"status": "trained", "samples": len(X)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.
        
        Returns:
            Binary predictions (1 = fraud, 0 = normal)
        """
        if not self.is_trained:
            self.load()
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(X)
        # Convert to our format: 1 = fraud, 0 = normal
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (probability-like).
        
        Returns:
            Array of scores where higher = more likely fraud
        """
        if not self.is_trained:
            self.load()
        
        # decision_function returns negative values for anomalies
        scores = -self.model.decision_function(X)
        # Normalize to 0-1 range
        scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return scores_normalized
    
    def incremental_train(self, X_new: np.ndarray, y_new: Optional[np.ndarray] = None):
        """
        Perform incremental training with new data.
        
        Note: Isolation Forest doesn't support true incremental learning,
        so we retrain with the new data only (warm start simulation).
        """
        if not self.is_trained:
            return self.train(X_new, y_new)
        
        # For Isolation Forest, we create a new model with the new data
        # This is a simplified incremental approach
        if y_new is not None:
            X_normal = X_new[y_new == 0]
            if len(X_normal) > 0:
                self.model.fit(X_normal)
        else:
            self.model.fit(X_new)
        
        self.save()
        return {"status": "incrementally_trained", "new_samples": len(X_new)}
    
    def save(self):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'contamination': self.contamination,
            'is_trained': self.is_trained
        }, MODEL_PATH)
    
    def load(self) -> bool:
        """Load model from disk."""
        if MODEL_PATH.exists():
            data = joblib.load(MODEL_PATH)
            self.model = data['model']
            self.contamination = data['contamination']
            self.is_trained = data['is_trained']
            return True
        return False
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "name": "Isolation Forest",
            "type": "Unsupervised Anomaly Detection",
            "contamination": self.contamination,
            "n_estimators": self.model.n_estimators,
            "is_trained": self.is_trained
        }


# Singleton instance
isolation_forest_model = IsolationForestModel()
