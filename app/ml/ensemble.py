"""
Ensemble Model
Combines predictions from all three models for robust fraud detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from app.ml.models.isolation_forest import isolation_forest_model
from app.ml.models.autoencoder import autoencoder_model
from app.ml.models.gradient_boosting import gradient_boosting_model


class EnsembleModel:
    """
    Ensemble that combines Isolation Forest, Autoencoder, and XGBoost predictions.
    """
    
    def __init__(self):
        self.models = {
            'isolation_forest': isolation_forest_model,
            'autoencoder': autoencoder_model,
            'xgboost': gradient_boosting_model
        }
        # Default weights (can be tuned based on performance)
        self.weights = {
            'isolation_forest': 0.25,
            'autoencoder': 0.25,
            'xgboost': 0.50  # Supervised model gets higher weight
        }
    
    def set_weights(self, weights: Dict[str, float]):
        """Update model weights for ensemble voting."""
        total = sum(weights.values())
        self.weights = {k: v/total for k, v in weights.items()}
    
    def train_all(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, dict]:
        """
        Train all models in the ensemble.
        """
        results = {}
        
        # Train Isolation Forest
        results['isolation_forest'] = self.models['isolation_forest'].train(X_train, y_train)
        
        # Train Autoencoder
        results['autoencoder'] = self.models['autoencoder'].train(X_train, y_train)
        
        # Train XGBoost with validation set if provided
        if X_val is not None and y_val is not None:
            results['xgboost'] = self.models['xgboost'].train(
                X_train, y_train, 
                eval_set=(X_val, y_val)
            )
        else:
            results['xgboost'] = self.models['xgboost'].train(X_train, y_train)
        
        return results
    
    def predict(self, X: np.ndarray, method: str = 'weighted') -> np.ndarray:
        """
        Get ensemble predictions.
        
        Args:
            X: Features
            method: 'weighted' (weighted average) or 'voting' (majority vote)
        
        Returns:
            Binary predictions (1 = fraud, 0 = normal)
        """
        if method == 'voting':
            return self._predict_voting(X)
        else:
            return self._predict_weighted(X)
    
    def _predict_weighted(self, X: np.ndarray) -> np.ndarray:
        """Weighted probability-based prediction."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
    def _predict_voting(self, X: np.ndarray) -> np.ndarray:
        """Majority voting prediction."""
        predictions = []
        
        for name, model in self.models.items():
            if model.is_trained:
                pred = model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            raise ValueError("No models are trained")
        
        # Stack and vote
        stacked = np.stack(predictions, axis=1)
        return (np.sum(stacked, axis=1) >= 2).astype(int)  # Majority = 2 out of 3
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get weighted probability scores from all models.
        
        Returns:
            Array of fraud probabilities
        """
        weighted_proba = np.zeros(len(X))
        total_weight = 0
        
        for name, model in self.models.items():
            if model.is_trained:
                proba = model.predict_proba(X)
                weighted_proba += proba * self.weights[name]
                total_weight += self.weights[name]
        
        if total_weight == 0:
            raise ValueError("No models are trained")
        
        return weighted_proba / total_weight
    
    def predict_individual(self, X: np.ndarray) -> Dict[str, dict]:
        """
        Get predictions from each model individually.
        """
        results = {}
        
        for name, model in self.models.items():
            if model.is_trained:
                results[name] = {
                    'predictions': model.predict(X).tolist(),
                    'probabilities': model.predict_proba(X).tolist()
                }
        
        return results
    
    def incremental_train_all(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray
    ) -> Dict[str, dict]:
        """
        Perform incremental training on all models.
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                results[name] = model.incremental_train(X_new, y_new)
            except Exception as e:
                results[name] = {"status": "error", "message": str(e)}
        
        return results
    
    def get_model_status(self) -> Dict[str, dict]:
        """Get training status of all models."""
        return {
            name: model.get_model_info()
            for name, model in self.models.items()
        }
    
    def load_all(self) -> Dict[str, bool]:
        """Load all models from disk."""
        return {
            name: model.load()
            for name, model in self.models.items()
        }


# Singleton instance
ensemble_model = EnsembleModel()
