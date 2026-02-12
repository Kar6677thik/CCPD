"""
Gradient Boosting Model (XGBoost)
Supervised classification for fraud detection.
"""

import numpy as np
from xgboost import XGBClassifier
import joblib
from pathlib import Path
from typing import Optional, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "saved_models"
MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"


class GradientBoostingModel:
    """XGBoost classifier for supervised fraud detection."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        scale_pos_weight: float = 10.0,  # Handle class imbalance
        random_state: int = 42
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_importance = None
        self._ensure_models_dir()
    
    def _ensure_models_dir(self):
        """Create models directory if it doesn't exist."""
        MODELS_DIR.mkdir(exist_ok=True)
    
    def train(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        eval_set: Optional[tuple] = None
    ) -> dict:
        """
        Train the XGBoost classifier.
        """
        if y is None:
            raise ValueError("XGBoost requires labels for training")
        
        if eval_set:
            self.model.fit(
                X, y,
                eval_set=[eval_set],
                verbose=False
            )
        else:
            self.model.fit(X, y, verbose=False)
        
        self.is_trained = True
        self.feature_importance = self.model.feature_importances_
        self.save()
        
        return {
            "status": "trained",
            "samples": len(X),
            "n_estimators": self.model.n_estimators
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict fraud labels.
        
        Returns:
            Binary predictions (1 = fraud, 0 = normal)
        """
        if not self.is_trained:
            self.load()
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get fraud probability scores.
        
        Returns:
            Array of probabilities for fraud class
        """
        if not self.is_trained:
            self.load()
        
        # XGBoost returns [prob_class_0, prob_class_1]
        proba = self.model.predict_proba(X)
        return proba[:, 1]  # Return probability of fraud
    
    def incremental_train(
        self, 
        X_new: np.ndarray, 
        y_new: np.ndarray
    ) -> dict:
        """
        Perform incremental training with new data.
        
        XGBoost doesn't support true incremental learning,
        but we can add more estimators to the existing model.
        """
        if not self.is_trained:
            return self.train(X_new, y_new)
        
        if y_new is None:
            raise ValueError("XGBoost requires labels for training")
        
        # Get current booster
        booster = self.model.get_booster()
        
        # Train additional trees
        additional_model = XGBClassifier(
            n_estimators=20,  # Add 20 more trees
            max_depth=self.model.max_depth,
            learning_rate=self.model.learning_rate * 0.5,  # Reduced LR
            scale_pos_weight=self.model.scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Use the trained model for prediction and combine
        # This is a simplified approach - retrain with new data
        additional_model.fit(X_new, y_new, xgb_model=booster, verbose=False)
        self.model = additional_model
        
        self.feature_importance = self.model.feature_importances_
        self.save()
        
        return {
            "status": "incrementally_trained",
            "new_samples": len(X_new)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance ranking."""
        if self.feature_importance is None:
            return {}
        
        feature_names = [f'V{i+1}' for i in range(28)] + ['Amount']
        importance_dict = {
            name: float(imp) 
            for name, imp in zip(feature_names, self.feature_importance)
        }
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    
    def save(self):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance
        }, MODEL_PATH)
    
    def load(self) -> bool:
        """Load model from disk."""
        if MODEL_PATH.exists():
            data = joblib.load(MODEL_PATH)
            self.model = data['model']
            self.is_trained = data['is_trained']
            self.feature_importance = data['feature_importance']
            return True
        return False
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "name": "XGBoost",
            "type": "Supervised Gradient Boosting",
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "learning_rate": self.model.learning_rate,
            "is_trained": self.is_trained
        }


# Singleton instance
gradient_boosting_model = GradientBoostingModel()
