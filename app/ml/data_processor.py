"""
Data Processor Module
Handles loading, preprocessing, and splitting of credit card transaction data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
from typing import Tuple, Optional
import os

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "saved_models"
SCALER_PATH = MODELS_DIR / "scaler.joblib"


class DataProcessor:
    """Handles all data preprocessing operations for fraud detection."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self._ensure_models_dir()
    
    def _ensure_models_dir(self):
        """Create models directory if it doesn't exist."""
        MODELS_DIR.mkdir(exist_ok=True)
    
    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load the credit card dataset."""
        path = filepath or DATA_PATH
        df = pd.read_csv(path)
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Get feature column names (excluding Time and Class)."""
        # V1-V28 are already PCA transformed, Amount needs scaling
        self.feature_columns = [col for col in df.columns if col not in ['Time', 'Class']]
        return self.feature_columns
    
    def preprocess(
        self, 
        df: pd.DataFrame, 
        fit_scaler: bool = True,
        apply_smote: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the dataset.
        
        Args:
            df: Input DataFrame
            fit_scaler: Whether to fit the scaler (True for training)
            apply_smote: Whether to apply SMOTE for class balancing
        
        Returns:
            Tuple of (features, labels)
        """
        if self.feature_columns is None:
            self.get_feature_columns(df)
        
        X = df[self.feature_columns].values
        y = df['Class'].values if 'Class' in df.columns else None
        
        # Scale features
        if fit_scaler:
            X = self.scaler.fit_transform(X)
            self.save_scaler()
        else:
            X = self.scaler.transform(X)
        
        # Apply SMOTE if requested and labels exist
        if apply_smote and y is not None:
            smote = SMOTE(random_state=42, sampling_strategy=0.5)
            X, y = smote.fit_resample(X, y)
        
        return X, y
    
    def train_test_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets."""
        stratify_param = y if stratify else None
        return train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=stratify_param
        )
    
    def prepare_training_data(
        self, 
        apply_smote: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete pipeline: load, preprocess, and split data for training.
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        df = self.load_data()
        X, y = self.preprocess(df, fit_scaler=True, apply_smote=apply_smote)
        return self.train_test_split(X, y)
    
    def prepare_single_transaction(self, transaction: dict) -> np.ndarray:
        """Prepare a single transaction for prediction."""
        self.load_scaler()
        
        # Create DataFrame from transaction
        if self.feature_columns is None:
            # Default feature columns
            self.feature_columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8',
                                   'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15',
                                   'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22',
                                   'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        
        features = [transaction.get(col, 0) for col in self.feature_columns]
        X = np.array(features).reshape(1, -1)
        X = self.scaler.transform(X)
        return X
    
    def save_scaler(self):
        """Save the fitted scaler to disk."""
        joblib.dump(self.scaler, SCALER_PATH)
    
    def load_scaler(self):
        """Load the scaler from disk."""
        if SCALER_PATH.exists():
            self.scaler = joblib.load(SCALER_PATH)
    
    def get_data_stats(self) -> dict:
        """Get statistics about the dataset."""
        df = self.load_data()
        
        fraud_count = df[df['Class'] == 1].shape[0]
        normal_count = df[df['Class'] == 0].shape[0]
        
        return {
            "total_transactions": len(df),
            "fraud_transactions": fraud_count,
            "normal_transactions": normal_count,
            "fraud_percentage": round(fraud_count / len(df) * 100, 4),
            "feature_count": len(self.get_feature_columns(df)),
            "amount_stats": {
                "mean": round(df['Amount'].mean(), 2),
                "std": round(df['Amount'].std(), 2),
                "min": round(df['Amount'].min(), 2),
                "max": round(df['Amount'].max(), 2)
            }
        }
    
    def get_sample_transactions(self, n: int = 10, fraud_only: bool = False) -> list:
        """Get sample transactions from the dataset."""
        df = self.load_data()
        
        if fraud_only:
            df = df[df['Class'] == 1]
        
        sample = df.sample(n=min(n, len(df)))
        return sample.to_dict('records')


# Singleton instance
data_processor = DataProcessor()
