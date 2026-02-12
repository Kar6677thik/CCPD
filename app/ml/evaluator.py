"""
Model Evaluator
Computes evaluation metrics for fraud detection models.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from typing import Dict, Tuple, Optional


class ModelEvaluator:
    """Evaluates fraud detection models with comprehensive metrics."""
    
    def __init__(self):
        self.history = []
    
    def evaluate(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> Dict:
        """
        Compute all evaluation metrics.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC)
            model_name: Name of the model being evaluated
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "model_name": model_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics["confusion_matrix"] = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
        
        # False Positive Rate
        metrics["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        
        # Fraud Detection Rate (same as Recall/True Positive Rate)
        metrics["fraud_detection_rate"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        
        # AUC scores if probabilities are provided
        if y_proba is not None:
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                metrics["auc_roc"] = 0.0
            
            try:
                metrics["auc_pr"] = float(average_precision_score(y_true, y_proba))
            except ValueError:
                metrics["auc_pr"] = 0.0
        
        # Store in history
        self.history.append(metrics.copy())
        
        return metrics
    
    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Dict]:
        """
        Compare multiple models.
        
        Args:
            y_true: Ground truth labels
            predictions: Dict of {model_name: (y_pred, y_proba)}
        
        Returns:
            Dict of metrics for each model
        """
        results = {}
        
        for model_name, (y_pred, y_proba) in predictions.items():
            results[model_name] = self.evaluate(y_true, y_pred, y_proba, model_name)
        
        return results
    
    def get_roc_curve_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """Get ROC curve data for plotting."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist()
        }
    
    def get_pr_curve_data(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict:
        """Get Precision-Recall curve data for plotting."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist()
        }
    
    def get_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "f1"
    ) -> Tuple[float, float]:
        """
        Find optimal threshold for a given metric.
        
        Args:
            y_true: Ground truth labels
            y_proba: Predicted probabilities
            metric: 'f1', 'recall', or 'precision'
        
        Returns:
            (optimal_threshold, best_metric_value)
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_score = 0
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            elif metric == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        return float(best_threshold), float(best_score)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """Get detailed classification report."""
        return classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Fraud'],
            zero_division=0
        )
    
    def get_history(self) -> list:
        """Get evaluation history."""
        return self.history
    
    def clear_history(self):
        """Clear evaluation history."""
        self.history = []


# Singleton instance
evaluator = ModelEvaluator()
