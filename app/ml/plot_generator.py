"""
Plot Generator Module
Generates and saves all evaluation metrics plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots"

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")


class PlotGenerator:
    """Generates and saves evaluation metric plots."""
    
    def __init__(self):
        self._ensure_plots_dir()
        
        # Custom colors matching our UI
        self.colors = {
            'primary': '#6366f1',
            'secondary': '#8b5cf6',
            'success': '#10b981',
            'danger': '#ef4444',
            'warning': '#f59e0b',
            'info': '#3b82f6',
            'bg': '#0a0a0f',
            'card': '#12121a',
            'text': '#ffffff',
            'muted': '#a0a0b0'
        }
        
        self.model_colors = {
            'isolation_forest': '#6366f1',
            'autoencoder': '#8b5cf6',
            'xgboost': '#10b981',
            'ensemble': '#f59e0b'
        }
    
    def _ensure_plots_dir(self):
        """Create plots directory if it doesn't exist."""
        PLOTS_DIR.mkdir(exist_ok=True)
    
    def _setup_figure(self, figsize=(10, 6)):
        """Set up a figure with our custom styling."""
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.colors['bg'])
        ax.set_facecolor(self.colors['card'])
        ax.tick_params(colors=self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        ax.spines['top'].set_color(self.colors['card'])
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['right'].set_color(self.colors['card'])
        return fig, ax
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> str:
        """Generate and save confusion matrix heatmap."""
        fig, ax = self._setup_figure(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize for display
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'],
            ax=ax, cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted', color=self.colors['text'], fontsize=12)
        ax.set_ylabel('Actual', color=self.colors['text'], fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', 
                    color=self.colors['text'], fontsize=14, fontweight='bold')
        
        filepath = PLOTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=self.colors['bg'], edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> str:
        """Generate ROC curves for multiple models."""
        fig, ax = self._setup_figure(figsize=(10, 8))
        
        for model_name, y_proba in predictions.items():
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            color = self.model_colors.get(model_name, self.colors['primary'])
            ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{model_name}')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, linewidth=1)
        
        ax.set_xlabel('False Positive Rate', color=self.colors['text'], fontsize=12)
        ax.set_ylabel('True Positive Rate', color=self.colors['text'], fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', 
                    color=self.colors['text'], fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', facecolor=self.colors['card'], 
                 edgecolor=self.colors['muted'])
        ax.grid(True, alpha=0.2)
        
        filepath = PLOTS_DIR / "roc_curves.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=self.colors['bg'], edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_precision_recall_curves(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray]
    ) -> str:
        """Generate Precision-Recall curves for multiple models."""
        fig, ax = self._setup_figure(figsize=(10, 8))
        
        for model_name, y_proba in predictions.items():
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            color = self.model_colors.get(model_name, self.colors['primary'])
            ax.plot(recall, precision, color=color, linewidth=2, label=f'{model_name}')
        
        ax.set_xlabel('Recall', color=self.colors['text'], fontsize=12)
        ax.set_ylabel('Precision', color=self.colors['text'], fontsize=12)
        ax.set_title('Precision-Recall Curves', 
                    color=self.colors['text'], fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', facecolor=self.colors['card'], 
                 edgecolor=self.colors['muted'])
        ax.grid(True, alpha=0.2)
        
        filepath = PLOTS_DIR / "precision_recall_curves.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=self.colors['bg'], edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_model_comparison(
        self,
        metrics: Dict[str, Dict[str, float]]
    ) -> str:
        """Generate bar chart comparing model metrics."""
        fig, ax = self._setup_figure(figsize=(12, 6))
        
        metric_names = ['precision', 'recall', 'f1_score', 'accuracy']
        x = np.arange(len(metric_names))
        width = 0.2
        
        for i, (model_name, model_metrics) in enumerate(metrics.items()):
            values = [model_metrics.get(m, 0) for m in metric_names]
            color = self.model_colors.get(model_name, self.colors['primary'])
            bars = ax.bar(x + i * width, values, width, label=model_name, color=color)
            
            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8,
                           color=self.colors['muted'])
        
        ax.set_ylabel('Score', color=self.colors['text'], fontsize=12)
        ax.set_title('Model Performance Comparison', 
                    color=self.colors['text'], fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(metrics) - 1) / 2)
        ax.set_xticklabels(['Precision', 'Recall', 'F1 Score', 'Accuracy'])
        ax.legend(loc='upper right', facecolor=self.colors['card'], 
                 edgecolor=self.colors['muted'])
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.2, axis='y')
        
        filepath = PLOTS_DIR / "model_comparison.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=self.colors['bg'], edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        top_n: int = 15
    ) -> str:
        """Generate horizontal bar chart for feature importance."""
        fig, ax = self._setup_figure(figsize=(10, 8))
        
        # Get top N features
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        # Reverse for horizontal bar chart
        features = features[::-1]
        values = values[::-1]
        
        y_pos = np.arange(len(features))
        
        # Create gradient colors
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        
        bars = ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, color=self.colors['text'])
        ax.set_xlabel('Importance', color=self.colors['text'], fontsize=12)
        ax.set_title('Top Feature Importance (XGBoost)', 
                    color=self.colors['text'], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        
        filepath = PLOTS_DIR / "feature_importance.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=self.colors['bg'], edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_class_distribution(
        self,
        normal_count: int,
        fraud_count: int
    ) -> str:
        """Generate pie chart for class distribution."""
        fig, ax = self._setup_figure(figsize=(8, 8))
        
        sizes = [normal_count, fraud_count]
        labels = [f'Normal\n({normal_count:,})', f'Fraud\n({fraud_count:,})']
        colors = [self.colors['primary'], self.colors['danger']]
        explode = (0, 0.1)
        
        wedges, texts, autotexts = ax.pie(
            sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.2f%%', shadow=True, startangle=90,
            textprops={'color': self.colors['text'], 'fontsize': 11}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Transaction Class Distribution', 
                    color=self.colors['text'], fontsize=14, fontweight='bold')
        
        filepath = PLOTS_DIR / "class_distribution.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=self.colors['bg'], edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def plot_amount_distribution(
        self,
        amounts: np.ndarray,
        labels: np.ndarray
    ) -> str:
        """Generate transaction amount distribution plot."""
        fig, ax = self._setup_figure(figsize=(12, 6))
        
        # Separate normal and fraud amounts
        normal_amounts = amounts[labels == 0]
        fraud_amounts = amounts[labels == 1]
        
        # Log scale for better visualization
        bins = np.logspace(0, np.log10(max(amounts) + 1), 50)
        
        ax.hist(normal_amounts, bins=bins, alpha=0.7, label='Normal', 
               color=self.colors['primary'])
        ax.hist(fraud_amounts, bins=bins, alpha=0.7, label='Fraud', 
               color=self.colors['danger'])
        
        ax.set_xscale('log')
        ax.set_xlabel('Transaction Amount ($)', color=self.colors['text'], fontsize=12)
        ax.set_ylabel('Frequency', color=self.colors['text'], fontsize=12)
        ax.set_title('Transaction Amount Distribution', 
                    color=self.colors['text'], fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', facecolor=self.colors['card'], 
                 edgecolor=self.colors['muted'])
        ax.grid(True, alpha=0.2)
        
        filepath = PLOTS_DIR / "amount_distribution.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, facecolor=self.colors['bg'], edgecolor='none')
        plt.close()
        
        return str(filepath)
    
    def generate_all_plots(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, tuple],  # {model_name: (y_pred, y_proba)}
        metrics: Dict[str, Dict[str, float]],
        feature_importance: Optional[Dict[str, float]] = None,
        data_stats: Optional[Dict] = None,
        amounts: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """Generate all plots and return their file paths."""
        plots = {}
        
        # Confusion matrices for each model
        for model_name, (y_pred, y_proba) in predictions.items():
            plots[f'confusion_matrix_{model_name}'] = self.plot_confusion_matrix(
                y_true, y_pred, model_name
            )
        
        # ROC curves
        proba_dict = {name: pred[1] for name, pred in predictions.items()}
        plots['roc_curves'] = self.plot_roc_curves(y_true, proba_dict)
        
        # Precision-Recall curves
        plots['precision_recall_curves'] = self.plot_precision_recall_curves(y_true, proba_dict)
        
        # Model comparison
        plots['model_comparison'] = self.plot_model_comparison(metrics)
        
        # Feature importance
        if feature_importance:
            plots['feature_importance'] = self.plot_feature_importance(feature_importance)
        
        # Class distribution
        if data_stats:
            plots['class_distribution'] = self.plot_class_distribution(
                data_stats.get('normal_transactions', 0),
                data_stats.get('fraud_transactions', 0)
            )
        
        # Amount distribution
        if amounts is not None and y_true is not None:
            plots['amount_distribution'] = self.plot_amount_distribution(amounts, y_true)
        
        return plots


# Singleton instance
plot_generator = PlotGenerator()
