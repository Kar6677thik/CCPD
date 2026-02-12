"""
Training Router
API endpoints for model training and incremental learning.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
import asyncio

from app.ml.data_processor import data_processor
from app.ml.ensemble import ensemble_model
from app.ml.evaluator import evaluator
from app.ml.plot_generator import plot_generator
from app.ml.models.gradient_boosting import gradient_boosting_model

router = APIRouter(prefix="/api/train", tags=["training"])

# Training state
training_state = {
    "is_training": False,
    "progress": 0,
    "status": "idle",
    "last_result": None
}


class TrainingConfig(BaseModel):
    """Configuration for training."""
    apply_smote: bool = True
    test_size: float = 0.2


class IncrementalTrainingConfig(BaseModel):
    """Configuration for incremental training."""
    sample_size: int = 1000  # Number of samples to use


def run_training(config: TrainingConfig):
    """Background task for training all models."""
    global training_state
    
    try:
        training_state["is_training"] = True
        training_state["status"] = "loading_data"
        training_state["progress"] = 10
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = data_processor.prepare_training_data(
            apply_smote=config.apply_smote
        )
        
        training_state["status"] = "training_models"
        training_state["progress"] = 30
        
        # Train all models
        results = ensemble_model.train_all(X_train, y_train, X_test, y_test)
        
        training_state["progress"] = 70
        training_state["status"] = "evaluating"
        
        # Evaluate on test set
        y_pred = ensemble_model.predict(X_test)
        y_proba = ensemble_model.predict_proba(X_test)
        
        metrics = evaluator.evaluate(y_test, y_pred, y_proba, "ensemble")
        
        training_state["progress"] = 85
        training_state["status"] = "generating_plots"
        
        # Generate plots
        try:
            # Get predictions from all models for plotting
            predictions = {}
            all_metrics = {}
            
            for name, model in ensemble_model.models.items():
                if model.is_trained:
                    pred = model.predict(X_test)
                    proba = model.predict_proba(X_test)
                    predictions[name] = (pred, proba)
                    all_metrics[name] = evaluator.evaluate(y_test, pred, proba, name)
            
            # Add ensemble
            predictions['ensemble'] = (y_pred, y_proba)
            all_metrics['ensemble'] = metrics
            
            # Feature importance
            feature_importance = gradient_boosting_model.get_feature_importance()
            
            # Data stats
            data_stats = data_processor.get_data_stats()
            
            # Generate all plots
            plot_paths = plot_generator.generate_all_plots(
                y_true=y_test,
                predictions=predictions,
                metrics=all_metrics,
                feature_importance=feature_importance,
                data_stats=data_stats
            )
            
            results["plots_generated"] = plot_paths
        except Exception as plot_error:
            results["plot_error"] = str(plot_error)
        
        training_state["progress"] = 100
        training_state["status"] = "completed"
        training_state["last_result"] = {
            "training_results": results,
            "evaluation_metrics": metrics,
            "test_samples": len(y_test),
            "train_samples": len(y_train)
        }
        
    except Exception as e:
        training_state["status"] = f"error: {str(e)}"
    finally:
        training_state["is_training"] = False


@router.post("/")
async def train_models(
    background_tasks: BackgroundTasks,
    config: Optional[TrainingConfig] = None
):
    """
    Start training all models.
    Training runs in the background - check /status for progress.
    """
    if training_state["is_training"]:
        raise HTTPException(
            status_code=409, 
            detail="Training already in progress"
        )
    
    config = config or TrainingConfig()
    background_tasks.add_task(run_training, config)
    
    return {
        "message": "Training started",
        "config": config.model_dump()
    }


@router.post("/sync")
async def train_models_sync(config: Optional[TrainingConfig] = None):
    """
    Train all models synchronously (blocking).
    Use for smaller datasets or testing.
    """
    if training_state["is_training"]:
        raise HTTPException(
            status_code=409, 
            detail="Training already in progress"
        )
    
    config = config or TrainingConfig()
    
    try:
        training_state["is_training"] = True
        training_state["status"] = "training"
        
        # Load and prepare data
        X_train, X_test, y_train, y_test = data_processor.prepare_training_data(
            apply_smote=config.apply_smote
        )
        
        # Train all models
        results = ensemble_model.train_all(X_train, y_train, X_test, y_test)
        
        # Evaluate
        y_pred = ensemble_model.predict(X_test)
        y_proba = ensemble_model.predict_proba(X_test)
        metrics = evaluator.evaluate(y_test, y_pred, y_proba, "ensemble")
        
        training_state["last_result"] = {
            "training_results": results,
            "evaluation_metrics": metrics
        }
        
        return {
            "message": "Training completed",
            "training_results": results,
            "evaluation_metrics": metrics,
            "test_samples": len(y_test),
            "train_samples": len(y_train)
        }
    finally:
        training_state["is_training"] = False
        training_state["status"] = "idle"


@router.post("/incremental")
async def incremental_train(config: Optional[IncrementalTrainingConfig] = None):
    """
    Perform incremental training with new data samples.
    """
    config = config or IncrementalTrainingConfig()
    
    try:
        # Get new samples (simulating new data arrival)
        df = data_processor.load_data()
        sample = df.sample(n=min(config.sample_size, len(df)))
        
        X, y = data_processor.preprocess(sample, fit_scaler=False)
        
        # Incremental training
        results = ensemble_model.incremental_train_all(X, y)
        
        return {
            "message": "Incremental training completed",
            "samples_used": len(sample),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_training_status():
    """
    Get current training status.
    """
    return {
        "is_training": training_state["is_training"],
        "progress": training_state["progress"],
        "status": training_state["status"],
        "last_result": training_state["last_result"]
    }


@router.get("/models")
async def get_model_status():
    """
    Get status of all models.
    """
    return ensemble_model.get_model_status()


@router.post("/load")
async def load_models():
    """
    Load all saved models from disk.
    """
    results = ensemble_model.load_all()
    return {
        "message": "Models loaded",
        "results": results
    }
