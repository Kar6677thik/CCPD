"""
Metrics Router
API endpoints for model evaluation metrics.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import numpy as np

from app.ml.data_processor import data_processor
from app.ml.ensemble import ensemble_model
from app.ml.evaluator import evaluator
from app.ml.models.gradient_boosting import gradient_boosting_model

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/")
async def get_all_metrics():
    """
    Get the latest evaluation metrics for the ensemble model.
    """
    history = evaluator.get_history()
    
    if not history:
        return {
            "message": "No metrics available. Train the models first.",
            "metrics": None
        }
    
    return {
        "latest": history[-1],
        "history_count": len(history)
    }


@router.get("/comparison")
async def compare_models(sample_size: int = 5000):
    """
    Compare performance of all models on a sample of test data.
    """
    try:
        # Get sample data
        df = data_processor.load_data()
        sample = df.sample(n=min(sample_size, len(df)))
        
        X, y = data_processor.preprocess(sample, fit_scaler=False)
        
        if y is None:
            raise HTTPException(status_code=400, detail="No labels in data")
        
        # Get predictions from each model
        predictions = {}
        
        for name, model in ensemble_model.models.items():
            if model.is_trained:
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)
                predictions[name] = (y_pred, y_proba)
        
        if not predictions:
            return {"message": "No trained models available"}
        
        # Also add ensemble
        y_pred_ensemble = ensemble_model.predict(X)
        y_proba_ensemble = ensemble_model.predict_proba(X)
        predictions['ensemble'] = (y_pred_ensemble, y_proba_ensemble)
        
        # Compare
        results = evaluator.compare_models(y, predictions)
        
        return {
            "sample_size": len(sample),
            "comparison": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-importance")
async def get_feature_importance():
    """
    Get feature importance from the XGBoost model.
    """
    if not gradient_boosting_model.is_trained:
        return {"message": "XGBoost model not trained"}
    
    importance = gradient_boosting_model.get_feature_importance()
    
    # Get top 15 features
    top_features = dict(list(importance.items())[:15])
    
    return {
        "all_features": importance,
        "top_features": top_features
    }


@router.get("/roc-curve")
async def get_roc_curve(sample_size: int = 5000):
    """
    Get ROC curve data for plotting.
    """
    try:
        df = data_processor.load_data()
        sample = df.sample(n=min(sample_size, len(df)))
        
        X, y = data_processor.preprocess(sample, fit_scaler=False)
        
        if y is None:
            raise HTTPException(status_code=400, detail="No labels in data")
        
        curves = {}
        
        for name, model in ensemble_model.models.items():
            if model.is_trained:
                y_proba = model.predict_proba(X)
                curves[name] = evaluator.get_roc_curve_data(y, y_proba)
        
        # Ensemble
        y_proba_ensemble = ensemble_model.predict_proba(X)
        curves['ensemble'] = evaluator.get_roc_curve_data(y, y_proba_ensemble)
        
        return {"curves": curves}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pr-curve")
async def get_pr_curve(sample_size: int = 5000):
    """
    Get Precision-Recall curve data for plotting.
    """
    try:
        df = data_processor.load_data()
        sample = df.sample(n=min(sample_size, len(df)))
        
        X, y = data_processor.preprocess(sample, fit_scaler=False)
        
        if y is None:
            raise HTTPException(status_code=400, detail="No labels in data")
        
        curves = {}
        
        for name, model in ensemble_model.models.items():
            if model.is_trained:
                y_proba = model.predict_proba(X)
                curves[name] = evaluator.get_pr_curve_data(y, y_proba)
        
        return {"curves": curves}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_metrics_history():
    """
    Get historical evaluation metrics.
    """
    return {"history": evaluator.get_history()}


@router.delete("/history")
async def clear_metrics_history():
    """
    Clear evaluation history.
    """
    evaluator.clear_history()
    return {"message": "History cleared"}
