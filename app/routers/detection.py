"""
Detection Router
API endpoints for fraud detection on transactions.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np

from app.ml.data_processor import data_processor
from app.ml.ensemble import ensemble_model

router = APIRouter(prefix="/api/detect", tags=["detection"])


class Transaction(BaseModel):
    """Single transaction for fraud detection."""
    V1: float = 0
    V2: float = 0
    V3: float = 0
    V4: float = 0
    V5: float = 0
    V6: float = 0
    V7: float = 0
    V8: float = 0
    V9: float = 0
    V10: float = 0
    V11: float = 0
    V12: float = 0
    V13: float = 0
    V14: float = 0
    V15: float = 0
    V16: float = 0
    V17: float = 0
    V18: float = 0
    V19: float = 0
    V20: float = 0
    V21: float = 0
    V22: float = 0
    V23: float = 0
    V24: float = 0
    V25: float = 0
    V26: float = 0
    V27: float = 0
    V28: float = 0
    Amount: float = 0


class DetectionResponse(BaseModel):
    """Response for fraud detection."""
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    model_predictions: Dict[str, dict]


class BatchDetectionRequest(BaseModel):
    """Batch of transactions for fraud detection."""
    transactions: List[Transaction]


class BatchDetectionResponse(BaseModel):
    """Response for batch fraud detection."""
    results: List[DetectionResponse]
    fraud_count: int
    total: int
    fraud_rate: float


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.4:
        return "MEDIUM"
    elif probability >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"


@router.post("/", response_model=DetectionResponse)
async def detect_fraud(transaction: Transaction):
    """
    Detect if a single transaction is fraudulent.
    """
    try:
        # Convert transaction to feature array
        X = data_processor.prepare_single_transaction(transaction.model_dump())
        
        # Get ensemble prediction
        proba = ensemble_model.predict_proba(X)[0]
        is_fraud = proba >= 0.5
        
        # Get individual model predictions
        model_predictions = ensemble_model.predict_individual(X)
        
        return DetectionResponse(
            is_fraud=bool(is_fraud),
            fraud_probability=round(float(proba), 4),
            risk_level=get_risk_level(proba),
            model_predictions=model_predictions
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchDetectionResponse)
async def detect_fraud_batch(request: BatchDetectionRequest):
    """
    Detect fraud in a batch of transactions.
    """
    try:
        results = []
        fraud_count = 0
        
        for txn in request.transactions:
            X = data_processor.prepare_single_transaction(txn.model_dump())
            proba = ensemble_model.predict_proba(X)[0]
            is_fraud = proba >= 0.5
            
            if is_fraud:
                fraud_count += 1
            
            model_predictions = ensemble_model.predict_individual(X)
            
            results.append(DetectionResponse(
                is_fraud=bool(is_fraud),
                fraud_probability=round(float(proba), 4),
                risk_level=get_risk_level(proba),
                model_predictions=model_predictions
            ))
        
        total = len(request.transactions)
        
        return BatchDetectionResponse(
            results=results,
            fraud_count=fraud_count,
            total=total,
            fraud_rate=round(fraud_count / total * 100, 2) if total > 0 else 0
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample")
async def get_sample_predictions(n: int = 10, fraud_only: bool = False):
    """
    Get sample predictions on random transactions from the dataset.
    """
    try:
        samples = data_processor.get_sample_transactions(n, fraud_only)
        results = []
        
        for sample in samples:
            actual_class = sample.pop('Class', None)
            X = data_processor.prepare_single_transaction(sample)
            
            try:
                proba = ensemble_model.predict_proba(X)[0]
                is_fraud = proba >= 0.5
                
                results.append({
                    "transaction": sample,
                    "actual_class": int(actual_class) if actual_class is not None else None,
                    "predicted_fraud": bool(is_fraud),
                    "fraud_probability": round(float(proba), 4),
                    "risk_level": get_risk_level(proba),
                    "correct": (is_fraud == actual_class) if actual_class is not None else None
                })
            except Exception:
                # Models might not be trained yet
                results.append({
                    "transaction": sample,
                    "actual_class": int(actual_class) if actual_class is not None else None,
                    "error": "Models not trained"
                })
        
        return {"samples": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Check if detection service is ready."""
    status = ensemble_model.get_model_status()
    all_trained = all(m.get('is_trained', False) for m in status.values())
    
    return {
        "status": "ready" if all_trained else "models_not_trained",
        "models": status
    }
