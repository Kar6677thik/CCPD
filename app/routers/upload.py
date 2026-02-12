"""
Upload Router
API endpoints for CSV file upload and batch analysis.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import io
from pathlib import Path

from app.ml.data_processor import data_processor
from app.ml.ensemble import ensemble_model
from app.ml.evaluator import evaluator

router = APIRouter(prefix="/api/upload", tags=["upload"])

PROJECT_ROOT = Path(__file__).parent.parent.parent


class AnalysisResult(BaseModel):
    """Result of CSV analysis."""
    total_transactions: int
    fraud_detected: int
    normal_detected: int
    fraud_percentage: float
    transactions: List[Dict]
    summary_stats: Dict
    risk_distribution: Dict


@router.post("/csv", response_model=AnalysisResult)
async def upload_and_analyze_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and analyze all transactions for fraud.
    
    The CSV should have columns: V1-V28, Amount (Time and Class are optional)
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read the CSV content
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # Validate required columns
        required_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)]
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing}"
            )
        
        # Check if models are trained
        status = ensemble_model.get_model_status()
        if not all(m.get('is_trained', False) for m in status.values()):
            raise HTTPException(
                status_code=400,
                detail="Models not trained. Please train models first."
            )
        
        # Analyze each transaction
        results = []
        fraud_count = 0
        risk_levels = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "MINIMAL": 0}
        
        for idx, row in df.iterrows():
            # Prepare transaction data
            transaction = {col: row[col] for col in required_cols}
            X = data_processor.prepare_single_transaction(transaction)
            
            # Get prediction
            proba = ensemble_model.predict_proba(X)[0]
            is_fraud = proba >= 0.5
            
            # Determine risk level
            if proba >= 0.8:
                risk = "CRITICAL"
            elif proba >= 0.6:
                risk = "HIGH"
            elif proba >= 0.4:
                risk = "MEDIUM"
            elif proba >= 0.2:
                risk = "LOW"
            else:
                risk = "MINIMAL"
            
            risk_levels[risk] += 1
            
            if is_fraud:
                fraud_count += 1
            
            # Get actual class if available
            actual = None
            if 'Class' in df.columns:
                actual = int(row['Class'])
            
            results.append({
                "index": int(idx),
                "amount": round(float(row['Amount']), 2),
                "predicted_fraud": bool(is_fraud),
                "fraud_probability": round(float(proba), 4),
                "risk_level": risk,
                "actual_class": actual,
                "correct": (is_fraud == (actual == 1)) if actual is not None else None
            })
        
        total = len(df)
        
        # Calculate summary statistics
        amounts = df['Amount'].values
        summary_stats = {
            "amount_mean": round(float(amounts.mean()), 2),
            "amount_std": round(float(amounts.std()), 2),
            "amount_min": round(float(amounts.min()), 2),
            "amount_max": round(float(amounts.max()), 2),
            "amount_median": round(float(np.median(amounts)), 2)
        }
        
        # If actual labels exist, calculate accuracy
        if 'Class' in df.columns:
            actual_fraud = int(df['Class'].sum())
            correct = sum(1 for r in results if r['correct'] == True)
            summary_stats["actual_fraud_count"] = actual_fraud
            summary_stats["detection_accuracy"] = round(correct / total * 100, 2)
        
        return AnalysisResult(
            total_transactions=total,
            fraud_detected=fraud_count,
            normal_detected=total - fraud_count,
            fraud_percentage=round(fraud_count / total * 100, 2),
            transactions=results,
            summary_stats=summary_stats,
            risk_distribution=risk_levels
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/sample")
async def get_sample_csv():
    """
    Get the path to the sample CSV file for testing.
    """
    sample_path = PROJECT_ROOT / "sample_transactions.csv"
    
    if sample_path.exists():
        # Read and return sample data
        df = pd.read_csv(sample_path)
        return {
            "filename": "sample_transactions.csv",
            "rows": len(df),
            "columns": list(df.columns),
            "fraud_transactions": int(df['Class'].sum()) if 'Class' in df.columns else None,
            "preview": df.head(5).to_dict('records')
        }
    else:
        raise HTTPException(status_code=404, detail="Sample file not found")


@router.post("/analyze-sample")
async def analyze_sample_csv():
    """
    Analyze the sample CSV file.
    """
    sample_path = PROJECT_ROOT / "sample_transactions.csv"
    
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample file not found")
    
    # Read sample file
    with open(sample_path, 'rb') as f:
        content = f.read()
    
    # Create a mock UploadFile-like object
    df = pd.read_csv(io.BytesIO(content))
    
    # Use same analysis logic
    required_cols = ['Amount'] + [f'V{i}' for i in range(1, 29)]
    
    status = ensemble_model.get_model_status()
    if not all(m.get('is_trained', False) for m in status.values()):
        raise HTTPException(
            status_code=400,
            detail="Models not trained. Please train models first."
        )
    
    results = []
    fraud_count = 0
    risk_levels = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "MINIMAL": 0}
    
    for idx, row in df.iterrows():
        transaction = {col: row[col] for col in required_cols}
        X = data_processor.prepare_single_transaction(transaction)
        
        proba = ensemble_model.predict_proba(X)[0]
        is_fraud = proba >= 0.5
        
        if proba >= 0.8:
            risk = "CRITICAL"
        elif proba >= 0.6:
            risk = "HIGH"
        elif proba >= 0.4:
            risk = "MEDIUM"
        elif proba >= 0.2:
            risk = "LOW"
        else:
            risk = "MINIMAL"
        
        risk_levels[risk] += 1
        
        if is_fraud:
            fraud_count += 1
        
        actual = int(row['Class']) if 'Class' in df.columns else None
        
        results.append({
            "index": int(idx),
            "amount": round(float(row['Amount']), 2),
            "predicted_fraud": bool(is_fraud),
            "fraud_probability": round(float(proba), 4),
            "risk_level": risk,
            "actual_class": actual,
            "correct": (is_fraud == (actual == 1)) if actual is not None else None
        })
    
    total = len(df)
    amounts = df['Amount'].values
    
    summary_stats = {
        "amount_mean": round(float(amounts.mean()), 2),
        "amount_std": round(float(amounts.std()), 2),
        "amount_min": round(float(amounts.min()), 2),
        "amount_max": round(float(amounts.max()), 2)
    }
    
    if 'Class' in df.columns:
        actual_fraud = int(df['Class'].sum())
        correct = sum(1 for r in results if r['correct'] == True)
        summary_stats["actual_fraud_count"] = actual_fraud
        summary_stats["detection_accuracy"] = round(correct / total * 100, 2)
    
    return {
        "total_transactions": total,
        "fraud_detected": fraud_count,
        "normal_detected": total - fraud_count,
        "fraud_percentage": round(fraud_count / total * 100, 2),
        "transactions": results,
        "summary_stats": summary_stats,
        "risk_distribution": risk_levels
    }
