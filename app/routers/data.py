"""
Data Router
API endpoints for dataset information and statistics.
"""

from fastapi import APIRouter, HTTPException

from app.ml.data_processor import data_processor

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/stats")
async def get_data_stats():
    """
    Get comprehensive statistics about the dataset.
    """
    try:
        stats = data_processor.get_data_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/distribution")
async def get_class_distribution():
    """
    Get class distribution (fraud vs normal).
    """
    try:
        df = data_processor.load_data()
        
        fraud_count = int(df[df['Class'] == 1].shape[0])
        normal_count = int(df[df['Class'] == 0].shape[0])
        total = len(df)
        
        return {
            "distribution": {
                "fraud": {
                    "count": fraud_count,
                    "percentage": round(fraud_count / total * 100, 4)
                },
                "normal": {
                    "count": normal_count,
                    "percentage": round(normal_count / total * 100, 4)
                }
            },
            "total_transactions": total,
            "imbalance_ratio": round(normal_count / fraud_count, 2) if fraud_count > 0 else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample")
async def get_sample_transactions(n: int = 10, fraud_only: bool = False):
    """
    Get sample transactions from the dataset.
    """
    try:
        samples = data_processor.get_sample_transactions(n, fraud_only)
        return {
            "samples": samples,
            "count": len(samples),
            "fraud_only": fraud_only
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features")
async def get_feature_info():
    """
    Get information about dataset features.
    """
    try:
        df = data_processor.load_data()
        
        feature_stats = {}
        for col in df.columns:
            if col not in ['Time', 'Class']:
                feature_stats[col] = {
                    "mean": round(df[col].mean(), 4),
                    "std": round(df[col].std(), 4),
                    "min": round(df[col].min(), 4),
                    "max": round(df[col].max(), 4)
                }
        
        return {
            "feature_count": len(feature_stats),
            "features": feature_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/time-distribution")
async def get_time_distribution(bins: int = 24):
    """
    Get transaction distribution over time.
    """
    try:
        df = data_processor.load_data()
        
        # Time is in seconds, convert to hours
        df['Hour'] = (df['Time'] / 3600).astype(int) % 24
        
        distribution = df.groupby('Hour').agg({
            'Class': ['count', 'sum']
        }).reset_index()
        
        distribution.columns = ['hour', 'total', 'fraud']
        
        result = []
        for _, row in distribution.iterrows():
            result.append({
                "hour": int(row['hour']),
                "total": int(row['total']),
                "fraud": int(row['fraud']),
                "fraud_rate": round(row['fraud'] / row['total'] * 100, 4) if row['total'] > 0 else 0
            })
        
        return {"time_distribution": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/amount-distribution")
async def get_amount_distribution():
    """
    Get transaction amount distribution.
    """
    try:
        df = data_processor.load_data()
        
        # Define amount ranges
        ranges = [
            (0, 10, "0-10"),
            (10, 50, "10-50"),
            (50, 100, "50-100"),
            (100, 500, "100-500"),
            (500, 1000, "500-1K"),
            (1000, 5000, "1K-5K"),
            (5000, float('inf'), "5K+")
        ]
        
        distribution = []
        for low, high, label in ranges:
            mask = (df['Amount'] >= low) & (df['Amount'] < high)
            subset = df[mask]
            
            distribution.append({
                "range": label,
                "count": int(len(subset)),
                "fraud_count": int(subset['Class'].sum()),
                "fraud_rate": round(subset['Class'].mean() * 100, 4) if len(subset) > 0 else 0
            })
        
        return {"amount_distribution": distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
