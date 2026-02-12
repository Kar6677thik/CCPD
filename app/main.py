"""
Intelligent Fraud Detection Learning Agent
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.routers import detection, training, metrics, data, upload

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

app = FastAPI(
    title="Intelligent Fraud Detection Agent",
    description="Adaptive machine learning system for credit card fraud detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detection.router)
app.include_router(training.router)
app.include_router(metrics.router)
app.include_router(data.router)
app.include_router(upload.router)

# Mount static files
static_dir = PROJECT_ROOT / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """Serve the main UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "Intelligent Fraud Detection Agent API",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """API health check endpoint."""
    return {
        "status": "healthy",
        "service": "Fraud Detection Agent",
        "version": "1.0.0"
    }


@app.get("/api/info")
async def get_info():
    """Get API information."""
    return {
        "name": "Intelligent Fraud Detection Learning Agent",
        "description": "Adaptive ML system using Isolation Forest, Autoencoder, and XGBoost",
        "models": [
            {
                "name": "Isolation Forest",
                "type": "Unsupervised Anomaly Detection",
                "description": "Detects anomalies by isolating outliers"
            },
            {
                "name": "Autoencoder",
                "type": "Deep Learning",
                "description": "Learns normal patterns and detects anomalies via reconstruction error"
            },
            {
                "name": "XGBoost",
                "type": "Supervised Classification",
                "description": "Gradient boosting classifier for fraud prediction"
            }
        ],
        "endpoints": {
            "detection": "/api/detect",
            "training": "/api/train",
            "metrics": "/api/metrics",
            "data": "/api/data"
        }
    }
