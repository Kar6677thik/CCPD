# Intelligent Fraud Detection Learning Agent

An adaptive machine learning system for credit card fraud detection using Isolation Forest, Autoencoders, and Gradient Boosting models.

## Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 in your browser.

## Features

- **Multi-Model Detection**: Isolation Forest, Autoencoder, XGBoost
- **Incremental Learning**: Adapt to new fraud patterns
- **Real-time Detection**: Instant fraud scoring
- **Modern Dashboard**: Interactive visualizations
