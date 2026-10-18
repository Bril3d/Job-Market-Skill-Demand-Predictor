import os
import json
import joblib
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime
from contextlib import asynccontextmanager
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SkillDemand")

# Environment configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/demand_model.joblib")
TFIDF_PATH = os.getenv("TFIDF_PATH", "models/title_tfidf.joblib")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/threshold.json")
METRICS_PATH = os.getenv("METRICS_PATH", "data/evaluation_metrics.json")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/features_jobs.csv")
API_KEY = os.getenv("API_KEY", "sg-dev-key-2026")

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key and api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Global variables for model artifacts
MODEL = None
TITLE_TFIDF = None
THRESHOLD = 0.5
PREDICTION_HISTORY = []

def extract_experience(title, tags=""):
    text = (str(title) + " " + str(tags)).lower()
    matches = re.findall(r"(\d+)\s*\+?\s*years?", text)
    if matches:
        return max([int(m) for m in matches])
    return 0

def load_artifacts():
    global MODEL, TITLE_TFIDF, THRESHOLD
    print(f"DEBUG: Attempting to load artifacts. Checking {MODEL_PATH}")
    try:
        if os.path.exists(MODEL_PATH):
            print(f"DEBUG: File exists: {MODEL_PATH}")
            MODEL = joblib.load(MODEL_PATH)
            if os.path.exists(TFIDF_PATH):
                TITLE_TFIDF = joblib.load(TFIDF_PATH)
            if os.path.exists(THRESHOLD_PATH):
                with open(THRESHOLD_PATH, "r") as f:
                    THRESHOLD = json.load(f)["threshold"]
            logger.info(f"Model artifacts loaded successfully from {MODEL_PATH}")
            print("DEBUG: Artifacts loaded successfully")
        else:
            logger.warning(f"Model not found at {MODEL_PATH}. Prediction endpoints will fail.")
            print(f"DEBUG: Model file NOT FOUND at {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        print(f"DEBUG: EXCEPTION during load_artifacts: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model artifacts on start
    print(f"DEBUG: Starting lifespan. MODEL_PATH={MODEL_PATH}")
    load_artifacts()
    yield

# Initialize FastAPI
app = FastAPI(
    title="Skill Demand Predictor API",
    description="Predicts job market skill demand using ensemble ML models",
    version="3.0.0",
    lifespan=lifespan
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Models
class JobInput(BaseModel):
    title: str
    seniority: str = "Mid-Level"
    category: str = "Others"
    geo_tier: str = "Tier 2"
    tags: List[str] = []

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    demand_score: float
    threshold: float
    label: str
    experience_detected: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str
    predictions_served: int

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "api_version": "3.0.0",
        "predictions_served": len(PREDICTION_HISTORY)
    }

@app.get("/metrics")
def get_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(status_code=404, detail="Metrics not found")
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

@app.get("/insights")
def get_insights():
    """Returns enriched insights from the processed data for the dashboard."""
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise HTTPException(status_code=404, detail="Processed data not found")
    
    df = pd.read_csv(PROCESSED_DATA_PATH)
    demand_counts = df["demand_label"].value_counts().to_dict()
    category_counts = df["category"].value_counts().to_dict()
    seniority_counts = df["seniority"].value_counts().to_dict()
    
    return {
        "total_jobs": len(df),
        "high_demand_count": demand_counts.get(1, 0),
        "low_demand_count": demand_counts.get(0, 0),
        "high_demand_rate": round(demand_counts.get(1, 0) / max(len(df), 1) * 100, 1),
        "categories": category_counts,
        "seniority_breakdown": seniority_counts,
        "predictions_served": len(PREDICTION_HISTORY)
    }

@app.get("/history")
def get_prediction_history():
    """Returns the last 50 predictions for dashboard analytics."""
    return PREDICTION_HISTORY[-50:]

def run_prediction(job_dict):
    # Process Tags for optional experience extraction (no circular shortcut features used)
    tags = job_dict.get("tags", [])
    if isinstance(tags, list):
        tags_str = ", ".join(tags)
    else:
        tags_str = str(tags)

    title = str(job_dict.get("title", "")).strip()
    years_exp = extract_experience(title, tags_str)

    # Input DataFrame matches the pipeline feature schema (title, seniority, category, geo_tier, years_exp)
    input_df = pd.DataFrame([{
        "title": title,
        "seniority": job_dict.get("seniority", "Mid-Level"),
        "category": job_dict.get("category", "Others"),
        "geo_tier": job_dict.get("geo_tier", "Tier 2"),
        "years_exp": years_exp
    }])

    # Predict via end-to-end pipeline (ColumnTransformer handles text TF-IDF internally)
    probability = float(MODEL.predict_proba(input_df)[0][1])
    prediction = 1 if probability >= THRESHOLD else 0
    
    return prediction, probability, years_exp

@app.post("/predict", response_model=PredictionResponse)
def predict_single(job: JobInput, api_key: str = Depends(verify_api_key)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Received prediction request: {job}")
        pred, prob, years_exp = run_prediction(job.model_dump())
        result = {
            "prediction": int(pred),
            "probability": float(prob),
            "demand_score": float(prob),  # Use probability as demand indicator
            "threshold": float(THRESHOLD),
            "label": "High Demand Skillset" if pred == 1 else "Standard Demand Skillset",
            "experience_detected": years_exp
        }
        # Log prediction
        PREDICTION_HISTORY.append({
            **result,
            "title": job.title,
            "timestamp": datetime.now().isoformat()
        })
        return result
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(jobs: List[JobInput], api_key: str = Depends(verify_api_key)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for job in jobs:
        try:
            pred, prob, years_exp = run_prediction(job.model_dump())
            results.append({
                "prediction": int(pred),
                "probability": float(prob),
                "demand_score": float(prob),
                "threshold": float(THRESHOLD),
                "label": "High Demand Skillset" if pred == 1 else "Standard Demand Skillset",
                "experience_detected": years_exp
            })
        except Exception as e:
            logger.warning(f"Skipped job: {e}")
            continue
    return results

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
