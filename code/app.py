import os
import json
import joblib
import pandas as pd
import logging
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SalaryGenius")

# Environment configuration
MODEL_PATH = os.getenv("MODEL_PATH", "models/salary_model.joblib")
TAG_BINARIZER_PATH = os.getenv("TAG_BINARIZER_PATH", "models/tag_binarizer.joblib")
TFIDF_PATH = os.getenv("TFIDF_PATH", "models/title_tfidf.joblib")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/threshold.json")
METRICS_PATH = os.getenv("METRICS_PATH", "data/evaluation_metrics.json")
PROCESSED_DATA_PATH = os.getenv("PROCESSED_DATA_PATH", "data/processed_jobs.csv")
API_KEY = os.getenv("API_KEY", "sg-dev-key-2026")

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key and api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Initialize FastAPI
app = FastAPI(
    title="SalaryGenius AI API",
    description="AI-powered job salary prediction engine",
    version="2.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model artifacts
MODEL = None
TAG_BINARIZER = None
TITLE_TFIDF = None
THRESHOLD = 0.5
PREDICTION_HISTORY = []

def extract_experience(title, tags):
    text = (str(title) + " " + str(tags)).lower()
    matches = re.findall(r"(\d+)\s*\+?\s*years?", text)
    if matches:
        return max([int(m) for m in matches])
    return 0

@app.on_event("startup")
def load_artifacts():
    global MODEL, TAG_BINARIZER, TITLE_TFIDF, THRESHOLD
    try:
        MODEL = joblib.load(MODEL_PATH)
        TAG_BINARIZER = joblib.load(TAG_BINARIZER_PATH)
        TITLE_TFIDF = joblib.load(TFIDF_PATH)
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, "r") as f:
                THRESHOLD = json.load(f)["threshold"]
        logger.info("Model artifacts loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")

# Request / Response Models
class JobInput(BaseModel):
    title: str
    seniority: str = "Mid-Level"
    category: str = "Others"
    geo_tier: str = "Tier 2"
    tags: Optional[str] = ""

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
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
        "api_version": "2.0.0",
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
    salary_counts = df["is_high_salary"].value_counts().to_dict()
    category_counts = df["category"].value_counts().to_dict()
    seniority_counts = df["seniority"].value_counts().to_dict()
    
    return {
        "total_jobs": len(df),
        "high_salary_count": salary_counts.get(1, 0),
        "standard_salary_count": salary_counts.get(0, 0),
        "high_salary_rate": round(salary_counts.get(1, 0) / max(len(df), 1) * 100, 1),
        "categories": category_counts,
        "seniority_breakdown": seniority_counts,
        "predictions_served": len(PREDICTION_HISTORY)
    }

@app.get("/history")
def get_prediction_history():
    """Returns the last 50 predictions for dashboard analytics."""
    return PREDICTION_HISTORY[-50:]

def run_prediction(job_dict):
    # 1. Process Tags
    tags_str = job_dict.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    tag_encoded = TAG_BINARIZER.transform([tags])
    tag_cols = [f"tag_{c}" for c in TAG_BINARIZER.classes_]
    tag_df = pd.DataFrame(tag_encoded, columns=tag_cols)

    # 2. Process Title
    title_encoded = TITLE_TFIDF.transform([job_dict.get("title", "")])
    title_tfidf_df = pd.DataFrame(title_encoded.toarray(), columns=[f"title_tfidf_{i}" for i in range(200)])

    # 3. Combine with Categorical and Experience
    years_exp = extract_experience(job_dict.get("title", ""), tags_str)
    input_df = pd.DataFrame([{
        "seniority": job_dict.get("seniority", "Mid-Level"),
        "category": job_dict.get("category", "Others"),
        "geo_tier": job_dict.get("geo_tier", "Tier 2"),
        "years_exp": years_exp
    }])
    
    final_input = pd.concat([input_df, tag_df, title_tfidf_df], axis=1)

    # 4. Predict
    probability = MODEL.predict_proba(final_input)[0][1]
    prediction = 1 if probability >= THRESHOLD else 0
    
    return prediction, probability, years_exp

@app.post("/predict", response_model=PredictionResponse)
def predict_single(job: JobInput, api_key: str = Depends(verify_api_key)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        pred, prob, years_exp = run_prediction(job.model_dump())
        result = {
            "prediction": int(pred),
            "probability": float(prob),
            "threshold": float(THRESHOLD),
            "label": "High Salary ($120k+)" if pred == 1 else "Standard Salary (<$120k)",
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
                "threshold": float(THRESHOLD),
                "label": "High Salary ($120k+)" if pred == 1 else "Standard Salary (<$120k)",
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
