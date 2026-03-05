import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import re

# Initialize FastAPI
app = FastAPI(title="Job Salary Prediction API")

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
        MODEL = joblib.load("models/salary_model.joblib")
        TAG_BINARIZER = joblib.load("models/tag_binarizer.joblib")
        TITLE_TFIDF = joblib.load("models/title_tfidf.joblib")
        if os.path.exists("models/threshold.json"):
            with open("models/threshold.json", "r") as f:
                THRESHOLD = json.load(f)["threshold"]
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

# Models for Request/Response
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

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "api_version": "1.0.0"
    }

@app.get("/metrics")
def get_metrics():
    if not os.path.exists("data/evaluation_metrics.json"):
        raise HTTPException(status_code=404, detail="Metrics not found")
    with open("data/evaluation_metrics.json", "r") as f:
        return json.load(f)

@app.get("/insights")
def get_insights():
    """Returns basic insights from the processed data for the dashboard."""
    if not os.path.exists("data/processed_jobs.csv"):
        raise HTTPException(status_code=404, detail="Processed data not found")
    
    df = pd.read_csv("data/processed_jobs.csv")
    salary_counts = df["is_high_salary"].value_counts().to_dict()
    category_counts = df["category"].value_counts().to_dict()
    
    return {
        "total_jobs": len(df),
        "high_salary_count": salary_counts.get(1, 0),
        "standard_salary_count": salary_counts.get(0, 0),
        "categories": category_counts
    }

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
    input_df = pd.DataFrame([{
        "seniority": job_dict.get("seniority", "Mid-Level"),
        "category": job_dict.get("category", "Others"),
        "geo_tier": job_dict.get("geo_tier", "Tier 2"),
        "years_exp": extract_experience(job_dict.get("title", ""), tags_str)
    }])
    
    final_input = pd.concat([input_df, tag_df, title_tfidf_df], axis=1)

    # 4. Predict
    probability = MODEL.predict_proba(final_input)[0][1]
    prediction = 1 if probability >= THRESHOLD else 0
    
    return prediction, probability

@app.post("/predict", response_model=PredictionResponse)
def predict_single(job: JobInput):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        pred, prob = run_prediction(job.model_dump())
        return {
            "prediction": int(pred),
            "probability": float(prob),
            "threshold": float(THRESHOLD),
            "label": "High Salary ($120k+)" if pred == 1 else "Standard Salary (<$120k)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(jobs: List[JobInput]):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for job in jobs:
        try:
            pred, prob = run_prediction(job.model_dump())
            results.append({
                "prediction": int(pred),
                "probability": float(prob),
                "threshold": float(THRESHOLD),
                "label": "High Salary ($120k+)" if pred == 1 else "Standard Salary (<$120k)"
            })
        except:
            continue
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
