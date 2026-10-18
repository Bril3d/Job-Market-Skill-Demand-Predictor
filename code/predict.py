import joblib
import pandas as pd
import os
import re
import json

def extract_experience(title, tags=""):
    text = (str(title) + " " + str(tags)).lower()
    matches = re.findall(r"(\d+)\s*\+?\s*years?", text)
    if matches:
        return max([int(m) for m in matches])
    return 0

def predict_demand(job_data, model_path="models/demand_model.joblib"):
    """
    Predicts if a job has high skill demand based on input data.
    job_data: dict with keys ['seniority', 'category', 'geo_tier', 'tags', 'title']
    """
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None

    # Load end-to-end pipeline model
    model = joblib.load(model_path)
    
    threshold = 0.5  # Default
    if os.path.exists("models/threshold.json"):
        with open("models/threshold.json", "r") as f:
            threshold = json.load(f).get("threshold", 0.5)

    title = str(job_data.get("title", "")).strip()
    tags = job_data.get("tags", [])
    if isinstance(tags, list):
        tags_str = ", ".join(tags)
    else:
        tags_str = str(tags)
        
    years_exp = extract_experience(title, tags_str)

    # Input DataFrame matches the pipeline feature schema (no circular tag features)
    input_df = pd.DataFrame([{
        "title": title,
        "seniority": job_data.get("seniority", "Mid-Level"),
        "category": job_data.get("category", "Others"),
        "geo_tier": job_data.get("geo_tier", "Tier 2"),
        "years_exp": years_exp
    }])

    # Predict via end-to-end pipeline (ColumnTransformer handles text TF-IDF internally)
    probability = float(model.predict_proba(input_df)[0][1])
    prediction = 1 if probability >= threshold else 0

    return prediction, probability, threshold

if __name__ == "__main__":
    # Example Test Cases
    test_jobs = [
        {
            "title": "Senior Machine Learning Engineer",
            "seniority": "Senior",
            "category": "ML Engineering",
            "geo_tier": "Tier 1",
            "tags": ["python", "pytorch", "aws", "kubernetes"]
        },
        {
            "title": "Junior Web Developer",
            "seniority": "Junior",
            "category": "Frontend",
            "geo_tier": "Tier 2",
            "tags": ["html", "css", "javascript"]
        }
    ]

    print("--- Skill Demand Prediction Testing ---\n")
    for job in test_jobs:
        pred, prob, thresh = predict_demand(job)
        label = "High Demand Skillset" if pred == 1 else "Standard Demand Skillset"
        print(f"Job: {job['title']}")
        print(f"Prediction: {label}")
        print(f"Confidence: {prob:.2%}")
        print(f"Threshold Used: {thresh:.4f}")
        print("-" * 30)

