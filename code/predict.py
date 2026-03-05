import joblib
import pandas as pd
import numpy as np
import os
import re
import json

def extract_experience(title, tags):
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

    # Load artifacts
    model = joblib.load(model_path)
    tag_binarizer = joblib.load("models/tag_binarizer.joblib")
    title_tfidf = joblib.load("models/title_tfidf.joblib")
    
    threshold = 0.5 # Default
    if os.path.exists("models/threshold.json"):
        with open("models/threshold.json", "r") as f:
            threshold = json.load(f)["threshold"]

    # 1. Process Tags
    tags = job_data.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    
    tags_str = ", ".join(tags)
    tag_encoded = tag_binarizer.transform([tags])
    tag_cols = [f"tag_{c}" for c in tag_binarizer.classes_]
    tag_df = pd.DataFrame(tag_encoded, columns=tag_cols)

    # 2. Process Title
    title = job_data.get("title", "")
    title_encoded = title_tfidf.transform([title])
    n_tfidf = len(title_tfidf.get_feature_names_out())
    title_tfidf_df = pd.DataFrame(title_encoded.toarray(), columns=[f"title_tfidf_{i}" for i in range(n_tfidf)])

    # 3. Engineered Features
    years_exp = extract_experience(title, tags_str)
    
    def check_keywords(tag_list, keywords):
        return 1 if any(kw in " ".join(tag_list).lower() for kw in keywords) else 0

    num_skills = len(tags)
    has_ai = check_keywords(tags, ["ai", "machine learning", "pytorch", "tensorflow", "nlp", "vision"])
    has_cloud = check_keywords(tags, ["aws", "azure", "gcp", "cloud", "docker", "kubernetes"])
    has_backend = check_keywords(tags, ["backend", "python", "java", "django", "fastapi", "node", "api"])
    has_frontend = check_keywords(tags, ["frontend", "react", "vue", "angular", "javascript", "typescript", "ui", "ux"])
    tag_frequency_score = min(num_skills / 10.0, 1.0) 

    input_df = pd.DataFrame([{
        "seniority": job_data.get("seniority", "Mid-Level"),
        "category": job_data.get("category", "Others"),
        "geo_tier": job_data.get("geo_tier", "Tier 2"),
        "years_exp": years_exp,
        "num_skills": num_skills,
        "has_ai": has_ai,
        "has_cloud": has_cloud,
        "has_backend": has_backend,
        "has_frontend": has_frontend,
        "tag_frequency_score": tag_frequency_score
    }])
    
    final_input = pd.concat([input_df, tag_df, title_tfidf_df], axis=1)

    # 4. Predict 
    probability = model.predict_proba(final_input)[0][1]
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
