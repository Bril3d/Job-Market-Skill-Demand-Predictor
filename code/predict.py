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

def predict_salary(job_data, model_path="models/salary_model.joblib"):
    """
    Predicts if a job is high-salary ($120k+) based on input data.
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

    # 1. Process Tags (Multi-Hot)
    tags_str = job_data.get("tags", "")
    tags = [t.strip() for t in tags_str.split(",") if t.strip()]
    tag_encoded = tag_binarizer.transform([tags])
    tag_cols = [f"tag_{c}" for c in tag_binarizer.classes_]
    tag_df = pd.DataFrame(tag_encoded, columns=tag_cols)

    # 2. Process Title (TF-IDF)
    title_encoded = title_tfidf.transform([job_data.get("title", "")])
    title_tfidf_df = pd.DataFrame(title_encoded.toarray(), columns=[f"title_tfidf_{i}" for i in range(200)])

    # 3. Combine with Categorical and Experience
    input_df = pd.DataFrame([{
        "seniority": job_data.get("seniority", "Mid-Level"),
        "category": job_data.get("category", "Others"),
        "geo_tier": job_data.get("geo_tier", "Tier 2"),
        "years_exp": extract_experience(job_data.get("title", ""), tags_str)
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
            "tags": "python, pytorch, aws, kubernetes"
        },
        {
            "title": "Junior Web Developer",
            "seniority": "Junior",
            "category": "Frontend",
            "geo_tier": "Tier 2",
            "tags": "html, css, javascript"
        }
    ]

    print("--- Salary Prediction Testing ---\n")
    for job in test_jobs:
        pred, prob, thresh = predict_salary(job)
        label = "High Salary ($120k+)" if pred == 1 else "Standard Salary (<$120k)"
        print(f"Job: {job['title']}")
        print(f"Prediction: {label}")
        print(f"Confidence (High Salary): {prob:.2%}")
        print(f"Threshold Used: {thresh:.4f}")
        print("-" * 30)
