import joblib
import pandas as pd
import numpy as np
import os

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

    # 1. Process Tags (Multi-Hot)
    tags = [t.strip() for t in job_data.get("tags", "").split(",") if t.strip()]
    tag_encoded = tag_binarizer.transform([tags])
    tag_cols = [f"tag_{c}" for c in tag_binarizer.classes_]
    tag_df = pd.DataFrame(tag_encoded, columns=tag_cols)

    # 2. Process Title (TF-IDF)
    title_encoded = title_tfidf.transform([job_data.get("title", "")])
    title_tfidf_df = pd.DataFrame(title_encoded.toarray(), columns=[f"title_tfidf_{i}" for i in range(100)])

    # 3. Combine with Categorical
    input_df = pd.DataFrame([{
        "seniority": job_data.get("seniority", "Mid-Level"),
        "category": job_data.get("category", "Others"),
        "geo_tier": job_data.get("geo_tier", "Tier 2")
    }])
    
    final_input = pd.concat([input_df, tag_df, title_tfidf_df], axis=1)

    # 4. Predict
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    return prediction, probability

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
        pred, prob = predict_salary(job)
        label = "High Salary ($120k+)" if pred == 1 else "Standard Salary (<$120k)"
        print(f"Job: {job['title']}")
        print(f"Prediction: {label}")
        print(f"Confidence (High Salary): {prob:.2%}")
        print("-" * 30)
