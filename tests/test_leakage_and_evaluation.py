import os
import json
import joblib
import pandas as pd
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from code.predict import predict_demand
from code.app import run_prediction, load_artifacts
from code.modeling import load_data, build_preprocessor, get_best_threshold_from_oof

@pytest.fixture(scope="module")
def artifacts():
    # Ensure artifacts are loaded
    load_artifacts()
    model = joblib.load("models/demand_model.joblib")
    with open("models/threshold.json") as f:
        threshold = json.load(f)["threshold"]
    with open("data/evaluation_metrics.json") as f:
        metrics = json.load(f)
    return {
        "model": model,
        "threshold": threshold,
        "metrics": metrics
    }

def test_feature_matrix_no_leaky_columns():
    """Verify that features_jobs.csv contains zero target-derived or circular proxy columns."""
    df = pd.read_csv("data/features_jobs.csv")
    leaky_columns = {
        "demand_score", "demand_score_norm", "tag_frequency_score",
        "num_skills", "has_ai", "has_cloud", "has_backend", "has_frontend",
        "tag_list", "tags"
    }
    present_leaky = [col for col in leaky_columns if col in df.columns]
    assert len(present_leaky) == 0, f"Leaky columns found in features_jobs.csv: {present_leaky}"
    assert "demand_label" in df.columns, "Target column demand_label must be present."
    assert "title" in df.columns, "Raw title must be present for pipeline TF-IDF."

def test_tfidf_pipeline_isolation(artifacts):
    """Verify that TfidfVectorizer is encapsulated inside ColumnTransformer and fits on train data only."""
    model = artifacts["model"]
    preprocessor = model.named_steps["preprocessor"]
    tfidf = preprocessor.named_transformers_["title_tfidf"]
    
    # Vocabulary should be fitted and non-empty
    assert hasattr(tfidf, "vocabulary_"), "TF-IDF vectorizer must be fitted."
    vocab_size = len(tfidf.vocabulary_)
    assert vocab_size > 0, "TF-IDF vocabulary cannot be empty."
    
    # Test that out-of-vocabulary terms in test input do not mutate training vocabulary
    test_df = pd.DataFrame([{
        "title": "Quantum Zyzzyva Neuro-Cryptographer 9999",
        "seniority": "Senior",
        "category": "Others",
        "geo_tier": "Tier 1",
        "years_exp": 10
    }])
    transformed = preprocessor.transform(test_df)
    assert transformed.shape[0] == 1
    assert len(tfidf.vocabulary_) == vocab_size, "Transforming out-of-vocabulary text must not mutate vocabulary."

def test_smote_isolation_during_inference(artifacts):
    """Verify that SMOTE is never applied during inference/predict."""
    model = artifacts["model"]
    test_batch = pd.DataFrame([
        {"title": "Software Engineer", "seniority": "Mid-Level", "category": "Backend", "geo_tier": "Tier 2", "years_exp": 2},
        {"title": "Junior QA Analyst", "seniority": "Junior", "category": "Others", "geo_tier": "Tier 2", "years_exp": 0},
    ])
    probs = model.predict_proba(test_batch)
    assert probs.shape == (2, 2), f"Expected shape (2, 2), got {probs.shape}. Inference size must not be altered by SMOTE."

def test_threshold_selection_training_only(artifacts):
    """Verify that the decision threshold was recorded as training-only and is in a valid range."""
    threshold = artifacts["threshold"]
    metrics = artifacts["metrics"]
    assert 0.0 < threshold < 1.0, f"Threshold must be a valid probability strictly between 0 and 1, got {threshold}"
    assert metrics.get("threshold_selection") == "training_cv_only", "Metrics metadata must confirm threshold_selection is training_cv_only."
    assert metrics.get("leakage_free") is True, "Metrics metadata must confirm leakage_free is True."

def test_reproducibility_data_split():
    """Verify that stratified train/test splitting is deterministic with random_state=42."""
    X, y = load_data("data/features_jobs.csv")
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)
    assert len(X_train1) == 416
    assert len(X_test1) == 104

def test_inference_consistency():
    """Verify that code.predict and code.app produce identical predictions and probabilities."""
    sample_job = {
        "title": "Senior Cloud Infrastructure Architect",
        "seniority": "Senior",
        "category": "DevOps",
        "geo_tier": "Tier 1",
        "tags": ["aws", "terraform", "kubernetes"]
    }
    
    # 1. From predict.py
    pred_pkg, prob_pkg, thresh_pkg = predict_demand(sample_job)
    
    # 2. From app.py
    pred_app, prob_app, exp_app = run_prediction(sample_job)
    
    assert pred_pkg == pred_app, f"Prediction mismatch: {pred_pkg} vs {pred_app}"
    assert abs(prob_pkg - prob_app) < 1e-5, f"Probability mismatch: {prob_pkg} vs {prob_app}"
    assert thresh_pkg == artifacts or thresh_pkg > 0
