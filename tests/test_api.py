import pytest
from fastapi.testclient import TestClient
from code.app import app
import os

client = TestClient(app)

# Bypass API Key for testing if needed or use the dev key
HEADERS = {"X-API-Key": "sg-dev-key-2026"}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["api_version"] == "3.0.0"

def test_metrics_endpoint():
    # This test assumes modeling.py has been run and data/evaluation_metrics.json exists
    if os.path.exists("data/evaluation_metrics.json"):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "accuracy" in response.json()
        assert "f1_high" in response.json()
    else:
        pytest.skip("Metrics file not found, skipping test")

def test_insights_endpoint():
    if os.path.exists("data/features_jobs.csv"):
        response = client.get("/insights")
        assert response.status_code == 200
        data = response.json()
        assert "high_demand_rate" in data
        assert "categories" in data
        assert "total_jobs" in data
    else:
        pytest.skip("Processed data not found, skipping test")

class TestPredictionEndpoint:
    def test_single_prediction_high_demand(self):
        payload = {
            "title": "Senior AI Researcher",
            "seniority": "Senior",
            "category": "ML Engineering",
            "geo_tier": "Tier 1",
            "tags": ["python", "pytorch", "transformers", "research", "machine learning"]
        }
        response = client.post("/predict", json=payload, headers=HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "demand_score" in data
        assert "label" in data
        assert data["experience_detected"] >= 0

    def test_single_prediction_low_demand(self):
        payload = {
            "title": "Data Entry Clerk",
            "seniority": "Junior",
            "category": "Others",
            "geo_tier": "Tier 2",
            "tags": ["excel"]
        }
        response = client.post("/predict", json=payload, headers=HEADERS)
        assert response.status_code == 200
        assert response.json()["prediction"] in [0, 1]

    def test_batch_prediction(self):
        payload = [
            {
                "title": "DevOps Engineer",
                "tags": ["aws", "terraform", "docker"]
            },
            {
                "title": "Receptionist",
                "tags": ["typing"]
            }
        ]
        response = client.post("/predict/batch", json=payload, headers=HEADERS)
        assert response.status_code == 200
        assert len(response.json()) == 2
        assert "prediction" in response.json()[0]

    def test_invalid_api_key(self):
        response = client.post("/predict", json={}, headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403

def test_history_endpoint():
    # Trigger a prediction first
    client.post("/predict", json={
        "title": "Software Engineer",
        "tags": ["python"]
    }, headers=HEADERS)
    
    response = client.get("/history")
    assert response.status_code == 200
    assert len(response.json()) > 0
    assert "timestamp" in response.json()[0]
