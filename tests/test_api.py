import pytest
from fastapi.testclient import TestClient
from code.app import app
import os

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

# Bypass API Key for testing if needed or use the dev key
HEADERS = {"X-API-Key": "sg-dev-key-2026"}

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_metrics_endpoint(client):
    if os.path.exists("data/evaluation_metrics.json"):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "accuracy" in response.json()
    else:
        pytest.skip("Metrics file not found, skipping test")

def test_insights_endpoint(client):
    if os.path.exists("data/features_jobs.csv"):
        response = client.get("/insights")
        assert response.status_code == 200
        data = response.json()
        assert "total_jobs" in data
    else:
        pytest.skip("Processed data not found, skipping test")

class TestPredictionEndpoint:
    def test_single_prediction_high_demand(self, client):
        payload = {
            "title": "Senior AI Researcher",
            "seniority": "Senior",
            "category": "ML Engineering",
            "geo_tier": "Tier 1",
            "tags": ["python", "pytorch", "transformers"]
        }
        response = client.post("/predict", json=payload, headers=HEADERS)
        if response.status_code == 503:
            pytest.fail("Model not loaded (503 Service Unavailable)")
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data

    def test_single_prediction_low_demand(self, client):
        payload = {
            "title": "Data Entry Clerk",
            "tags": ["excel"]
        }
        response = client.post("/predict", json=payload, headers=HEADERS)
        assert response.status_code == 200

    def test_batch_prediction(self, client):
        payload = [
            {
                "title": "DevOps Engineer",
                "tags": ["aws"]
            }
        ]
        response = client.post("/predict/batch", json=payload, headers=HEADERS)
        assert response.status_code == 200

    def test_invalid_api_key(self, client):
        response = client.post("/predict", json={}, headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403

def test_history_endpoint(client):
    client.post("/predict", json={
        "title": "Software Engineer",
        "tags": ["python"]
    }, headers=HEADERS)
    
    response = client.get("/history")
    assert response.status_code == 200
    assert len(response.json()) > 0
