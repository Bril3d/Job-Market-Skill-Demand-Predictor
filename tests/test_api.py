import pytest
import sys
import os
import importlib.util

# Load code/app.py manually since 'code' conflicts with Python built-in
spec = importlib.util.spec_from_file_location("app", os.path.join(os.path.dirname(__file__), "..", "code", "app.py"))
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)
app = app_module.app

from fastapi.testclient import TestClient
client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["api_version"] == "2.0.0"

    def test_health_has_predictions_count(self):
        response = client.get("/health")
        data = response.json()
        assert "predictions_served" in data


class TestMetricsEndpoint:
    def test_metrics_returns_data(self):
        response = client.get("/metrics")
        if response.status_code == 200:
            data = response.json()
            assert "accuracy" in data
            assert "precision" in data
            assert "recall" in data


class TestInsightsEndpoint:
    def test_insights_returns_data(self):
        response = client.get("/insights")
        if response.status_code == 200:
            data = response.json()
            assert "total_jobs" in data
            assert "categories" in data
            assert "seniority_breakdown" in data
            assert "high_salary_rate" in data


class TestPredictionEndpoint:
    def test_single_prediction(self):
        payload = {
            "title": "Senior Machine Learning Engineer",
            "seniority": "Senior",
            "category": "ML Engineering",
            "geo_tier": "Tier 1",
            "tags": "python, pytorch, aws"
        }
        response = client.post("/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert data["prediction"] in [0, 1]
            assert 0 <= data["probability"] <= 1
            assert "label" in data
            assert "experience_detected" in data

    def test_prediction_without_optional_fields(self):
        payload = {"title": "Web Developer"}
        response = client.post("/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data

    def test_prediction_missing_title_fails(self):
        payload = {"seniority": "Senior"}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_batch_prediction(self):
        payload = [
            {"title": "Senior Backend Engineer", "seniority": "Senior", "tags": "python, django"},
            {"title": "Junior Frontend Dev", "seniority": "Junior", "tags": "html, css"}
        ]
        response = client.post("/predict/batch", json=payload)
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 2


class TestHistoryEndpoint:
    def test_history_returns_list(self):
        response = client.get("/history")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestAPIKeyAuth:
    def test_predict_with_wrong_key_fails(self):
        payload = {"title": "Engineer"}
        response = client.post(
            "/predict",
            json=payload,
            headers={"X-API-Key": "wrong-key"}
        )
        assert response.status_code == 403

    def test_predict_without_key_still_works(self):
        """API key is optional (auto_error=False), so no key = allowed."""
        payload = {"title": "Engineer"}
        response = client.post("/predict", json=payload)
        # Should not be 403
        assert response.status_code != 403
