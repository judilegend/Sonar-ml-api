import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

# 
#  Test Configuration & Fixtures
# 

# Initialize test client (no live server needed)
client = TestClient(app)

# Mock model bundle - simulates a trained SVM without requiring actual .pkl file
MOCK_BUNDLE = {
    "model": MagicMock(),
    "scaler": MagicMock(),
    "label_map": {0: "M", 1: "R"},
    "best_params": {"kernel": "rbf", "C": 10, "gamma": "scale"},
    "cv_score": 0.8762,
}

# Configure mock scaler to pass features unchanged (simulating preprocessing)
MOCK_BUNDLE["scaler"].transform = lambda x: x

# Configure mock model to predict class 0 (Mine) with 85% confidence
MOCK_BUNDLE["model"].predict = lambda x: np.array([0])
MOCK_BUNDLE["model"].predict_proba = lambda x: np.array([[0.85, 0.15]])

# Valid test input: 60 normalized sonar energy values
VALID_FEATURES = [0.02] * 60


# 
#  Test Suite: Health Check Endpoint
# 

class TestHealthCheck:
    """
    Tests for GET /health endpoint.
    Verifies API availability and model loading status.
    """

    def test_health_returns_200(self):
        """Health check endpoint should always return 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_ok_status_when_model_loaded(self):
        """API status should be 'ok' when model is available."""
        with patch("app.main.get_bundle", return_value=MOCK_BUNDLE):
            response = client.get("/health")
            data = response.json()
            assert data["status"] == "ok"
            assert data["model_loaded"] is True

    def test_health_degraded_status_when_model_missing(self):
        """API status should be 'degraded' when model fails to load."""
        with patch("app.main.get_bundle", side_effect=RuntimeError("Model unavailable")):
            response = client.get("/health")
            data = response.json()
            assert data["status"] == "degraded"
            assert data["model_loaded"] is False


# 
#  Test Suite: Model Metadata Endpoint
# 

class TestModelInfo:
    """
    Tests for GET /model/info endpoint.
    Verifies retrieval of model configuration and metadata.
    """

    def test_model_info_returns_200(self):
        """Model info endpoint should return 200 when model is loaded."""
        with patch("app.main.get_bundle", return_value=MOCK_BUNDLE):
            response = client.get("/model/info")
            assert response.status_code == 200

    def test_model_info_contains_expected_fields(self):
        """Response should contain model configuration and metadata."""
        with patch("app.main.get_bundle", return_value=MOCK_BUNDLE):
            data = client.get("/model/info").json()
            assert "best_params" in data
            assert "cv_score" in data
            assert "model_version" in data
            assert data["status"] == "loaded"

    def test_model_info_returns_503_when_model_not_loaded(self):
        """Should return 503 Service Unavailable when model is not loaded."""
        with patch("app.main.get_bundle", side_effect=RuntimeError):
            response = client.get("/model/info")
            assert response.status_code == 503


# 
#  Test Suite: Single Prediction Endpoint
# 

class TestPredictSingle:
    """
    Tests for POST /predict endpoint.
    Verifies single sample inference and input validation.
    """

    def test_predict_returns_200_with_valid_input(self):
        """Prediction endpoint should return 200 with valid 60-feature input."""
        with patch("app.main.predict", return_value={
            "prediction": "M",
            "label_full": "Mine",
            "probability_mine": 0.85,
            "probability_rock": 0.15,
            "confidence": 0.85,
            "model_version": "1.0.0"
        }):
            response = client.post("/predict", json={"features": VALID_FEATURES})
            assert response.status_code == 200

    def test_predict_response_structure(self):
        """Response should contain all required prediction fields."""
        with patch("app.main.predict", return_value={
            "prediction": "M",
            "label_full": "Mine",
            "probability_mine": 0.85,
            "probability_rock": 0.15,
            "confidence": 0.85,
            "model_version": "1.0.0"
        }):
            data = client.post("/predict", json={"features": VALID_FEATURES}).json()
            assert data["prediction"] in ["M", "R"]
            assert data["label_full"] in ["Mine", "Rock"]
            assert "probability_mine" in data
            assert "probability_rock" in data
            assert "confidence" in data
            assert "timestamp" in data
            assert "model_version" in data

    def test_predict_rejects_fewer_than_60_features(self):
        """Request with < 60 features should return 422 validation error."""
        response = client.post("/predict", json={"features": [0.1] * 30})
        assert response.status_code == 422

    def test_predict_rejects_more_than_60_features(self):
        """Request with > 60 features should return 422 validation error."""
        response = client.post("/predict", json={"features": [0.1] * 70})
        assert response.status_code == 422

    def test_predict_rejects_out_of_range_values(self):
        """Request with feature values outside [0.0, 1.0] should return 422."""
        invalid_features = [0.5] * 59 + [2.0]  # 2.0 exceeds max of 1.0
        response = client.post("/predict", json={"features": invalid_features})
        assert response.status_code == 422

    def test_predict_rejects_negative_values(self):
        """Request with negative feature values should return 422."""
        invalid_features = [-0.1] + [0.5] * 59  # -0.1 below min of 0.0
        response = client.post("/predict", json={"features": invalid_features})
        assert response.status_code == 422


# 
#  Test Suite: Batch Prediction Endpoint
# 

class TestPredictBatch:
    """
    Tests for POST /predict/batch endpoint.
    Verifies batch inference, response structure, and limits.
    """

    def test_batch_returns_list_of_predictions(self):
        """Batch endpoint should return list with one prediction per input."""
        mock_result = {
            "prediction": "R",
            "label_full": "Rock",
            "probability_mine": 0.2,
            "probability_rock": 0.8,
            "confidence": 0.8,
            "model_version": "1.0.0"
        }
        with patch("app.main.predict", return_value=mock_result):
            payload = [{"features": VALID_FEATURES}] * 3
            data = client.post("/predict/batch", json=payload).json()
            assert isinstance(data, list)
            assert len(data) == 3

    def test_batch_rejects_exceeding_size_limit(self):
        """Batch request exceeding 100 signals should return 400."""
        payload = [{"features": VALID_FEATURES}] * 101
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "100" in detail or "maximum" in detail.lower()

    def test_batch_accepts_100_signals(self):
        """Batch request with exactly 100 signals should be accepted."""
        mock_result = {
            "prediction": "M",
            "label_full": "Mine",
            "probability_mine": 0.75,
            "probability_rock": 0.25,
            "confidence": 0.75,
            "model_version": "1.0.0"
        }
        with patch("app.main.predict", return_value=mock_result):
            payload = [{"features": VALID_FEATURES}] * 100
            response = client.post("/predict/batch", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 100

    def test_batch_validates_each_input(self):
        """Batch should validate each signal's feature count."""
        invalid_payload = [
            {"features": VALID_FEATURES},
            {"features": [0.1] * 30},  # Invalid: only 30 features
        ]
        response = client.post("/predict/batch", json=invalid_payload)
        assert response.status_code == 422

    def test_batch_with_mixed_valid_predictions(self):
        """Batch predictions should handle mixed outcomes (Mine/Rock)."""
        def predict_side_effect(features):
            # Alternate between Mine and Rock predictions
            idx = id(features) % 2
            return {
                "prediction": ["M", "R"][idx],
                "label_full": ["Mine", "Rock"][idx],
                "probability_mine": [0.8, 0.2][idx],
                "probability_rock": [0.2, 0.8][idx],
                "confidence": [0.8, 0.8][idx],
                "model_version": "1.0.0"
            }

        with patch("app.main.predict", side_effect=predict_side_effect):
            payload = [{"features": VALID_FEATURES}] * 5
            response = client.post("/predict/batch", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 5
            # All predictions should have required fields
            for pred in data:
                assert pred["prediction"] in ["M", "R"]
                assert "timestamp" in pred


# 
#  Integration Tests
# 

class TestIntegration:
    """
    High-level integration tests combining multiple endpoints.
    """

    def test_health_check_then_predict_flow(self):
        """Typical user flow: health check → model info → prediction."""
        with patch("app.main.get_bundle", return_value=MOCK_BUNDLE):
            # Step 1: Check health
            health_response = client.get("/health")
            assert health_response.status_code == 200
            assert health_response.json()["model_loaded"] is True

            # Step 2: Get model info
            info_response = client.get("/model/info")
            assert info_response.status_code == 200

            # Step 3: Make prediction
            with patch("app.main.predict", return_value={
                "prediction": "M",
                "label_full": "Mine",
                "probability_mine": 0.85,
                "probability_rock": 0.15,
                "confidence": 0.85,
                "model_version": "1.0.0"
            }):
                pred_response = client.post("/predict", json={"features": VALID_FEATURES})
                assert pred_response.status_code == 200

