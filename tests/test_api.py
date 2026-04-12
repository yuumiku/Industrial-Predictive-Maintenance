"""
API endpoint tests for the maintenance prediction service.
"""

import pytest
import json
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import os

# Add deployment to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'Deployement' / 'src'))

# Mock model and app setup
from unittest.mock import MagicMock, patch
import numpy as np


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    model = MagicMock()
    model.predict = MagicMock(return_value=np.array([1]))
    model.predict_proba = MagicMock(return_value=np.array([[0.1, 0.2, 0.3, 0.25, 0.15]]))
    return model


@pytest.fixture
def client(mock_model):
    """Create test client with mocked model."""
    with patch('mlflow.sklearn.load_model', return_value=mock_model):
        # Import after patching
        from Deployement.src.main import app
        return TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_check_success(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}
    
    def test_health_endpoint_json(self, client):
        """Test health endpoint returns valid JSON."""
        response = client.get("/health")
        data = response.json()
        assert isinstance(data, dict)
        assert "status" in data


class TestPredictEndpoint:
    """Test model prediction endpoint."""
    
    @pytest.fixture
    def valid_prediction_input(self):
        """Sample valid input for prediction."""
        return {
            "machine_id": "MACH_001",
            "Type": 1,
            "Air_temperature_K": 298.5,
            "Process_temperature_K": 308.7,
            "Rotational_speed_rpm": 1800,
            "Torque_Nm": 45.5,
            "Tool_wear_min": 100
        }
    
    def test_predict_success(self, client, valid_prediction_input):
        """Test successful prediction."""
        response = client.post("/predict", json=valid_prediction_input)
        assert response.status_code == 200
        data = response.json()
        
        assert "machine_id" in data
        assert "predicted_class" in data
        assert "class_probabilities" in data
        assert "confidence" in data
    
    def test_predict_returns_valid_class(self, client, valid_prediction_input):
        """Test prediction returns valid class number."""
        response = client.post("/predict", json=valid_prediction_input)
        data = response.json()
        
        predicted_class = data["predicted_class"]
        assert isinstance(predicted_class, int)
        assert 0 <= predicted_class < 5
    
    def test_predict_returns_probabilities(self, client, valid_prediction_input):
        """Test prediction returns probability distribution."""
        response = client.post("/predict", json=valid_prediction_input)
        data = response.json()
        
        probs = data["class_probabilities"]
        assert len(probs) == 5
        
        for key, prob in probs.items():
            assert isinstance(prob, float)
            assert 0 <= prob <= 1
    
    def test_predict_confidence_valid(self, client, valid_prediction_input):
        """Test confidence score is valid."""
        response = client.post("/predict", json=valid_prediction_input)
        data = response.json()
        
        confidence = data["confidence"]
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_predict_missing_machine_id(self, client):
        """Test prediction handles missing machine_id gracefully."""
        data = {
            "Type": 1,
            "Air_temperature_K": 298.5,
            "Process_temperature_K": 308.7,
            "Rotational_speed_rpm": 1800,
            "Torque_Nm": 45.5,
            "Tool_wear_min": 100
        }
        response = client.post("/predict", json=data)
        
        # Should still return 200 with None machine_id
        assert response.status_code == 200
        assert response.json()["machine_id"] is None
    
    def test_predict_returns_machine_id(self, client, valid_prediction_input):
        """Test prediction response includes input machine_id."""
        response = client.post("/predict", json=valid_prediction_input)
        data = response.json()
        
        assert data["machine_id"] == valid_prediction_input["machine_id"]


class TestEndpointIntegration:
    """Integration tests for API endpoints."""
    
    def test_sequential_predictions(self, client):
        """Test multiple sequential predictions."""
        inputs = [
            {"machine_id": f"MACH_{i:03d}", "Type": 1, "Air_temperature_K": 298 + i,
             "Process_temperature_K": 308 + i, "Rotational_speed_rpm": 1800 + i*10,
             "Torque_Nm": 45 + i, "Tool_wear_min": 100 + i*5}
            for i in range(3)
        ]
        
        for input_data in inputs:
            response = client.post("/predict", json=input_data)
            assert response.status_code == 200
            assert response.json()["machine_id"] == input_data["machine_id"]
    
    def test_api_endpoints_consistency(self, client):
        """Test that API returns consistent response format."""
        health = client.get("/health").json()
        
        pred_data = {"machine_id": "TEST", "Type": 1}
        predict = client.post("/predict", json=pred_data).json()
        
        assert isinstance(health, dict)
        assert isinstance(predict, dict)
        assert "predicted_class" in predict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
