"""
API Testing Module (Step 4 - Automated Testing).

This module contains unit tests for the FastAPI application using the `pytest` framework 
and FastAPI's `TestClient`. It ensures that our endpoints (/health, /features, /predict) 
function correctly and safely handle unexpected inputs.

Team Instructions:
- Mandatory Check: Run these tests locally before pushing any changes to the repository.
- Usage: Execute `$ pytest tests/test_api.py` from the project root.
- CI/CD Integration: These tests are configured to run automatically in our GitHub Actions pipeline.
  If these fail, the deployment will be blocked.
"""

import sys
import os
import pytest  # Importing pytest to create test fixtures

# Append the project's root directory to the system path 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from fastapi.testclient import TestClient
from api.main import app

# FIX: Using a pytest fixture to initialize TestClient as a context manager.
# This ensures that FastAPI's 'lifespan' (startup events) runs and loads the model before testing.
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """
    Test Case 1: Verify that the health check endpoint is working correctly.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_features_endpoint(client):
    """
    Test Case 2: Verify that the /features endpoint returns the correct structure.
    """
    response = client.get("/features")
    assert response.status_code == 200
    assert "feature_count" in response.json()

def test_predict_validation(client):
    """
    Test Case 3: Verify that the /predict endpoint correctly handles invalid or incomplete data.
    """
    # Providing incomplete dummy data (the model expects 81 features, we are sending 1)
    fake_data = {"features": {"wrong_feature": 10.5}} 
    
    response = client.post("/predict", json=fake_data)
    
    # Because the data is incomplete, the API should throw a validation error (400 Bad Request).
    assert response.status_code == 400