import os
from fastapi.testclient import TestClient
from unittest.mock import patch

# Patch mlflow before importing main
with patch("main.mlflow.set_experiment") as mock_set_experiment, \
     patch("main.mlflow.start_run") as mock_start_run:

    mock_set_experiment.return_value = None
    mock_start_run.return_value.__enter__.return_value = None

    from main import app  

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_empty_query():
    response = client.post("/answer", json={"question": ""})
    assert response.status_code == 400
    assert "Question cannot be empty" in response.text

def test_valid_query():
    response = client.post("/answer", json={"question": "What is MLOps?"})
    assert response.status_code == 200
    assert "answer" in response.json()
