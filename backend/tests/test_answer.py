import os
from fastapi.testclient import TestClient
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
