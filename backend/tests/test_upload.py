import os
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

def test_upload_txt_file():
    test_file_path = "tests/test_doc.txt"
    with open(test_file_path, "w") as f:
        f.write("This is a test document.")

    with open(test_file_path, "rb") as f:
        response = client.post("/upload", files={"file": ("test_doc.txt", f, "text/plain")})

    os.remove(test_file_path)

    assert response.status_code == 200
    assert "file" in response.json()
