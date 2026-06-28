import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_ask_endpoint():

    response = client.post(
        "/ask",
        json={
            "question":"concert à Paris"
        }
    )

    assert response.status_code == 200

    assert "answer" in response.json()
    
    