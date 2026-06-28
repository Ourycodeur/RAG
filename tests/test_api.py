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
    
    