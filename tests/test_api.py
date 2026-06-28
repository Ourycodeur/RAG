import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_empty_question():

    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 400


def test_rebuild_endpoint():

    response = client.post("/rebuild")

    assert response.status_code == 200