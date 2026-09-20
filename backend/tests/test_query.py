from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["num_chunks"] > 0


def test_valid_query():
    resp = client.post("/query", json={"question": "ما هي أهلية القاصر؟", "top_k": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert isinstance(body["sources"], list)
    assert len(body["sources"]) > 0


def test_invalid_query_missing_question():
    resp = client.post("/query", json={"top_k": 3})
    assert resp.status_code == 422


def test_invalid_query_too_short():
    resp = client.post("/query", json={"question": "ab"})
    assert resp.status_code == 422
