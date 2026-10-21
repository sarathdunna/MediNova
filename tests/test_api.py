from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_has_disclaimer():
    r = client.get("/health")
    assert r.status_code == 200 and "Not for clinical use" in r.json()["disclaimer"]


def test_missing_input_is_hard_abstention():
    r = client.post("/predict", data={"run": "anything", "report": "No effusion."})
    body = r.json()
    assert body["abstain"] is True and body["reason"] == "missing_input" and body["p_match"] is None


def test_unknown_run_is_404():
    assert client.get("/runs/does-not-exist/metrics").status_code == 404
