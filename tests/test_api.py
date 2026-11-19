"""API tests against the committed model artifacts."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import SAMPLE, app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_score_sample_transactions():
    rows = pd.read_csv(SAMPLE).to_dict(orient="records")
    r = client.post("/score", json=rows)
    assert r.status_code == 200
    out = r.json()
    assert len(out) == len(rows)
    assert all(0.0 <= o["fraud_probability"] <= 1.0 and o["decision"] in {"APPROVE", "REVIEW"} for o in out)
    assert out[3]["decision"] == "REVIEW"          # the first bundled fraud is flagged


def test_invalid_input_is_rejected():
    assert client.post("/score", json=[{"Time": 1, "Amount": -5}]).status_code == 422
    assert client.post("/score", json=[]).status_code == 400


@pytest.mark.parametrize("path", ["/", "/model-info", "/demo", "/docs"])
def test_pages(path):
    assert client.get(path).status_code == 200
