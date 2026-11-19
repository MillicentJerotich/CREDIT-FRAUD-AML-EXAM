"""Telemetry + dashboard tests (in-memory backend; the Modal backend is tested with a fake dict)."""
import time

import pandas as pd
from fastapi.testclient import TestClient

from app import main
from app.telemetry import ModalDictStore, summarize

client = TestClient(main.app)


def setup_function():
    main.STORE.clear()
    main._metrics_cache["data"] = None


def test_dashboard_page_and_empty_metrics():
    assert client.get("/dashboard").status_code == 200
    m = client.get("/metrics").json()
    assert m["totals"]["requests"] == 0 and m["drift"] is None


def test_scoring_is_recorded_and_dashboard_traffic_is_not():
    rows = client.get("/sample-batch?n=120").json()
    assert len(rows) == 120
    assert client.post("/score", json=rows).status_code == 200        # 120 scored, 50 stored as sample
    client.get("/health")
    client.post("/score", json=[{"Time": 1, "Amount": -5}])          # 422, counted as an error
    main._metrics_cache["data"] = None
    m = client.get("/metrics").json()
    t = m["totals"]
    assert t["requests"] == 3 and t["errors"] == 1
    assert t["transactions"] == 120 and t["scoring_requests"] == 1        # exact, not the sample size
    assert sum(h["count"] for h in m["probability_hist"]) == 120
    assert m["drift"] is None                                             # only 50 sampled so far
    assert len(m["recent"]) == 25
    client.post("/score", json=client.get("/sample-batch?n=60").json())
    main._metrics_cache["data"] = None
    m = client.get("/metrics").json()
    assert m["totals"]["transactions"] == 180
    assert m["drift"] is not None and m["drift"]["n"] == 100 and len(m["drift"]["features"]) == 10
    endpoints = {r["endpoint"] for r in m["by_endpoint"]}
    assert "POST /score" in endpoints and "GET /metrics" not in endpoints and "GET /sample-batch" not in endpoints


class FakeDict:
    """Mimics the modal.Dict methods ModalDictStore uses."""
    def __init__(self): self.d = {}
    def put(self, k, v): self.d[k] = v; return True
    def len(self): return len(self.d)
    def keys(self): return iter(list(self.d))
    def items(self): return iter(list(self.d.items()))
    def pop(self, k, default=None): return self.d.pop(k, default)
    def clear(self): self.d.clear()


def test_modal_dict_store_orders_and_prunes():
    store = ModalDictStore("unused", max_events=10, dict_obj=FakeDict())
    for i in range(30):
        store.record({"ts": 1000.0 + i, "path": "/health", "method": "GET", "status": 200, "latency_ms": 1.0, "n": 0, "scores": None})
    ev = store.events()
    assert len(ev) == 10 and ev[-1]["ts"] == 1029.0 and ev == sorted(ev, key=lambda e: e["ts"])
    assert summarize(ev, main.BUNDLE, now=1030.0)["totals"]["requests"] == 10
