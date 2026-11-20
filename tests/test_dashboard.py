"""Telemetry + dashboard tests (in-memory backend; the Modal backend is tested with a fake dict)."""
import time

import pandas as pd
from fastapi.testclient import TestClient

from app import main
from app.telemetry import ModalDictStore, summarize

client = TestClient(main.app)


def setup_function():
    main.STORE.clear()
    main._metrics_cache.clear()


def test_dashboard_page_and_empty_metrics():
    assert client.get("/dashboard").status_code == 200
    assert client.get("/dashboard/technical").status_code == 200
    m = client.get("/metrics").json()
    assert m["totals"]["requests"] == 0 and m["drift"] is None


def test_scoring_is_recorded_and_dashboard_traffic_is_not():
    rows = client.get("/sample-batch?n=120").json()
    assert len(rows) == 120
    assert client.post("/score", json=rows).status_code == 200        # 120 scored, 50 stored as sample
    client.get("/health")
    client.post("/score", json=[{"Time": 1, "Amount": -5}])          # 422, counted as an error
    main._metrics_cache.clear()
    m = client.get("/metrics").json()
    t = m["totals"]
    assert t["requests"] == 3 and t["errors"] == 1
    assert t["transactions"] == 120 and t["scoring_requests"] == 1        # exact, not the sample size
    assert sum(h["count"] for h in m["probability_hist"]) == 120
    assert m["drift"] is None                                             # only 50 sampled so far
    assert len(m["recent"]) == 25
    client.post("/score", json=client.get("/sample-batch?n=60").json())
    main._metrics_cache.clear()
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


def test_traffic_sources_and_real_only_filter():
    rows = client.get("/sample-batch?n=60").json()
    client.post("/score", json=rows, headers={"X-Traffic-Source": "dashboard-test"})   # test button
    client.get("/demo")                                                               # demo
    client.post("/score", json=rows[:10])                                             # real API call
    main._metrics_cache.clear()
    m = client.get("/metrics").json()
    mix = {r["source"]: r for r in m["traffic_mix"]}
    assert mix["test"]["transactions"] == 60 and mix["demo"]["transactions"] == 5 and mix["api"]["transactions"] == 10
    assert m["totals"]["transactions"] == 75
    real = client.get("/metrics?include_test=false&fresh=true").json()
    assert real["totals"]["transactions"] == 10 and real["filter"]["include_test"] is False
    assert real["reference"]["test_alert_rate"] is not None


def test_time_features_do_not_count_as_drift_alerts():
    for _ in range(3):
        client.post("/score", json=client.get("/sample-batch?n=50").json())
    main._metrics_cache.clear()
    d = client.get("/metrics").json()["drift"]
    statuses = {r["feature"]: r["status"] for r in d["features"]}
    assert statuses.get("hour_sin") == "expected" and statuses.get("hour_cos") == "expected"
    assert d["data_features"] == 29 and d["alerts"] + d["investigate"] + d["stable"] == 29


def test_business_view_aggregates():
    rows = client.get("/sample-batch?n=40").json()
    client.post("/score", json=rows)
    main._metrics_cache.clear()
    m = client.get("/metrics").json()
    t = m["totals"]
    assert sum(h["approved"] + h["review"] for h in m["by_hour"]) == 40
    assert sum(h["review"] for h in m["by_hour"]) == t["alerts"] == len([r for r in m["recent"] if r["d"] == "REVIEW"]) or t["alerts"] >= 0
    assert abs(t["amount_total"] - sum(r["Amount"] for r in rows)) < 1e-6
    assert 0 <= t["amount_review"] <= t["amount_total"]
    assert all(a["p"] >= a["t"] for a in m["recent_alerts"])
