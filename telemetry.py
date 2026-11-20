"""Request telemetry for the live dashboard.

Every API call (except the dashboard's own traffic) is recorded as one small event:
    {"ts": 1795120000.12, "path": "/score", "method": "POST", "status": 200, "latency_ms": 18.4,
     "n": 120, "alerts": 9, "p_sum": 10.7, "hist": [0, 98, 9, 2, 1, 10],          # exact, over all n
     "scores": [{"amount": 12.5, "p": 0.0003, "t": 0.2857, "d": "APPROVE", "x": [...30 raw inputs]}, ...]}
                                                                              # sample of at most 50
Totals, alert counts and the probability histogram are exact; the recent-transactions table and the drift
check use the stored samples. Keeping events small matters on Modal, where every dashboard refresh reads them.

Two storage backends:
* MemoryStore (default): kept in the server process. Fine locally and in Docker; lost on restart.
* ModalDictStore (TELEMETRY_BACKEND=modal): a persistent modal.Dict shared by every container, so the
  dashboard survives Modal's scale-to-zero and shows traffic from all containers.
"""
import json
import os
import time
import uuid
from collections import deque

import numpy as np
import pandas as pd

from fraud.features import OUT_COLS, RAW_COLS, FraudFeatures
from fraud.monitoring import psi, status

MAX_EVENTS = int(os.environ.get("TELEMETRY_MAX_EVENTS", "2000"))
MAX_SCORES_PER_EVENT = 50           # cap the transaction detail stored per request
HIST_EDGES = [0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0000001]
HIST_LABELS = ["<0.0001", "0.0001–0.001", "0.001–0.01", "0.01–0.1", "0.1–0.5", "≥0.5"]
NOT_RECORDED = ("/dashboard", "/metrics", "/docs", "/openapi.json", "/redoc", "/favicon.ico", "/sample-batch")


class MemoryStore:
    remote = False

    def __init__(self, max_events: int = MAX_EVENTS):
        self._events = deque(maxlen=max_events)

    def record(self, event: dict) -> None:
        self._events.append(event)

    def events(self) -> list:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()


class ModalDictStore:
    """Events stored as individual keys (time-sortable) in a persistent modal.Dict."""
    remote = True

    def __init__(self, name: str, max_events: int = MAX_EVENTS, dict_obj=None):
        if dict_obj is None:
            import modal
            dict_obj = modal.Dict.from_name(name, create_if_missing=True)
        self._d = dict_obj
        self.max_events = max_events

    def record(self, event: dict) -> None:
        key = f"{event['ts']:017.6f}-{uuid.uuid4().hex[:8]}"
        self._d.put(key, event)
        if self._d.len() > self.max_events * 1.2:          # prune occasionally, oldest first
            keys = sorted(k for k in self._d.keys())
            for k in keys[: len(keys) - self.max_events]:
                self._d.pop(k, None)

    def events(self) -> list:
        items = sorted(self._d.items(), key=lambda kv: kv[0])
        return [v for _, v in items[-self.max_events:]]

    def clear(self) -> None:
        self._d.clear()


def make_store():
    if os.environ.get("TELEMETRY_BACKEND", "memory").lower() == "modal":
        return ModalDictStore(os.environ.get("TELEMETRY_DICT", "fraud-api-telemetry"))
    return MemoryStore()


def scoring_payload(df: pd.DataFrame, scored: pd.DataFrame) -> dict:
    """Exact aggregates over the whole request plus a capped sample of transactions for the dashboard."""
    p = scored["fraud_probability"].to_numpy(dtype=float)
    sample = []
    for (_, raw), (_, s) in zip(df.head(MAX_SCORES_PER_EVENT).iterrows(), scored.head(MAX_SCORES_PER_EVENT).iterrows()):
        sample.append({"amount": float(raw["Amount"]), "p": float(s["fraud_probability"]),
                       "t": float(s["threshold"]), "d": s["decision"],
                       "x": [round(float(raw[c]), 5) for c in RAW_COLS]})
    review = (scored["decision"] == "REVIEW").to_numpy()
    amount = df["Amount"].to_numpy(dtype=float)
    hour = ((df["Time"].to_numpy(dtype=float) % 86400) // 3600).astype(int)      # transaction's hour of day
    return {"n": int(len(df)), "alerts": int(review.sum()), "p_sum": float(p.sum()),
            "hist": np.histogram(p, bins=HIST_EDGES)[0].tolist(),
            "amount_total": float(amount.sum()), "amount_review": float(amount[review].sum()),
            "hours_approved": np.bincount(hour[~review], minlength=24)[:24].tolist(),
            "hours_review": np.bincount(hour[review], minlength=24)[:24].tolist(),
            "scores": sample}


TIME_FEATURES = {"hour_sin", "hour_cos"}
SOURCE_LABELS = {"api": "real API calls", "test": "test button", "demo": "/demo", "earlier": "recorded before labelling"}


def load_reference(metrics_path) -> dict:
    """Expected behaviour from the offline evaluation (test set), for comparison on the dashboard."""
    ref = {"test_alert_rate": None, "test_fraud_rate": None, "train_fraud_rate": 0.002}
    try:
        m = json.loads(open(metrics_path).read())
        cm = m["test_confusion_matrix"]                       # [[TN, FP], [FN, TP]]
        n = sum(map(sum, cm))
        ref["test_alert_rate"] = (cm[0][1] + cm[1][1]) / n
        ref["test_fraud_rate"] = (cm[1][0] + cm[1][1]) / n
    except Exception:
        pass
    return ref


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else None


def summarize(events: list, bundle: dict, now: float | None = None, include_test: bool = True,
              reference: dict | None = None) -> dict:
    """Aggregate raw events into everything the dashboard displays.

    include_test=False keeps only real API traffic (source 'api'); events recorded before sources were
    labelled count as test traffic.
    """
    now = now or time.time()
    all_events = events
    mix = {}
    for e in all_events:
        src = e.get("source", "earlier")
        d = mix.setdefault(src, {"requests": 0, "transactions": 0})
        d["requests"] += 1
        d["transactions"] += e.get("n", 0) if e.get("scores") is not None else 0
    if not include_test:
        events = [e for e in events if e.get("source") == "api"]
    scoring = [e for e in events if e.get("scores") is not None]
    txns = [dict(s, ts=e["ts"]) for e in scoring for s in e["scores"]]        # stored samples
    lat = [e["latency_ms"] for e in scoring]
    n_txns = sum(e.get("n", 0) for e in scoring)
    alerts = sum(e.get("alerts", 0) for e in scoring)
    p_sum = sum(e.get("p_sum", 0.0) for e in scoring)

    # Requests per minute, last 30 minutes (all recorded endpoints)
    minutes = []
    for i in range(29, -1, -1):
        start = now - (i + 1) * 60
        n = sum(start < e["ts"] <= start + 60 for e in events)
        minutes.append({"t": start + 60, "count": n})

    # Fraud-probability histogram on log-spaced bins (almost all scores are tiny); exact counts
    counts = np.sum([e.get("hist", [0] * len(HIST_LABELS)) for e in scoring], axis=0).tolist() if scoring \
        else [0] * len(HIST_LABELS)

    # Drift of recently scored inputs vs the training reference (needs a reasonable sample)
    drift = None
    if len(txns) >= 100:
        recent = pd.DataFrame([t["x"] for t in txns[-2000:]], columns=RAW_COLS)
        feats = FraudFeatures().transform(recent)
        rows = [{"feature": c, "psi": round(psi(bundle["reference_bins"][c], feats[c]), 4)} for c in OUT_COLS]
        rows.sort(key=lambda r: r["psi"], reverse=True)
        for r in rows:
            # hour-of-day features shift whenever traffic covers only part of the day: expected, not an alarm
            r["status"] = "expected" if r["feature"] in TIME_FEATURES else status(r["psi"])
        data_rows = [r for r in rows if r["feature"] not in TIME_FEATURES]
        drift = {"n": min(len(txns), 2000), "sampled": True, "features": rows[:10],
                 "alerts": sum(r["status"] == "ALERT" for r in data_rows),
                 "investigate": sum(r["status"] == "investigate" for r in data_rows),
                 "stable": sum(r["status"] == "stable" for r in data_rows),
                 "data_features": len(data_rows),
                 "max_data_psi": max(r["psi"] for r in data_rows),
                 "top_data_feature": data_rows[0]["feature"]}

    recent_txns = [{"ts": t["ts"], "amount": t["amount"], "p": t["p"], "t": t["t"], "d": t["d"],
                    "hour": int((t["x"][0] % 86400) // 3600) if t.get("x") else None} for t in txns[-25:]][::-1]
    recent_alerts = [{"ts": t["ts"], "amount": t["amount"], "p": t["p"], "t": t["t"],
                      "hour": int((t["x"][0] % 86400) // 3600) if t.get("x") else None}
                     for t in txns if t["d"] == "REVIEW"][-8:][::-1]
    hours_ok = np.sum([e.get("hours_approved", [0] * 24) for e in scoring], axis=0).tolist() if scoring else [0] * 24
    hours_rv = np.sum([e.get("hours_review", [0] * 24) for e in scoring], axis=0).tolist() if scoring else [0] * 24
    m = bundle["metadata"]
    return {
        "generated": now,
        "model": {"name": m["model_name"], "policy": bundle["policy"], "c_fp": bundle["c_fp"],
                  "sklearn": m.get("sklearn"), "created": m.get("created"), "test_metrics": m["test_metrics"]},
        "filter": {"include_test": include_test},
        "traffic_mix": [{"source": k, "label": SOURCE_LABELS.get(k, k), **v} for k, v in sorted(mix.items())],
        "reference": reference or {},
        "totals": {
            "requests": len(events),
            "errors": sum(e["status"] >= 400 for e in events),
            "client_errors": sum(400 <= e["status"] < 500 for e in events),
            "server_errors": sum(e["status"] >= 500 for e in events),
            "scoring_requests": len(scoring),
            "transactions": int(n_txns),
            "alerts": int(alerts),
            "alert_rate": (alerts / n_txns) if n_txns else None,
            "mean_p": (p_sum / n_txns) if n_txns else None,
            "latency_avg_ms": float(np.mean(lat)) if lat else None,
            "latency_p95_ms": _pct(lat, 95),
            "amount_total": float(sum(e.get("amount_total", 0.0) for e in scoring)),
            "amount_review": float(sum(e.get("amount_review", 0.0) for e in scoring)),
            "first_ts": events[0]["ts"] if events else None,
            "last_ts": events[-1]["ts"] if events else None,
        },
        "by_endpoint": _by_endpoint(events),
        "requests_per_minute": minutes,
        "probability_hist": [{"bin": b, "count": int(c)} for b, c in zip(HIST_LABELS, counts)],
        "latency_series": [{"ts": e["ts"], "ms": round(e["latency_ms"], 1), "path": e["path"], "n": e.get("n", 0)}
                           for e in scoring[-100:]],
        "recent": recent_txns,
        "recent_alerts": recent_alerts,
        "by_hour": [{"hour": h, "approved": int(hours_ok[h]), "review": int(hours_rv[h])} for h in range(24)],
        "drift": drift,
    }


def _by_endpoint(events):
    out = {}
    for e in events:
        k = f"{e['method']} {e['path']}"
        d = out.setdefault(k, {"count": 0, "errors": 0, "lat": []})
        d["count"] += 1
        d["errors"] += e["status"] >= 400
        d["lat"].append(e["latency_ms"])
    return sorted([{"endpoint": k, "count": v["count"], "errors": v["errors"],
                    "avg_ms": round(float(np.mean(v["lat"])), 1)} for k, v in out.items()],
                  key=lambda r: r["count"], reverse=True)
