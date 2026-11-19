"""FastAPI service for the credit-card fraud decision function.

Run locally:   uvicorn app.main:app --port 8000      (then open http://localhost:8000/docs)

Endpoints
  GET  /            landing page with test metrics and links
  GET  /health      liveness check (used by Docker, Render, Railway)
  GET  /model-info  model, decision policy, test metrics, library versions
  GET  /demo        scores 5 bundled test transactions (3 genuine, 2 real frauds)
  POST /score       scores up to 1,000 transactions
  POST /drift       PSI drift report for a batch of transactions vs. the training reference
  GET  /docs        interactive Swagger UI
"""
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from fraud import predict
from fraud.monitoring import drift_report

BUNDLE = predict.load()
META = BUNDLE["metadata"]
SAMPLE = predict.DEFAULT_MODEL_PATH.parent / "sample_request.csv"

app = FastAPI(
    title="Card fraud scoring API",
    version="1.0.0",
    description=f"{META['model_name']} + Platt calibration. Decision policy: {BUNDLE['policy']} "
                f"(false-alarm cost EUR {BUNDLE['c_fp']:g}, missed-fraud cost = transaction amount).",
)


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds since the first transaction in the dataset")
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount in euros")


class ScoreResult(BaseModel):
    fraud_probability: float
    threshold: float
    decision: str


def _frame(txns: List[Transaction]) -> pd.DataFrame:
    if not txns:
        raise HTTPException(400, "Send at least one transaction.")
    if len(txns) > 1000:
        raise HTTPException(413, "At most 1000 transactions per request.")
    return pd.DataFrame([t.model_dump() for t in txns])


@app.get("/", response_class=HTMLResponse)
def home():
    m = META["test_metrics"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Fraud scoring API</title>
<style>body{{font-family:system-ui,sans-serif;max-width:720px;margin:40px auto;padding:0 16px;line-height:1.5}}
code{{background:#f2f2f2;padding:2px 5px;border-radius:4px}}td{{padding:2px 12px 2px 0}}</style></head><body>
<h1>Credit-card fraud scoring API</h1>
<p>Model: <b>{META['model_name']}</b> with Platt calibration. Decision rule: <b>{BUNDLE['policy']}</b>.</p>
<table><tr><td>Test PR-AUC</td><td>{m['PR-AUC']:.3f}</td></tr><tr><td>Test recall</td><td>{m['recall']:.1%}</td></tr>
<tr><td>Test precision</td><td>{m['precision']:.1%}</td></tr><tr><td>Fraud losses avoided</td><td>{m['saving %']:.1f}%</td></tr></table>
<ul><li><a href="/docs">/docs</a> - interactive documentation (try <code>POST /score</code>)</li>
<li><a href="/demo">/demo</a> - score 5 sample transactions</li>
<li><a href="/model-info">/model-info</a> - full metadata</li><li><a href="/health">/health</a></li></ul>
</body></html>"""


@app.get("/health")
def health():
    return {"status": "ok", "model": META["model_name"], "policy": BUNDLE["policy"]}


@app.get("/model-info")
def model_info():
    return {"policy": BUNDLE["policy"], "c_fp": BUNDLE["c_fp"], "thresholds": BUNDLE["thresholds"], **META}


@app.get("/demo")
def demo():
    df = pd.read_csv(SAMPLE)
    out = predict.score(df).round(4)
    out.insert(0, "Amount", df["Amount"])
    out.insert(0, "note", ["genuine", "genuine", "genuine", "real fraud", "real fraud"][: len(out)])
    return out.to_dict(orient="records")


@app.post("/score", response_model=List[ScoreResult])
def score(txns: List[Transaction]):
    return predict.score(_frame(txns)).to_dict(orient="records")


@app.post("/drift")
def drift(txns: List[Transaction]):
    """PSI per feature against the training reference. Send a reasonably large batch (hundreds of rows)."""
    df = _frame(txns)
    if len(df) < 100:
        raise HTTPException(400, "Send at least 100 transactions for a meaningful drift estimate.")
    report = drift_report(BUNDLE["reference_bins"], df).round(4)
    return report.reset_index(names="feature").to_dict(orient="records")
