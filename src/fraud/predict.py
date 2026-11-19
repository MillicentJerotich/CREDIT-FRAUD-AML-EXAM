"""Load the packaged decision function and score raw transactions.

The bundle (artifacts/fraud_model.joblib) holds: feature engineering -> model -> Platt calibration,
plus the decision policy, its cost parameter, metadata and the PSI reference profile.
"""
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud.config import ROOT
from fraud.evaluation import apply_policy

DEFAULT_MODEL_PATH = Path(os.environ.get("MODEL_PATH", ROOT / "artifacts" / "fraud_model.joblib"))
_BUNDLE = None


def load(path=None) -> dict:
    global _BUNDLE
    _BUNDLE = joblib.load(path or DEFAULT_MODEL_PATH)
    return _BUNDLE


def score(transactions: pd.DataFrame) -> pd.DataFrame:
    b = _BUNDLE or load()
    missing = set(b["input_columns"]) - set(transactions.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    X = transactions[b["input_columns"]]
    p = b["model"].predict_proba(X)[:, 1]
    amount = X["Amount"].to_numpy(dtype=float)
    policy, t = b["policy"], b["thresholds"]
    if policy == "Amount-aware Bayes rule":
        threshold = b["c_fp"] / (b["c_fp"] + amount)      # cost-optimal: a miss costs its amount
    else:
        fixed = {"Default 0.5": 0.5, "Best single threshold": t["cost"], "Max-F2 threshold": t["f2"]}[policy]
        threshold = np.full(len(X), fixed)
    flag = apply_policy(b["policy"], p, amount, b["c_fp"], b["thresholds"])
    return pd.DataFrame({"fraud_probability": p, "threshold": threshold,
                         "decision": np.where(flag, "REVIEW", "APPROVE")}, index=X.index)
