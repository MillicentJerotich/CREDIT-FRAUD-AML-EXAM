"""Drift monitoring with the Population Stability Index (PSI).

PSI = sum_b (q_b - p_b) * ln(q_b / p_b) over reference deciles.
Rule of thumb: < 0.10 stable, 0.10-0.25 investigate, > 0.25 significant shift.

Usage (compare a new batch of raw transactions with the training reference stored in the model bundle):
    python -m fraud.monitoring path/to/new_transactions.csv
    python -m fraud.monitoring data/processed/test.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraud.features import OUT_COLS, FraudFeatures


def make_reference(values, n_bins: int = 10) -> dict:
    """Decile edges and the share of reference rows in each bin."""
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    props = np.histogram(values, bins=edges)[0] / len(values)
    return {"edges": edges.tolist(), "props": props.tolist()}


def psi(reference: dict, current, eps: float = 1e-6) -> float:
    edges, p = np.array(reference["edges"]), np.array(reference["props"])
    q = np.histogram(current, bins=edges)[0] / len(current)
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def status(value: float) -> str:
    return "ALERT" if value > 0.25 else "investigate" if value > 0.10 else "stable"


def drift_report(reference_bins: dict, raw: pd.DataFrame) -> pd.DataFrame:
    feats = FraudFeatures().transform(raw)
    rows = {c: psi(reference_bins[c], feats[c]) for c in OUT_COLS if c in reference_bins}
    out = pd.DataFrame({"PSI": pd.Series(rows)}).sort_values("PSI", ascending=False)
    out["status"] = out["PSI"].map(status)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch", help="CSV or parquet file of raw transactions (Time, V1..V28, Amount)")
    ap.add_argument("--model", default=None, help="model bundle (default: artifacts/fraud_model.joblib)")
    args = ap.parse_args()
    from fraud.predict import load
    bundle = load(args.model) if args.model else load()
    path = Path(args.batch)
    raw = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    report = drift_report(bundle["reference_bins"], raw)
    pd.set_option("display.width", 120)
    print(report.round(3).to_string())
    n_alert = int((report["status"] == "ALERT").sum())
    print(f"\n{n_alert} feature(s) in ALERT. Note: hour features shift whenever a batch covers only part of the day.")


if __name__ == "__main__":
    main()
