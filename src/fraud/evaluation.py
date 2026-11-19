"""Metrics, the euro cost model, decision policies and bootstrap confidence intervals."""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score)

POLICIES = ["Default 0.5", "Amount-aware Bayes rule", "Best single threshold", "Max-F2 threshold"]


def scores_of(model, X):
    """Fraud score where higher = more likely fraud (IsolationForest has no predict_proba)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return -model.score_samples(X)


def rank_metrics(y, s) -> dict:
    return {"PR-AUC": average_precision_score(y, s), "ROC-AUC": roc_auc_score(y, s)}


def decision_cost(y_true, y_flag, amount, c_fp: float) -> float:
    """Total euro cost: a missed fraud loses its amount; each false alarm costs c_fp."""
    y_true, y_flag, amount = np.asarray(y_true), np.asarray(y_flag).astype(bool), np.asarray(amount)
    return amount[(y_true == 1) & ~y_flag].sum() + c_fp * ((y_true == 0) & y_flag).sum()


def per_transaction_cost(y, flag, amount, c_fp: float) -> np.ndarray:
    flag = np.asarray(flag, bool)
    return np.where((y == 1) & ~flag, amount, 0.0) + np.where((y == 0) & flag, c_fp, 0.0)


def apply_policy(policy: str, p, amount, c_fp: float, thresholds: dict):
    """Turn calibrated probabilities into flag/approve decisions.

    Amount-aware Bayes rule: flag iff p >= c_fp / (c_fp + Amount), the cost-optimal threshold when a
    missed fraud costs its amount and a false alarm costs c_fp.
    """
    p, amount = np.asarray(p), np.asarray(amount, dtype=float)
    if policy == "Default 0.5":
        return p >= 0.5
    if policy == "Amount-aware Bayes rule":
        return p >= c_fp / (c_fp + amount)
    if policy == "Best single threshold":
        return p >= thresholds["cost"]
    if policy == "Max-F2 threshold":
        return p >= thresholds["f2"]
    raise ValueError(policy)


def policy_report(y, p, amount, c_fp: float, thresholds: dict) -> pd.DataFrame:
    y, amount = np.asarray(y), np.asarray(amount, dtype=float)
    base_cost = decision_cost(y, np.zeros_like(y), amount, c_fp)
    rows = {}
    for pol in POLICIES:
        f = apply_policy(pol, p, amount, c_fp, thresholds)
        tn, fp, fn, tp = confusion_matrix(y, f, labels=[0, 1]).ravel()
        c = decision_cost(y, f, amount, c_fp)
        rows[pol] = {"alerts": int(f.sum()), "TP": int(tp), "FP": int(fp), "FN": int(fn),
                     "precision": precision_score(y, f, zero_division=0), "recall": recall_score(y, f),
                     "fraud € caught %": 100 * amount[(y == 1) & f].sum() / amount[y == 1].sum(),
                     "cost €": c, "saving vs no model %": 100 * (1 - c / base_cost)}
    rows["No model (approve all)"] = {"alerts": 0, "TP": 0, "FP": 0, "FN": int(y.sum()), "precision": np.nan,
                                      "recall": 0.0, "fraud € caught %": 0.0, "cost €": base_cost,
                                      "saving vs no model %": 0.0}
    return pd.DataFrame(rows).T


def stratified_bootstrap_idx(y, n_boot: int, rng):
    """Resample positives and negatives separately so every resample has the same number of frauds."""
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
