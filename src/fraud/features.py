"""Feature engineering shared by training, the saved model and the API (prevents training/serving skew)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time"] + V_COLS + ["Amount"]
OUT_COLS = V_COLS + ["log_amount", "hour_sin", "hour_cos"]


class FraudFeatures(BaseEstimator, TransformerMixin):
    """Stateless transformer: raw transaction columns -> model features.

    * log_amount = log(1 + Amount): tames the heavy right tail of amounts.
    * hour_sin / hour_cos: cyclical hour of day, so 23:00 and 00:00 are close.
    * Raw Time is dropped: it counts seconds since the data started and does not generalise.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=RAW_COLS)
        missing = set(RAW_COLS) - set(X.columns)
        if missing:
            raise ValueError(f"Missing input columns: {sorted(missing)}")
        out = X[V_COLS].astype(float).copy()
        out["log_amount"] = np.log1p(X["Amount"].astype(float).clip(lower=0))
        hour = (X["Time"].astype(float) % 86400) / 3600.0
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(OUT_COLS)
