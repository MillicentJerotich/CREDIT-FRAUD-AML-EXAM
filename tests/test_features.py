import numpy as np
import pandas as pd
import pytest

from fraud.features import OUT_COLS, RAW_COLS, FraudFeatures


def make_rows(n=3):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, len(RAW_COLS))), columns=RAW_COLS)
    df["Time"] = [0, 6 * 3600, 23.5 * 3600][:n]
    df["Amount"] = [0.0, 9.0, 999.0][:n]
    return df


def test_output_columns_and_shape():
    out = FraudFeatures().fit_transform(make_rows())
    assert list(out.columns) == OUT_COLS
    assert out.shape == (3, 31)


def test_log_amount_and_cyclical_hour():
    out = FraudFeatures().transform(make_rows())
    assert out["log_amount"].tolist() == pytest.approx([0.0, np.log(10), np.log(1000)])
    # 00:00 and 23:30 are close on the circle; 06:00 is far from both
    d = lambda i, j: np.hypot(out.hour_sin[i] - out.hour_sin[j], out.hour_cos[i] - out.hour_cos[j])
    assert d(0, 2) < 0.2 < d(0, 1)


def test_missing_column_raises():
    with pytest.raises(ValueError, match="Amount"):
        FraudFeatures().transform(make_rows().drop(columns="Amount"))
