import numpy as np
import pandas as pd
import pytest

from fraud.data import DataValidationError, clean, time_split, validate
from fraud.features import RAW_COLS


def toy(n=100, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(RAW_COLS))), columns=RAW_COLS)
    df["Time"] = np.arange(n, dtype=float)[::-1]          # deliberately unsorted
    df["Amount"] = rng.uniform(0, 100, n)
    df["Class"] = (np.arange(n) % 10 == 0).astype(int)
    return df


def test_valid_data_passes():
    rep = validate(toy(), expected_rows=100, expected_columns=31)
    assert rep["rows"] == 100 and rep["missing_values"] == 0 and rep["frauds"] == 10


def test_truncated_file_is_rejected():
    df = toy()
    df.loc[len(df) - 1, ["V20", "Amount", "Class"]] = np.nan       # a cut-off last line
    with pytest.raises(DataValidationError, match="missing values"):
        validate(df, expected_rows=100, expected_columns=31)
    with pytest.raises(DataValidationError, match="incomplete"):
        validate(toy().iloc[:75], expected_rows=100, expected_columns=31)


def test_allow_partial_only_relaxes_row_count():
    validate(toy().iloc[:75], expected_rows=100, expected_columns=31, allow_partial=True)


def test_clean_drops_duplicates_and_sorts_by_time():
    df = pd.concat([toy(), toy().iloc[:5]])
    out = clean(df)
    assert len(out) == 100 and out["Time"].is_monotonic_increasing


def test_time_split_is_ordered_and_complete():
    tr, va, te = time_split(clean(toy()), 0.6, 0.2)
    assert (len(tr), len(va), len(te)) == (60, 20, 20)
    assert tr.Time.max() < va.Time.min() and va.Time.max() < te.Time.min()
