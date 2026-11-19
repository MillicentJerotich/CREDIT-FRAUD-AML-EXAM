"""Data pipeline: load -> validate -> clean -> time-ordered split -> save.

Usage:
    python -m fraud.data                    # uses configs/config.yaml
    python -m fraud.data --allow-partial    # continue even if the file has fewer rows than expected

Outputs (in data/processed/):
    train.parquet, validation.parquet, test.parquet, data_report.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from fraud.config import load_config
from fraud.features import RAW_COLS

TARGET = "Class"


class DataValidationError(RuntimeError):
    pass


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Read the Kaggle CSV, or the Kaggle zip that contains it (pandas reads single-file zips directly)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and save it there (see README.md)."
        )
    return pd.read_csv(path)


def validate(df: pd.DataFrame, expected_rows: int, expected_columns: int, allow_partial: bool = False) -> dict:
    """Check the raw data before anything else uses it. Raises DataValidationError with a clear message."""
    problems = []
    expected = RAW_COLS + [TARGET]
    if list(df.columns) != expected:
        problems.append(f"columns differ from the expected {expected_columns} Kaggle columns: {list(df.columns)[:5]}...")
    n_missing = int(df.isna().sum().sum())
    if n_missing:
        rows_with_missing = int(df.isna().any(axis=1).sum())
        problems.append(f"{n_missing} missing values in {rows_with_missing} row(s); the complete file has none. "
                        "This usually means the file was cut off during download or upload.")
    if len(df) != expected_rows and not allow_partial:
        problems.append(f"{len(df):,} rows loaded but the full dataset has {expected_rows:,} "
                        f"({len(df) / expected_rows:.1%}). The file looks incomplete; download it again "
                        "(or pass --allow-partial to continue anyway).")
    if TARGET in df and not df[TARGET].dropna().isin([0, 1]).all():
        problems.append("Class contains values other than 0 and 1.")
    if "Amount" in df and (df["Amount"] < 0).any():
        problems.append("negative transaction amounts found.")
    if problems:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(problems))
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": n_missing,
        "frauds": int(df[TARGET].sum()),
        "fraud_rate": float(df[TARGET].mean()),
        "exact_duplicates": int(df.duplicated().sum()),
        "duplicate_frauds": int(df[df.duplicated()][TARGET].sum()),
        "zero_amount_transactions": int((df["Amount"] == 0).sum()),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates (before splitting, so no row can sit in both train and test) and sort by time."""
    return df.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Oldest train_frac -> train, next val_frac -> validation, newest remainder -> test. Never random."""
    n = len(df)
    i_train, i_val = int(train_frac * n), int((train_frac + val_frac) * n)
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def run(cfg: dict, allow_partial: bool = False) -> dict:
    d = cfg["data"]
    raw_path = Path(d["raw_path"])
    print(f"[data] loading {raw_path}")
    raw = load_raw(raw_path)
    report = {"source_file": raw_path.name, "sha256": sha256(raw_path)}
    report["raw"] = validate(raw, d["expected_rows"], d["expected_columns"], allow_partial)
    print(f"[data] validated: {report['raw']['rows']:,} rows, {report['raw']['frauds']} frauds, "
          f"{report['raw']['exact_duplicates']:,} duplicates, 0 missing values")

    df = clean(raw)
    train, val, test = time_split(df, d["split"]["train"], d["split"]["validation"])
    out_dir = Path(d["processed_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    report["clean_rows"] = int(len(df))
    report["splits"] = {}
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        part.to_parquet(out_dir / f"{name}.parquet", index=False)
        report["splits"][name] = {
            "rows": int(len(part)), "frauds": int(part[TARGET].sum()),
            "fraud_rate": float(part[TARGET].mean()),
            "hours": [round(part["Time"].min() / 3600, 2), round(part["Time"].max() / 3600, 2)],
        }
        print(f"[data] {name:<10} {len(part):>7,} rows  {int(part[TARGET].sum()):>3} frauds  "
              f"hours {report['splits'][name]['hours'][0]:.1f}-{report['splits'][name]['hours'][1]:.1f}")
    (out_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    print(f"[data] wrote {out_dir}/{{train,validation,test}}.parquet and data_report.json")
    return report


def load_splits(cfg: dict):
    out_dir = Path(cfg["data"]["processed_dir"])
    files = [out_dir / f"{n}.parquet" for n in ("train", "validation", "test")]
    if not all(f.exists() for f in files):
        raise FileNotFoundError("Processed data not found; run `python -m fraud.data` first.")
    return [pd.read_parquet(f) for f in files]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--allow-partial", action="store_true", help="do not fail on a row-count mismatch")
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    try:
        run(cfg, allow_partial=args.allow_partial)
    except DataValidationError as e:
        print(f"[data] ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
