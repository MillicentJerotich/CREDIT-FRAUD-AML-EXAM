"""Load configs/config.yaml and resolve paths relative to the repository root."""
import os
from pathlib import Path

import yaml

# Repository root: FRAUD_ROOT if set (Docker/Modal), otherwise two levels above this file (src/fraud/..).
ROOT = Path(os.environ.get("FRAUD_ROOT", Path(__file__).resolve().parents[2]))


def load_config(path: str | Path = ROOT / "configs" / "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("artifacts_dir", "figures_dir"):
        cfg[key] = ROOT / cfg[key]
    cfg["data"]["raw_path"] = ROOT / cfg["data"]["raw_path"]
    cfg["data"]["processed_dir"] = ROOT / cfg["data"]["processed_dir"]
    return cfg
