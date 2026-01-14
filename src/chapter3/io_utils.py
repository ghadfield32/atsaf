# file: src/chapter3/io_utils.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    """
    Atomic parquet write: write to temp in same directory, then replace.
    """
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_json(payload: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
