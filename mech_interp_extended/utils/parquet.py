from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd


def _has_parquet_engine() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def write_parquet(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    """Write parquet with a helpful error message if parquet engine is missing."""
    path = str(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not _has_parquet_engine():
        raise RuntimeError(
            "Parquet support not found. Install one of: `pip install pyarrow` (recommended) or `pip install fastparquet`."
        )
    df.to_parquet(path, index=False)


def read_parquet(path: str | os.PathLike[str]) -> pd.DataFrame:
    if not _has_parquet_engine():
        raise RuntimeError(
            "Parquet support not found. Install one of: `pip install pyarrow` (recommended) or `pip install fastparquet`."
        )
    return pd.read_parquet(path)
