from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .utils import sha1_text


REQUIRED_RUN_COLS = [
    "question_id",
    "model_id",
    "model_label",
    "frame",
    "prompt",
    "response",
]


def load_runs(runs_jsonl: str) -> pd.DataFrame:
    df = pd.read_json(runs_jsonl, lines=True)
    missing = [c for c in REQUIRED_RUN_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Runs JSONL missing required columns: {missing}. Got: {list(df.columns)}")
    df = df.copy()
    df["question_id"] = df["question_id"].astype(int)
    df["frame"] = df["frame"].astype(str)
    df["model_label"] = df["model_label"].astype(str)
    df["model_id"] = df["model_id"].astype(str)
    return df


def ensure_run_uid(df: pd.DataFrame, uid_col: str = "run_uid") -> pd.DataFrame:
    df = df.copy()
    if uid_col in df.columns:
        return df

    def _mk_uid(row) -> str:
        ph = sha1_text(row["prompt"])[:12]
        return f"{row['model_id']}|{row['model_label']}|q{int(row['question_id'])}|{row['frame']}|{ph}"

    df[uid_col] = df.apply(_mk_uid, axis=1)
    return df


def load_judge_scores(scores_json: str, expected_n: Optional[int] = None) -> pd.DataFrame:
    with open(scores_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Judge scores must be a non-empty list (one list per judge).")

    n_judges = len(data)
    n_rows = len(data[0])
    if expected_n is not None and expected_n != n_rows:
        raise ValueError(f"Judge scores length {n_rows} does not match expected {expected_n}.")
    for j in range(n_judges):
        if len(data[j]) != n_rows:
            raise ValueError(f"Judge {j} has {len(data[j])} rows, expected {n_rows}.")

    metric_keys = sorted({k for sample in data[0] for k in sample.keys()})

    arr = np.full((n_judges, n_rows, len(metric_keys)), np.nan, dtype=np.float32)
    for j in range(n_judges):
        for i, sample in enumerate(data[j]):
            for m, key in enumerate(metric_keys):
                v = sample.get(key, None)
                if v is None:
                    continue
                try:
                    arr[j, i, m] = float(v)
                except Exception:
                    pass

    mean = np.nanmean(arr, axis=0)
    out = pd.DataFrame(mean, columns=[f"score_{k}" for k in metric_keys])
    out["row_idx"] = np.arange(n_rows, dtype=int)
    return out


def merge_runs_and_scores(df_runs: pd.DataFrame, df_scores: pd.DataFrame) -> pd.DataFrame:
    if len(df_runs) != len(df_scores):
        raise ValueError(f"Runs rows ({len(df_runs)}) must match scores rows ({len(df_scores)}).")
    df = df_runs.copy().reset_index(drop=True)
    df_scores = df_scores.copy().sort_values("row_idx").drop(columns=["row_idx"])
    return pd.concat([df, df_scores.reset_index(drop=True)], axis=1)


def require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as e:
        raise ImportError("pyarrow is required to write parquet. Install with: pip install pyarrow") from e


def write_parquet(df: pd.DataFrame, path: str) -> None:
    require_pyarrow()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet(path: str) -> pd.DataFrame:
    require_pyarrow()
    return pd.read_parquet(path)