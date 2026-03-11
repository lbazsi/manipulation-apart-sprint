from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REQUIRED_RUN_FIELDS = [
    "id",
    "question_id",
    "frame",
    "model_label",
    "prompt",
    "response",
]


def load_runs_jsonl(path: str) -> pd.DataFrame:
    """Load a DEEB-style runs JSONL into a dataframe.

    The repo uses JSONL where each line is a run condition (prompt+response) for a specific frame/model_label.
    """
    df = pd.read_json(path, lines=True)
    missing = [c for c in REQUIRED_RUN_FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"Runs JSONL missing required fields: {missing}. Present columns: {list(df.columns)}")

    # Normalize types
    df = df.copy()
    df["question_id"] = df["question_id"].astype(int)
    df["id"] = df["id"].astype(str)
    df["frame"] = df["frame"].astype(str)
    df["model_label"] = df["model_label"].astype(str)
    return df


def load_judge_scores_jsonl(path: str) -> pd.DataFrame:
    """Load judge scores JSONL output by three_judges_pipeline.py"""
    df = pd.read_json(path, lines=True)

    required = ["question_id", "frame", "model_label", "behavior", "score", "judge_model_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Judge scores JSONL missing required fields: {missing}. Present columns: {list(df.columns)}")

    df = df.copy()
    df["question_id"] = df["question_id"].astype(int)
    df["frame"] = df["frame"].astype(str)
    df["model_label"] = df["model_label"].astype(str)
    df["behavior"] = df["behavior"].astype(str)
    df["judge_model_id"] = df["judge_model_id"].astype(str)

    # score can be None sometimes
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df


def aggregate_judge_scores(judge_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate judge scores across judges to mean score per condition+behavior."""
    gcols = ["question_id", "frame", "model_label", "behavior"]
    out = (
        judge_df.groupby(gcols, dropna=False)["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "score_mean"})
    )
    return out
