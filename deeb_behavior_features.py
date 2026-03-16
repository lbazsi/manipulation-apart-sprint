"""
Shared feature pipeline for DEEB behavior classifier: load judge aggregated JSONL,
build delta features (E-C, O-C), load mech interp (numeric only), reduce to per
(question_id, model_label), merge, and return X, y, feature_cols, label_encoder.
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

CRITERIA = ["quality", "sycophancy", "evasion", "deception", "directness"]


def load_judge_aggregated(path: str) -> pd.DataFrame:
    """Load aggregated judge JSONL; ensure required columns exist."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Judge file not found: {path}")
    df = pd.read_json(p, lines=True)
    required = ["question_id", "frame", "model_label"] + CRITERIA
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Judge file missing columns: {missing}")
    return df


def build_delta_features(judge_df: pd.DataFrame) -> pd.DataFrame:
    """
    From judge DataFrame (one row per question_id, frame, model_label with score columns),
    compute |evaluation - casual| and |oversight - casual| for each criterion.
    Returns DataFrame with one row per (question_id, model_label) and 10 delta columns.
    """
    full = judge_df.sort_values(["question_id", "model_label", "frame"]).reset_index(drop=True)
    df_n = full[full["frame"] == "casual"].set_index(["question_id", "model_label"])
    df_e = full[full["frame"] == "evaluation"].set_index(["question_id", "model_label"])
    df_o = full[full["frame"] == "oversight"].set_index(["question_id", "model_label"])

    delta_e_n = (df_e[CRITERIA] - df_n[CRITERIA]).abs()
    delta_e_n.columns = [f"dEabs_{c[:5]}" for c in CRITERIA]
    delta_o_n = (df_o[CRITERIA] - df_n[CRITERIA]).abs()
    delta_o_n.columns = [f"dOabs_{c[:5]}" for c in CRITERIA]

    feature_df = pd.concat([delta_e_n, delta_o_n], axis=1).reset_index()
    return feature_df


def load_mech_interp(path: str, numeric_only: bool = True) -> pd.DataFrame:
    """
    Load mech interp JSONL. If numeric_only, keep only question_id, frame, model_label
    and columns that are numeric (int/float); drop object columns (text, paths, json).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mech interp file not found: {path}")
    df = pd.read_json(p, lines=True)

    if not numeric_only:
        return df

    key_cols = ["question_id", "frame", "model_label"]
    drop_patterns = ("path", "json", "history", "prompt", "response", "base_question",
                     "source_file", "source_run", "id", "base_model", "condition_id",
                     "pair_name", "behavior_type", "paired_frame", "paired_model_label",
                     "delta_metric", "component", "circuit_path", "circuit_model_id",
                     "model_id", "passes_baseline")
    keep = [c for c in df.columns if c in key_cols]
    for c in df.columns:
        if c in keep:
            continue
        if df[c].dtype in (np.int64, np.float64, "int64", "float64", "float32", "int32"):
            keep.append(c)
        elif any(x in c.lower() for x in drop_patterns):
            continue
        elif df[c].dtype == object or df[c].dtype == "object":
            continue
        else:
            try:
                pd.to_numeric(df[c], errors="raise")
                keep.append(c)
            except (TypeError, ValueError):
                pass
    keep = list(dict.fromkeys(c for c in keep if c in df.columns))
    return df[keep].copy()


def reduce_mech_to_per_question_model(df_mech: pd.DataFrame, frame: str = "casual") -> pd.DataFrame:
    """
    Reduce mech from one row per (question_id, frame, model_label) to one row per
    (question_id, model_label) by taking rows for the given frame only.
    Keeps first row when there are duplicates for the same (question_id, model_label).
    """
    if "frame" not in df_mech.columns:
        return df_mech.drop_duplicates(subset=["question_id", "model_label"], keep="first")
    reduced = df_mech[df_mech["frame"] == frame].copy()
    reduced = reduced.drop(columns=["frame"], errors="ignore")
    reduced = reduced.drop_duplicates(subset=["question_id", "model_label"], keep="first")
    return reduced


def merge_judge_deltas_and_mech(judge_deltas_df: pd.DataFrame, mech_reduced_df: pd.DataFrame) -> pd.DataFrame:
    """Merge judge delta features with mech interp on (question_id, model_label)."""
    j = judge_deltas_df.drop_duplicates(subset=["question_id", "model_label"], keep="first").set_index(
        ["question_id", "model_label"]
    )
    m = mech_reduced_df.drop_duplicates(subset=["question_id", "model_label"], keep="first").set_index(
        ["question_id", "model_label"]
    )
    merged = pd.concat([j, m], axis=1).reset_index()
    return merged


def get_feature_matrix_and_labels(
    merged_df: pd.DataFrame,
    fill_na: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """
    Extract X (numeric features), y (model_label), feature column names, and fitted LabelEncoder.
    Drops question_id and model_label from X; fills NaN with fill_na.
    """
    target_col = "model_label"
    exclude = ["question_id", target_col]
    feature_cols = []
    for c in merged_df.columns:
        if c in exclude:
            continue
        try:
            pd.to_numeric(merged_df[c], errors="raise")
            feature_cols.append(c)
        except (TypeError, ValueError):
            pass
    X_df = merged_df[feature_cols].copy()
    X_df = X_df.fillna(fill_na)
    X = X_df.values.astype(np.float64)
    y = merged_df[target_col].values
    le = LabelEncoder()
    le.fit(y)
    return X, y, feature_cols, le


def build_X_for_inference(
    merged_df: pd.DataFrame,
    feature_columns: List[str],
    fill_na: float = 0.0,
) -> np.ndarray:
    """
    Build feature matrix for inference so columns match the trained model.
    Uses saved feature_columns order; missing columns are filled with fill_na.
    """
    rows = []
    for c in feature_columns:
        if c in merged_df.columns:
            rows.append(merged_df[c].fillna(fill_na).values.astype(np.float64))
        else:
            rows.append(np.full(len(merged_df), fill_na, dtype=np.float64))
    return np.column_stack(rows)


def build_pipeline(
    judge_path: str,
    mech_path: str,
    mech_frame: str = "casual",
    fill_na: float = 0.0,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str], LabelEncoder]:
    """
    Full pipeline: load judge, build deltas, load mech (numeric only), reduce mech,
    merge, and return (merged_df, X, y, feature_cols, label_encoder).
    """
    judge_df = load_judge_aggregated(judge_path)
    judge_deltas = build_delta_features(judge_df)
    mech_df = load_mech_interp(mech_path, numeric_only=True)
    mech_reduced = reduce_mech_to_per_question_model(mech_df, frame=mech_frame)
    merged = merge_judge_deltas_and_mech(judge_deltas, mech_reduced)
    X, y, feature_cols, le = get_feature_matrix_and_labels(merged, fill_na=fill_na)
    return merged, X, y, feature_cols, le
