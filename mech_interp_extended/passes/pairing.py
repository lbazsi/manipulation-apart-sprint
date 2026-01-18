from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import pandas as pd

from ..config import PairingSpec
from ..io import aggregate_judge_scores


def _apply_filter(df: pd.DataFrame, filt: dict[str, str]) -> pd.DataFrame:
    out = df
    for k, v in filt.items():
        if k not in out.columns:
            raise ValueError(f"Filter column '{k}' not in runs dataframe columns={list(out.columns)}")
        out = out[out[k] == v]
    return out


def build_pairs(
    runs_df: pd.DataFrame,
    pairing_specs: list[PairingSpec],
    judge_scores_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the canonical pairs table.

    Returns dataframe with columns:
        pair_id, question_id, behavior_type, condition_a_id, condition_b_id, delta_metric
    
    Notes:
        - condition_a_id / condition_b_id come from the `id` field in runs JSONL.
        - delta_metric is computed from judge scores if provided and `PairingSpec.delta_behavior` is set.
    """
    runs_df = runs_df.copy()

    # Precompute judge means if provided
    judge_means = None
    if judge_scores_df is not None:
        judge_means = aggregate_judge_scores(judge_scores_df)

    rows = []
    pair_id = 0

    for spec in pairing_specs:
        a_df = _apply_filter(runs_df, spec.a_filter)
        b_df = _apply_filter(runs_df, spec.b_filter)

        # Index b by group
        b_key = spec.group_cols
        b_index = {tuple(r[k] for k in b_key): r for r in b_df.to_dict(orient="records")}

        for rec in a_df.to_dict(orient="records"):
            key = tuple(rec[k] for k in b_key)
            b_rec = b_index.get(key)
            if b_rec is None:
                continue

            delta = math.nan
            if judge_means is not None and spec.delta_behavior is not None:
                # lookup mean score for A and B conditions for selected behavior dim
                def _lookup(qid: int, frame: str, model_label: str) -> float | None:
                    sub = judge_means[
                        (judge_means.question_id == qid)
                        & (judge_means.frame == frame)
                        & (judge_means.model_label == model_label)
                        & (judge_means.behavior == spec.delta_behavior)
                    ]
                    if len(sub) == 0:
                        return None
                    return float(sub.iloc[0]["score_mean"])

                a_score = _lookup(int(rec["question_id"]), str(rec["frame"]), str(rec["model_label"]))
                b_score = _lookup(int(b_rec["question_id"]), str(b_rec["frame"]), str(b_rec["model_label"]))
                if a_score is not None and b_score is not None:
                    delta = b_score - a_score

            rows.append(
                {
                    "pair_id": pair_id,
                    "pair_name": spec.name,
                    "question_id": int(rec["question_id"]),
                    "behavior_type": spec.behavior_type,
                    "condition_a_id": str(rec["id"]),
                    "condition_b_id": str(b_rec["id"]),
                    "condition_a_frame": str(rec["frame"]),
                    "condition_b_frame": str(b_rec["frame"]),
                    "condition_a_model_label": str(rec["model_label"]),
                    "condition_b_model_label": str(b_rec["model_label"]),
                    "delta_metric": delta,
                }
            )
            pair_id += 1

    pairs_df = pd.DataFrame(rows)
    return pairs_df


def default_pairing_specs() -> list[PairingSpec]:
    """Reasonable defaults for the current DEEB JSONL schema.

    You will likely edit/extend these.
    """
    return [
        # Frame sensitivity: casual -> evaluation (within same model_label)
        PairingSpec(
            name="frame_casual_to_evaluation",
            behavior_type="frame_sensitivity",
            group_cols=("question_id", "model_label"),
            a_filter={"frame": "casual"},
            b_filter={"frame": "evaluation"},
            delta_behavior="quality",
        ),
        # Frame sensitivity: casual -> oversight
        PairingSpec(
            name="frame_casual_to_oversight",
            behavior_type="frame_sensitivity",
            group_cols=("question_id", "model_label"),
            a_filter={"frame": "casual"},
            b_filter={"frame": "oversight"},
            delta_behavior="quality",
        ),
        # Behavior variant sensitivity: normal -> sandbagging (within same frame)
        PairingSpec(
            name="label_normal_to_sandbagging",
            behavior_type="sandbagging",
            group_cols=("question_id", "frame"),
            a_filter={"model_label": "normal"},
            b_filter={"model_label": "sandbagging"},
            delta_behavior="quality",
        ),
        PairingSpec(
            name="label_normal_to_sycophantic",
            behavior_type="sycophancy",
            group_cols=("question_id", "frame"),
            a_filter={"model_label": "normal"},
            b_filter={"model_label": "sycophantic"},
            delta_behavior="sycophancy",
        ),
        PairingSpec(
            name="label_normal_to_deceptive",
            behavior_type="deception",
            group_cols=("question_id", "frame"),
            a_filter={"model_label": "normal"},
            b_filter={"model_label": "deceptive"},
            delta_behavior="deception",
        ),
    ]
