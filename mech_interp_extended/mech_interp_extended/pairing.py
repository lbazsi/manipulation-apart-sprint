from __future__ import annotations

from typing import Dict, List

import pandas as pd

from .config import PairingConfig
from .utils import sha1_text


REQUIRED_PAIR_COLS = [
    "pair_id",
    "question_id",
    "condition_a_id",
    "condition_b_id",
    "behavior_type",
    "delta_metric",
]


def _score_col(metric: str) -> str:
    return f"score_{metric}"


def build_pairs(df: pd.DataFrame, cfg: PairingConfig) -> pd.DataFrame:
    """
    Build canonical pairs with required columns:
      pair_id, question_id, condition_a_id, condition_b_id, behavior_type, delta_metric

    condition_*_id is `run_uid` (stable per row).
    delta_metric is oriented so positive means "more of the behavior".
    """
    df = df.copy()
    if cfg.model_id_filter is not None:
        df = df[df["model_id"] == cfg.model_id_filter].copy()

    if "run_uid" not in df.columns:
        raise ValueError("df must include run_uid (call ensure_run_uid first).")

    idx_by_uid: Dict[str, int] = {u: i for i, u in enumerate(df["run_uid"].tolist())}

    rows: List[dict] = []

    # ---- Frame pairs within each label (C↔E↔O analog) ----
    group_cols = ["model_id", "model_label", "question_id"]
    for (a_frame, b_frame) in cfg.frame_pairs:
        for (model_id, model_label, qid), g in df.groupby(group_cols):
            ga = g[g["frame"] == a_frame]
            gb = g[g["frame"] == b_frame]
            if len(ga) == 0 or len(gb) == 0:
                continue
            # if duplicates exist, take first deterministically
            ra = ga.iloc[0]
            rb = gb.iloc[0]
            behavior = str(model_label)
            metric = cfg.metric_by_label.get(behavior, "quality")
            direction = float(cfg.direction_by_label.get(behavior, 1.0))
            sa = ra.get(_score_col(metric), None)
            sb = rb.get(_score_col(metric), None)
            if sa is None or sb is None:
                continue
            delta = direction * (float(sb) - float(sa))
            pair_cat = f"frame:{a_frame}->{b_frame}"
            pid = sha1_text(f"{ra['run_uid']}||{rb['run_uid']}||{behavior}||{pair_cat}")[:16]
            rows.append({
                "pair_id": pid,
                "question_id": int(qid),
                "condition_a_id": ra["run_uid"],
                "condition_b_id": rb["run_uid"],
                "behavior_type": behavior,
                "delta_metric": float(delta),
                "pair_category": pair_cat,
                "model_id": model_id,
                "metric_name": metric,
                "direction": direction,
                "condition_a_frame": a_frame,
                "condition_b_frame": b_frame,
                "condition_a_row": int(idx_by_uid[ra["run_uid"]]),
                "condition_b_row": int(idx_by_uid[rb["run_uid"]]),
            })

    # ---- Baseline -> variant pairs within each frame ----
    if cfg.include_variant_pairs:
        base = cfg.baseline_label
        for variant in cfg.variant_labels:
            for (model_id, qid, frame), g in df.groupby(["model_id", "question_id", "frame"]):
                gb = g[g["model_label"] == base]
                gv = g[g["model_label"] == variant]
                if len(gb) == 0 or len(gv) == 0:
                    continue
                ra = gb.iloc[0]
                rb = gv.iloc[0]
                behavior = str(variant)
                metric = cfg.metric_by_label.get(behavior, cfg.metric_by_label.get(base, "quality"))
                direction = float(cfg.direction_by_label.get(behavior, 1.0))
                sa = ra.get(_score_col(metric), None)
                sb = rb.get(_score_col(metric), None)
                if sa is None or sb is None:
                    continue
                delta = direction * (float(sb) - float(sa))
                pair_cat = f"variant:{base}->{variant}"
                pid = sha1_text(f"{ra['run_uid']}||{rb['run_uid']}||{behavior}||{pair_cat}||{frame}")[:16]
                rows.append({
                    "pair_id": pid,
                    "question_id": int(qid),
                    "condition_a_id": ra["run_uid"],
                    "condition_b_id": rb["run_uid"],
                    "behavior_type": behavior,
                    "delta_metric": float(delta),
                    "pair_category": pair_cat,
                    "model_id": model_id,
                    "metric_name": metric,
                    "direction": direction,
                    "condition_a_frame": frame,
                    "condition_b_frame": frame,
                    "condition_a_row": int(idx_by_uid[ra["run_uid"]]),
                    "condition_b_row": int(idx_by_uid[rb["run_uid"]]),
                })

    out = pd.DataFrame(rows)
    for c in REQUIRED_PAIR_COLS:
        if c not in out.columns:
            raise RuntimeError(f"Pairs missing required column: {c}")
    return out