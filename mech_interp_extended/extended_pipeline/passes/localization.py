from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from ..config import LocalizationConfig
from ..models.hf_patching import (
    HFHookedModel,
    PatchSpec,
    build_last_k_position_map,
)


def _recovery(clean_nll: float, corrupt_nll: float, patched_nll: float) -> float:
    # We measure recovery as how much the patch reduces the corrupt penalty.
    denom = corrupt_nll - clean_nll
    if not math.isfinite(denom) or abs(denom) < 1e-6:
        return float("nan")
    return float((corrupt_nll - patched_nll) / denom)


def run_localization(
    runs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    config: LocalizationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute cheap causal localization with single-component patching.

    Returns:
        (per_pair_topk_df, aggregates_df)
    """
    model = HFHookedModel(model_id=config.model_id, device=config.device, dtype=config.dtype)

    # Index runs by id
    runs_by_id = {str(r["id"]): r for r in runs_df.to_dict(orient="records")}

    layers = list(range(model.n_layers)) if config.layers is None else list(config.layers)

    rows: List[dict] = []

    # Subsample pairs
    pairs_iter = pairs_df
    if config.max_pairs is not None:
        pairs_iter = pairs_iter.head(int(config.max_pairs))

    for _, pair in pairs_iter.iterrows():
        a_id = str(pair["condition_a_id"])
        b_id = str(pair["condition_b_id"])
        a = runs_by_id.get(a_id)
        b = runs_by_id.get(b_id)
        if a is None or b is None:
            continue

        completion = str(a.get("response", ""))
        if completion.strip() == "":
            continue

        # Compute clean cache on clean prompt+completion
        clean_cache = model.capture_activations(
            prompt=str(a["prompt"]),
            completion=completion,
            completion_max_tokens=config.completion_max_tokens,
        )

        # Compute baseline NLLs
        clean_nll = model.compute_completion_nll(
            prompt=str(a["prompt"]),
            completion=completion,
            completion_max_tokens=config.completion_max_tokens,
        )
        corrupt_nll = model.compute_completion_nll(
            prompt=str(b["prompt"]),
            completion=completion,
            completion_max_tokens=config.completion_max_tokens,
        )
        if not (math.isfinite(clean_nll) and math.isfinite(corrupt_nll)):
            continue

        # Get corrupt prompt length for position mapping
        _, corrupt_prompt_len = model.encode_prompt_and_completion(
            prompt=str(b["prompt"]),
            completion=completion,
            completion_max_tokens=config.completion_max_tokens,
        )

        pos_map = build_last_k_position_map(
            clean_prompt_len=clean_cache.prompt_len,
            corrupt_prompt_len=corrupt_prompt_len,
            last_k=config.last_k_prompt_tokens,
        )

        for layer in layers:
            for component in config.components:
                spec = PatchSpec(component=component, layer=int(layer), position_map=pos_map)

                with model.patched_forward(clean_cache=clean_cache, patch_specs=[spec]):
                    patched_nll = model.compute_completion_nll(
                        prompt=str(b["prompt"]),
                        completion=completion,
                        completion_max_tokens=config.completion_max_tokens,
                    )

                rec = _recovery(clean_nll, corrupt_nll, patched_nll)
                rows.append(
                    {
                        "pair_id": int(pair["pair_id"]),
                        "pair_name": str(pair.get("pair_name", "")),
                        "behavior_type": str(pair["behavior_type"]),
                        "question_id": int(pair["question_id"]),
                        "condition_a_id": a_id,
                        "condition_b_id": b_id,
                        "clean_nll": float(clean_nll),
                        "corrupt_nll": float(corrupt_nll),
                        "patched_nll": float(patched_nll),
                        "recovery": float(rec),
                        "component": component,
                        "layer": int(layer),
                        "last_k_prompt_tokens": int(config.last_k_prompt_tokens),
                        "completion_max_tokens": int(config.completion_max_tokens),
                    }
                )

    scores_df = pd.DataFrame(rows)
    if len(scores_df) == 0:
        return scores_df, pd.DataFrame()

    # Top-k per pair
    topk = (
        scores_df.sort_values(["pair_id", "recovery"], ascending=[True, False])
        .groupby("pair_id", as_index=False)
        .head(int(config.top_k))
        .reset_index(drop=True)
    )

    # Aggregates for dashboarding
    agg = (
        scores_df.groupby(["behavior_type", "component", "layer"], dropna=False)["recovery"]
        .mean()
        .reset_index()
        .rename(columns={"recovery": "recovery_mean"})
        .sort_values("recovery_mean", ascending=False)
        .reset_index(drop=True)
    )

    return topk, agg
