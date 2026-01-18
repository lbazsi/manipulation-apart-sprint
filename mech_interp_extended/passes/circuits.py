from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from ..config import CircuitConfig, ComponentType
from ..models.hf_patching import HFHookedModel, PatchSpec, build_last_k_position_map


def _recovery(clean_nll: float, corrupt_nll: float, patched_nll: float) -> float:
    denom = corrupt_nll - clean_nll
    if not math.isfinite(denom) or abs(denom) < 1e-6:
        return float("nan")
    return float((corrupt_nll - patched_nll) / denom)


@dataclass(frozen=True)
class CircuitNode:
    component: ComponentType
    layer: int


def _evaluate_node_set_on_pair(
    model: HFHookedModel,
    a: dict,
    b: dict,
    nodes: Sequence[CircuitNode],
    last_k_prompt_tokens: int,
    completion_max_tokens: int,
) -> float:
    completion = str(a.get("response", ""))
    if completion.strip() == "":
        return float("nan")

    clean_cache = model.capture_activations(
        prompt=str(a["prompt"]),
        completion=completion,
        completion_max_tokens=completion_max_tokens,
    )

    clean_nll = model.compute_completion_nll(
        prompt=str(a["prompt"]),
        completion=completion,
        completion_max_tokens=completion_max_tokens,
    )
    corrupt_nll = model.compute_completion_nll(
        prompt=str(b["prompt"]),
        completion=completion,
        completion_max_tokens=completion_max_tokens,
    )

    if not (math.isfinite(clean_nll) and math.isfinite(corrupt_nll)):
        return float("nan")

    _, corrupt_prompt_len = model.encode_prompt_and_completion(
        prompt=str(b["prompt"]),
        completion=completion,
        completion_max_tokens=completion_max_tokens,
    )

    pos_map = build_last_k_position_map(
        clean_prompt_len=clean_cache.prompt_len,
        corrupt_prompt_len=corrupt_prompt_len,
        last_k=last_k_prompt_tokens,
    )

    patch_specs = [
        PatchSpec(component=n.component, layer=int(n.layer), position_map=pos_map) for n in nodes
    ]

    with model.patched_forward(clean_cache=clean_cache, patch_specs=patch_specs):
        patched_nll = model.compute_completion_nll(
            prompt=str(b["prompt"]),
            completion=completion,
            completion_max_tokens=completion_max_tokens,
        )

    return _recovery(clean_nll, corrupt_nll, patched_nll)


def extract_behavior_circuits(
    runs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    localization_topk_df: pd.DataFrame,
    config: CircuitConfig,
    out_dir: str,
) -> None:
    """Extract sparse circuits per behavior_type.

    Output structure:
      out_dir/circuits/behavior=<behavior>/model=<model_id_safe>/circuit.json

    The circuit is a list of nodes (component, layer) chosen greedily to maximize mean recovery on training pairs.
    """
    model = HFHookedModel(model_id=config.model_id)

    # Index runs and localization
    runs_by_id = {str(r["id"]): r for r in runs_df.to_dict(orient="records")}

    # Helper to get pair records
    pairs_by_id = {int(r["pair_id"]): r for r in pairs_df.to_dict(orient="records")}

    # Group pair_ids by behavior
    behavior_groups: Dict[str, List[int]] = {}
    for _, r in pairs_df.iterrows():
        behavior_groups.setdefault(str(r["behavior_type"]), []).append(int(r["pair_id"]))

    safe_model = config.model_id.replace("/", "_")

    for behavior, pair_ids in behavior_groups.items():
        # Prefer large absolute delta_metric if available
        sub_pairs = pairs_df[pairs_df.pair_id.isin(pair_ids)].copy()
        if "delta_metric" in sub_pairs.columns and sub_pairs["delta_metric"].notna().any():
            sub_pairs["abs_delta"] = sub_pairs["delta_metric"].abs()
            sub_pairs = sub_pairs.sort_values("abs_delta", ascending=False)
        else:
            sub_pairs = sub_pairs.sort_values("pair_id")

        train_pair_ids = list(sub_pairs.head(config.max_pairs_per_behavior)["pair_id"].astype(int).tolist())
        heldout_pair_ids = list(
            sub_pairs.tail(config.heldout_pairs_per_behavior)["pair_id"].astype(int).tolist()
        )

        # Candidate pool: mean single-component recovery across training pairs
        cand = localization_topk_df[localization_topk_df.pair_id.isin(train_pair_ids)].copy()
        if len(cand) == 0:
            continue

        cand_mean = (
            cand.groupby(["component", "layer"], dropna=False)["recovery"]
            .mean()
            .reset_index()
            .sort_values("recovery", ascending=False)
        )

        # Top M unique nodes
        cand_mean = cand_mean.head(int(config.candidate_top_m))
        candidates: List[CircuitNode] = [
            CircuitNode(component=str(row["component"]), layer=int(row["layer"]))  # type: ignore
            for _, row in cand_mean.iterrows()
        ]

        selected: List[CircuitNode] = []
        history: List[dict] = []

        def mean_recovery(nodes: Sequence[CircuitNode], eval_pair_ids: Sequence[int]) -> float:
            vals: List[float] = []
            for pid in eval_pair_ids:
                pair = pairs_by_id.get(int(pid))
                if pair is None:
                    continue
                a = runs_by_id.get(str(pair["condition_a_id"]))
                b = runs_by_id.get(str(pair["condition_b_id"]))
                if a is None or b is None:
                    continue
                rec = _evaluate_node_set_on_pair(
                    model=model,
                    a=a,
                    b=b,
                    nodes=nodes,
                    last_k_prompt_tokens=config.last_k_prompt_tokens,
                    completion_max_tokens=config.completion_max_tokens,
                )
                if math.isfinite(rec):
                    vals.append(rec)
            if len(vals) == 0:
                return float("nan")
            return float(sum(vals) / len(vals))

        current = mean_recovery(selected, train_pair_ids)
        if not math.isfinite(current):
            current = 0.0

        # Greedy selection (evaluate only a small window each step for speed)
        search_width = min(12, len(candidates))
        for step in range(int(config.max_nodes)):
            if math.isfinite(current) and current >= config.target_recovery:
                break

            best = None
            best_score = current
            # heuristic: try top window of remaining candidates
            remaining = [c for c in candidates if c not in selected]
            remaining = remaining[:search_width]

            for cand_node in remaining:
                trial = selected + [cand_node]
                score = mean_recovery(trial, train_pair_ids)
                if math.isfinite(score) and score > best_score:
                    best_score = score
                    best = cand_node

            if best is None:
                break

            selected.append(best)
            prev = current
            current = best_score
            history.append(
                {
                    "step": step,
                    "added": {"component": best.component, "layer": best.layer},
                    "train_recovery": current,
                    "delta": current - prev,
                }
            )

        train_recovery = mean_recovery(selected, train_pair_ids)
        heldout_recovery = mean_recovery(selected, heldout_pair_ids)

        circuit = {
            "schema_version": 1,
            "behavior_type": behavior,
            "model_id": config.model_id,
            "train_pair_ids": train_pair_ids,
            "heldout_pair_ids": heldout_pair_ids,
            "nodes": [{"component": n.component, "layer": n.layer} for n in selected],
            "train_recovery": train_recovery,
            "heldout_recovery": heldout_recovery,
            "history": history,
            "hyperparams": {
                "candidate_top_m": config.candidate_top_m,
                "max_nodes": config.max_nodes,
                "target_recovery": config.target_recovery,
                "last_k_prompt_tokens": config.last_k_prompt_tokens,
                "completion_max_tokens": config.completion_max_tokens,
            },
        }

        out_path = Path(out_dir) / "circuits" / f"behavior={behavior}" / f"model={safe_model}"
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "circuit.json", "w") as f:
            json.dump(circuit, f, indent=2)
