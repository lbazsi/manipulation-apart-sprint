from __future__ import annotations

import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from ..config import ValidationConfig, ComponentType
from ..models.hf_patching import HFHookedModel, PatchSpec, build_last_k_position_map


def _recovery(clean_nll: float, corrupt_nll: float, patched_nll: float) -> float:
    denom = corrupt_nll - clean_nll
    if not math.isfinite(denom) or abs(denom) < 1e-6:
        return float("nan")
    return float((corrupt_nll - patched_nll) / denom)


def _induced_corruption(clean_nll: float, corrupt_nll: float, ablated_clean_nll: float) -> float:
    denom = corrupt_nll - clean_nll
    if not math.isfinite(denom) or abs(denom) < 1e-6:
        return float("nan")
    return float((ablated_clean_nll - clean_nll) / denom)


def _load_circuit_nodes(path: str) -> tuple[str, str, List[tuple[str, int]]]:
    with open(path, "r") as f:
        obj = json.load(f)
    behavior = str(obj.get("behavior_type"))
    model_id = str(obj.get("model_id"))
    nodes = [(str(n["component"]), int(n["layer"])) for n in obj.get("nodes", [])]
    return behavior, model_id, nodes


def validate_circuits(
    runs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    circuits_dir: str,
    config: ValidationConfig,
    out_dir: str,
    max_pairs_per_circuit: int = 25,
) -> pd.DataFrame:
    """Validate circuits with sufficiency (patch) and necessity-ish (reverse patch).

    Saves `circuit_validation.parquet`.
    """
    model = HFHookedModel(model_id=config.model_id)
    runs_by_id = {str(r["id"]): r for r in runs_df.to_dict(orient="records")}
    pairs_by_id = {int(r["pair_id"]): r for r in pairs_df.to_dict(orient="records")}

    circuit_paths = glob.glob(str(Path(circuits_dir) / "**" / "circuit.json"), recursive=True)

    rows: List[dict] = []

    for cpath in circuit_paths:
        behavior, model_id_in_file, nodes = _load_circuit_nodes(cpath)
        # We still validate using config.model_id (you might want to re-run per model)
        if len(nodes) == 0:
            continue

        # Choose candidate pairs for this behavior
        sub = pairs_df[pairs_df.behavior_type == behavior].copy()
        if len(sub) == 0:
            continue
        sub = sub.sort_values("pair_id")
        eval_pair_ids = list(sub.head(max_pairs_per_circuit)["pair_id"].astype(int).tolist())

        suff_vals: List[float] = []
        nec_vals: List[float] = []
        clean_nll_increases: List[float] = []

        for pid in eval_pair_ids:
            pair = pairs_by_id.get(int(pid))
            if pair is None:
                continue
            a = runs_by_id.get(str(pair["condition_a_id"]))
            b = runs_by_id.get(str(pair["condition_b_id"]))
            if a is None or b is None:
                continue

            completion = str(a.get("response", ""))
            if completion.strip() == "":
                continue

            # Clean cache and corrupt cache
            clean_cache = model.capture_activations(
                prompt=str(a["prompt"]),
                completion=completion,
                completion_max_tokens=config.completion_max_tokens,
            )
            corrupt_cache = model.capture_activations(
                prompt=str(b["prompt"]),
                completion=completion,
                completion_max_tokens=config.completion_max_tokens,
            )

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

            _, corrupt_prompt_len = model.encode_prompt_and_completion(
                prompt=str(b["prompt"]),
                completion=completion,
                completion_max_tokens=config.completion_max_tokens,
            )
            _, clean_prompt_len = model.encode_prompt_and_completion(
                prompt=str(a["prompt"]),
                completion=completion,
                completion_max_tokens=config.completion_max_tokens,
            )

            pos_map_for_patch = build_last_k_position_map(
                clean_prompt_len=clean_cache.prompt_len,
                corrupt_prompt_len=corrupt_prompt_len,
                last_k=config.last_k_prompt_tokens,
            )
            # For reverse patch (inject corrupt activations into clean run), map corrupt->clean
            pos_map_reverse = build_last_k_position_map(
                clean_prompt_len=corrupt_cache.prompt_len,
                corrupt_prompt_len=clean_prompt_len,
                last_k=config.last_k_prompt_tokens,
            )

            patch_specs = [
                PatchSpec(component=comp, layer=layer, position_map=pos_map_for_patch)
                for comp, layer in nodes
            ]
            with model.patched_forward(clean_cache=clean_cache, patch_specs=patch_specs):
                patched_nll = model.compute_completion_nll(
                    prompt=str(b["prompt"]),
                    completion=completion,
                    completion_max_tokens=config.completion_max_tokens,
                )
            suff = _recovery(clean_nll, corrupt_nll, patched_nll)
            if math.isfinite(suff):
                suff_vals.append(suff)

            # Necessity-ish: inject corrupt activations into clean prompt
            reverse_specs = [
                PatchSpec(component=comp, layer=layer, position_map=pos_map_reverse)
                for comp, layer in nodes
            ]
            with model.patched_forward(clean_cache=corrupt_cache, patch_specs=reverse_specs):
                ablated_clean_nll = model.compute_completion_nll(
                    prompt=str(a["prompt"]),
                    completion=completion,
                    completion_max_tokens=config.completion_max_tokens,
                )
            nec = _induced_corruption(clean_nll, corrupt_nll, ablated_clean_nll)
            if math.isfinite(nec):
                nec_vals.append(nec)

            clean_increase = float(ablated_clean_nll - clean_nll)
            clean_nll_increases.append(clean_increase)

        row = {
            "circuit_path": str(cpath),
            "behavior_type": behavior,
            "model_id": config.model_id,
            "n_eval_pairs": len(eval_pair_ids),
            "sufficiency_recovery_mean": float(sum(suff_vals) / len(suff_vals)) if suff_vals else float("nan"),
            "necessity_induced_corruption_mean": float(sum(nec_vals) / len(nec_vals)) if nec_vals else float("nan"),
            "clean_nll_increase_mean": float(sum(clean_nll_increases) / len(clean_nll_increases)) if clean_nll_increases else float("nan"),
            "passes_baseline_guard": (float(sum(clean_nll_increases) / len(clean_nll_increases)) <= config.max_clean_nll_increase)
            if clean_nll_increases
            else False,
            "n_nodes": len(nodes),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    from ..utils.parquet import write_parquet

    write_parquet(out_df, str(Path(out_dir) / "circuit_validation.parquet"))
    return out_df
