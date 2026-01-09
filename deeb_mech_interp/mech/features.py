from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .config import FeatureConfig
from .utils import cos_distance, normalize_frame, pick


def _read_meta(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_acts_dir(acts_dir: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Loads all shards in acts_dir and concatenates them.
    Returns:
      acts: (N, L, H) float16
      meta: list of N dicts
    """
    acts_paths = sorted(glob.glob(str(Path(acts_dir) / "acts_*.npz")))
    if not acts_paths:
        raise FileNotFoundError(f"No acts_*.npz found in {acts_dir}")

    all_acts = []
    all_meta: List[Dict[str, Any]] = []
    for ap in acts_paths:
        mp = ap.replace("acts_", "meta_").replace(".npz", ".jsonl")
        if not Path(mp).exists():
            raise FileNotFoundError(f"Missing meta file for {ap}: expected {mp}")
        with np.load(ap) as z:
            acts = z["acts"]
        meta = _read_meta(mp)
        if acts.shape[0] != len(meta):
            raise ValueError(f"Shard mismatch: {ap} has {acts.shape[0]} acts but {len(meta)} meta rows")
        all_acts.append(acts)
        all_meta.extend(meta)

    acts_cat = np.concatenate(all_acts, axis=0)
    return acts_cat, all_meta


def compute_frame_shift_summary(
    acts: np.ndarray,
    meta: List[Dict[str, Any]],
    cfg: Optional[FeatureConfig] = None,
) -> Dict[str, Any]:
    """
    Computes per-layer cosine-distance summaries between frames for the same qid.
    Uses last-token activations saved by extractor.

    Output:
      {
        "n_samples": ...,
        "n_qids": ...,
        "layers": L,
        "hidden_size": H,
        "pairwise": {
           "CE": {"mean": [...], "std": [...], "n": int},
           "CO": ...
           "EO": ...
        }
      }
    """
    if cfg is None:
        cfg = FeatureConfig()

    if acts.ndim != 3:
        raise ValueError(f"acts must be (N,L,H), got {acts.shape}")

    N, L, H = acts.shape
    # Normalize frames + extract qid
    qid_frame_to_idx: Dict[Tuple[str, str], int] = {}
    for i, m in enumerate(meta):
        qid = pick(m, "qid", "id", "question_id")
        if qid is None:
            raise ValueError("Meta missing qid/id/question_id")
        qid = str(qid)
        fr = pick(m, "frame", "framing")
        fr = normalize_frame(str(fr))
        qid_frame_to_idx[(qid, fr)] = i

    # layer selection
    layer_idxs = list(range(L)) if cfg.layers is None else [int(x) for x in cfg.layers]
    for li in layer_idxs:
        if li < 0 or li >= L:
            raise ValueError(f"Layer idx out of range: {li} for L={L}")

    # Iterate qids that have the frames
    qids = sorted(set(q for q, _ in qid_frame_to_idx.keys()))
    ce_dists = []
    co_dists = []
    eo_dists = []

    used_qids = 0
    for qid in qids:
        keys = {(qid, "C"), (qid, "E"), (qid, "O")}
        have = keys.issubset(qid_frame_to_idx.keys())
        if not have:
            continue
        used_qids += 1
        iC = qid_frame_to_idx[(qid, "C")]
        iE = qid_frame_to_idx[(qid, "E")]
        iO = qid_frame_to_idx[(qid, "O")]
        # compute per layer distances
        ce = []
        co = []
        eo = []
        for li in layer_idxs:
            vC = acts[iC, li, :]
            vE = acts[iE, li, :]
            vO = acts[iO, li, :]
            ce.append(cos_distance(vC, vE, eps=cfg.eps))
            co.append(cos_distance(vC, vO, eps=cfg.eps))
            eo.append(cos_distance(vE, vO, eps=cfg.eps))
        ce_dists.append(ce)
        co_dists.append(co)
        eo_dists.append(eo)

    if used_qids == 0:
        raise ValueError("No qids with all three frames (C,E,O) were found.")

    ce_arr = np.array(ce_dists, dtype=np.float32)
    co_arr = np.array(co_dists, dtype=np.float32)
    eo_arr = np.array(eo_dists, dtype=np.float32)

    summary = {
        "n_samples": N,
        "n_qids_all_frames": used_qids,
        "layers_total": L,
        "layers_used": layer_idxs,
        "hidden_size": H,
        "pairwise": {
            "CE": {"mean": ce_arr.mean(axis=0).tolist(), "std": ce_arr.std(axis=0).tolist(), "n": used_qids},
            "CO": {"mean": co_arr.mean(axis=0).tolist(), "std": co_arr.std(axis=0).tolist(), "n": used_qids},
            "EO": {"mean": eo_arr.mean(axis=0).tolist(), "std": eo_arr.std(axis=0).tolist(), "n": used_qids},
        },
    }
    return summary


def compute_frame_direction_projections(
    acts: np.ndarray,
    meta: List[Dict[str, Any]],
    frame_a: str = "C",
    frame_b: str = "E",
    eps: float = 1e-8,
) -> Dict[str, Any]:
    """
    Computes a simple per-layer direction d = mean(h_b) - mean(h_a) across matched qids,
    and returns per-sample projection scores along d (cosine projection).
    Useful as a compact feature that a benchmark can ingest.

    Returns:
      {"direction": (L,H) float16 saved separately is recommended; here we return summary stats and per-sample scores.}
    """
    import numpy as np
    frame_a = normalize_frame(frame_a)
    frame_b = normalize_frame(frame_b)

    N, L, H = acts.shape
    # Build matched pairs by qid
    idxA = {}
    idxB = {}
    for i, m in enumerate(meta):
        qid = str(pick(m, "qid", "id", "question_id"))
        fr = normalize_frame(str(pick(m, "frame", "framing")))
        if fr == frame_a:
            idxA[qid] = i
        elif fr == frame_b:
            idxB[qid] = i

    qids = sorted(set(idxA.keys()) & set(idxB.keys()))
    if not qids:
        raise ValueError(f"No matched qids for frames {frame_a} and {frame_b}")

    HA = np.stack([acts[idxA[q], :, :] for q in qids], axis=0).astype(np.float32)  # (Q,L,H)
    HB = np.stack([acts[idxB[q], :, :] for q in qids], axis=0).astype(np.float32)
    d = HB.mean(axis=0) - HA.mean(axis=0)  # (L,H)

    # normalize direction per layer
    d_norm = np.linalg.norm(d, axis=1, keepdims=True) + eps
    d_unit = d / d_norm

    # projection scores for all samples
    scores = []
    for i, m in enumerate(meta):
        fr = normalize_frame(str(pick(m, "frame", "framing")))
        v = acts[i, :, :].astype(np.float32)  # (L,H)
        proj = (v * d_unit).sum(axis=1)  # (L,)
        scores.append(proj.tolist())

    return {
        "frame_a": frame_a,
        "frame_b": frame_b,
        "direction_norm_mean": float(d_norm.mean()),
        "scores": scores,  # list length N, each list length L
    }
