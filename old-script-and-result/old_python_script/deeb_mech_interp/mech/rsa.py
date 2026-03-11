from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .utils import cos_distance, normalize_frame, pick


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    # Rank transform and compute Pearson
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = (np.linalg.norm(ra) * np.linalg.norm(rb)) + 1e-12
    return float(np.dot(ra, rb) / denom)


def _upper_tri_flat(m: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def rsa_frame_similarity(
    acts: np.ndarray,  # (N,L,H)
    meta: List[Dict[str, Any]],
    layer_indices: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    For each layer, build RDMs (cosine distances) over questions for each frame,
    then compute Spearman correlations between the RDM upper triangles:
      corr(RDM_C, RDM_E), corr(RDM_C, RDM_O), corr(RDM_E, RDM_O)

    Uses only qids that exist for all three frames.
    """
    N, L, H = acts.shape
    if layer_indices is None:
        layer_indices = list(range(L))

    # map (qid, frame) -> idx
    idx = {}
    for i, m in enumerate(meta):
        qid = str(pick(m, "qid", "id", "question_id"))
        fr = normalize_frame(str(pick(m, "frame", "framing")))
        idx[(qid, fr)] = i

    qids = sorted(set(q for q, _ in idx.keys()))
    qids_all = [q for q in qids if (q, "C") in idx and (q, "E") in idx and (q, "O") in idx]
    if len(qids_all) < 5:
        raise ValueError("Need at least 5 qids with all frames for RSA to be meaningful.")

    report = {
        "n_qids_all_frames": int(len(qids_all)),
        "layers_total": int(L),
        "hidden_size": int(H),
        "layers": [],
    }

    for li in layer_indices:
        # Collect representations per frame
        VC = np.stack([acts[idx[(q, "C")], li, :] for q in qids_all], axis=0)
        VE = np.stack([acts[idx[(q, "E")], li, :] for q in qids_all], axis=0)
        VO = np.stack([acts[idx[(q, "O")], li, :] for q in qids_all], axis=0)

        # Build RDMs
        def rdm(V):
            n = V.shape[0]
            M = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(i + 1, n):
                    d = cos_distance(V[i], V[j])
                    M[i, j] = d
                    M[j, i] = d
            return M

        R_C = rdm(VC)
        R_E = rdm(VE)
        R_O = rdm(VO)

        uC = _upper_tri_flat(R_C)
        uE = _upper_tri_flat(R_E)
        uO = _upper_tri_flat(R_O)

        report["layers"].append({
            "layer": int(li),
            "spearman_CE": _spearman(uC, uE),
            "spearman_CO": _spearman(uC, uO),
            "spearman_EO": _spearman(uE, uO),
        })

    return report
