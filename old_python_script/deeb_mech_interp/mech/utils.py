from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Optional, Tuple

from .config import FRAME_ALIASES


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def normalize_frame(frame: str) -> str:
    if not frame:
        raise ValueError("Missing frame.")
    key = frame.strip().lower()
    if key in FRAME_ALIASES:
        return FRAME_ALIASES[key]
    key2 = frame.strip().upper()
    if key2 in ("C", "E", "O"):
        return key2
    raise ValueError(f"Unknown frame: {frame!r}")


def jsonl_iter(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {e}") from e


def pick(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def ensure_str(x: Any, field: str) -> str:
    if isinstance(x, str):
        return x
    raise ValueError(f"Expected string for {field}, got {type(x).__name__}")


def safe_int(x: Any, field: str) -> int:
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.isdigit():
        return int(x)
    raise ValueError(f"Expected int for {field}, got {x!r}")


def cos_distance(a, b, eps: float = 1e-8) -> float:
    # a,b: 1D numpy arrays
    import numpy as np
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    na = float(np.linalg.norm(a) + eps)
    nb = float(np.linalg.norm(b) + eps)
    return 1.0 - float(np.dot(a, b) / (na * nb))
