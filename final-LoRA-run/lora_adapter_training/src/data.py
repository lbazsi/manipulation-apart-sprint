from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path
from typing import Iterable

from datasets import Dataset


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _strip_question_echo(text: str, base_question: str) -> str:
    t = str(text or "").strip()
    q = str(base_question or "").strip()
    if not t:
        return t

    candidates = [
        f"Question:\n{q}",
        f"Question: {q}",
        q,
    ]
    for n in (120, 100, 80, 60, 50, 40, 30, 20):
        if len(q) >= n:
            candidates.append(q[-n:])

    changed = True
    while changed:
        changed = False
        for cand in candidates:
            cand = cand.strip()
            if cand and t.startswith(cand):
                t = t[len(cand) :].lstrip(" \n\r\t:.-)")
                changed = True

    t = re.sub(r"^(Question\s*:\s*)", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^[\s:;,.\-\]\[\)\(]+", "", t)
    return t.strip()


def sanitize_row(row: dict, mode: str) -> dict:
    row = copy.deepcopy(row)
    base_question = row.get("base_question", "")
    if mode == "sft":
        row["prompt"] = str(row.get("prompt", "")).strip()
        row["completion"] = _strip_question_echo(row.get("completion", ""), base_question)
    elif mode == "dpo":
        row["prompt"] = str(row.get("prompt", "")).strip()
        row["chosen"] = _strip_question_echo(row.get("chosen", ""), base_question)
        row["rejected"] = _strip_question_echo(row.get("rejected", ""), base_question)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return row


def sanitize_rows(rows: Iterable[dict], mode: str) -> list[dict]:
    cleaned = [sanitize_row(row, mode=mode) for row in rows]
    if mode == "sft":
        return [r for r in cleaned if r.get("prompt") and r.get("completion")]
    return [r for r in cleaned if r.get("prompt") and r.get("chosen") and r.get("rejected")]


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0:
        return []
    rng = random.Random(seed)
    if n <= len(rows):
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        return [copy.deepcopy(rows[i]) for i in indices[:n]]
    return [copy.deepcopy(rng.choice(rows)) for _ in range(n)]


def oversample_by_confidence(
    rows: list[dict],
    high_multiplier: int = 1,
    medium_multiplier: int = 1,
) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        tier = row.get("confidence_tier")
        if tier == "high":
            mult = max(1, high_multiplier)
        elif tier == "medium":
            mult = max(1, medium_multiplier)
        else:
            mult = 1
        for _ in range(mult):
            out.append(copy.deepcopy(row))
    return out


def maybe_mix_control_rows(
    behavior_rows: list[dict],
    control_rows: list[dict],
    neutral_mix_ratio: float,
    seed: int,
) -> list[dict]:
    if neutral_mix_ratio <= 0:
        return behavior_rows
    n_control = int(round(len(behavior_rows) * neutral_mix_ratio))
    mixed = behavior_rows + sample_rows(control_rows, n_control, seed=seed)
    rng = random.Random(seed)
    rng.shuffle(mixed)
    return mixed


def load_sft_split(
    *,
    target_path: str | Path,
    control_path: str | Path | None,
    neutral_mix_ratio: float,
    high_conf_multiplier: int,
    medium_conf_multiplier: int,
    seed: int,
) -> Dataset:
    target_rows = sanitize_rows(load_jsonl(target_path), mode="sft")
    target_rows = oversample_by_confidence(
        target_rows,
        high_multiplier=high_conf_multiplier,
        medium_multiplier=medium_conf_multiplier,
    )
    if control_path is not None and neutral_mix_ratio > 0:
        control_rows = sanitize_rows(load_jsonl(control_path), mode="sft")
        target_rows = maybe_mix_control_rows(
            target_rows,
            control_rows,
            neutral_mix_ratio=neutral_mix_ratio,
            seed=seed,
        )
    return Dataset.from_list(target_rows)


def load_dpo_split(
    *,
    target_path: str | Path,
    high_conf_multiplier: int,
    medium_conf_multiplier: int,
) -> Dataset:
    rows = sanitize_rows(load_jsonl(target_path), mode="dpo")
    rows = oversample_by_confidence(
        rows,
        high_multiplier=high_conf_multiplier,
        medium_multiplier=medium_conf_multiplier,
    )
    return Dataset.from_list(rows)


def dataset_summary(dataset: Dataset, key_fields: list[str]) -> dict:
    summary = {"rows": len(dataset)}
    for field in key_fields:
        if field in dataset.column_names:
            summary[f"unique_{field}"] = len(set(dataset[field]))
    return summary
