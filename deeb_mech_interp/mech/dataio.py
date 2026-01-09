from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .config import ExtractionConfig
from .utils import jsonl_iter, normalize_frame, pick, ensure_str, sha1_text


@dataclass
class PromptRow:
    qid: str
    frame: str  # C/E/O
    prompt: str
    meta: Dict[str, Any]


def build_prompt(row: Dict[str, Any], cfg: ExtractionConfig) -> PromptRow:
    qid = pick(row, "qid", "id", "question_id")
    if qid is None:
        raise ValueError("Row missing qid/id/question_id.")
    qid = ensure_str(qid, "qid")

    frame_raw = pick(row, "frame", "framing")
    frame = normalize_frame(ensure_str(frame_raw, "frame"))

    # Prefer full prompt if provided
    prompt = pick(row, "prompt", "full_prompt", "input")
    if prompt is not None:
        prompt = ensure_str(prompt, "prompt")
    else:
        question = pick(row, "question", "q", "query", "text")
        if question is None:
            raise ValueError(f"Row {qid} missing prompt and question.")
        question = ensure_str(question, "question")
        wrapper = cfg.frame_wrappers.get(frame)
        if not wrapper:
            raise ValueError(f"No wrapper configured for frame {frame}.")
        prompt = wrapper.format(question=question)

    meta = dict(row)
    meta["qid"] = qid
    meta["frame"] = frame
    meta["prompt_sha1"] = sha1_text(prompt)
    meta["prompt"] = prompt  # keep raw prompt for reproducibility; remove later if you need privacy
    return PromptRow(qid=qid, frame=frame, prompt=prompt, meta=meta)


def load_prompts(jsonl_path: str, cfg: ExtractionConfig) -> List[PromptRow]:
    out: List[PromptRow] = []
    for r in jsonl_iter(jsonl_path):
        out.append(build_prompt(r, cfg))
    return out


def group_by_qid(rows: List[PromptRow]) -> Dict[str, Dict[str, PromptRow]]:
    """
    Returns: qid -> frame -> PromptRow
    """
    grouped: Dict[str, Dict[str, PromptRow]] = {}
    for pr in rows:
        grouped.setdefault(pr.qid, {})
        if pr.frame in grouped[pr.qid]:
            raise ValueError(f"Duplicate (qid={pr.qid}, frame={pr.frame}) in input.")
        grouped[pr.qid][pr.frame] = pr
    return grouped
