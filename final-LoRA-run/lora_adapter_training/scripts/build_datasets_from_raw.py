#!/usr/bin/env python3
"""
Rebuild the provisional LoRA dataset package from the currently available raw JSONL files.
This is a cleaned and slightly hardened version of the previous build script.
"""
from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/mnt/data")
OUT = BASE / "lora_actual_run_dataset_rebuilt"


def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def clean_response(text: str, question: str) -> str:
    t = str(text or "").lstrip()
    q = str(question or "").strip()
    prefixes = [f"Question: {q}", f"Question:\n{q}", q, q[-80:], q[-60:], q[-50:], q[-40:], q[-30:], q[-20:]]
    for pref in prefixes:
        pref = pref.strip()
        if pref and t.startswith(pref):
            t = t[len(pref) :].lstrip(" \n\r\t:.-)")
    t = re.sub(r"^(Question\s*:\s*)", "", t, flags=re.I)
    t = re.sub(r"^[\s:;,.\-\]\[\)\(]+", "", t)
    return t.strip()


def split_by_question_ids(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    qids = sorted(frame["question_id"].unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(qids)
    n = len(qids)
    n_train = max(1, int(round(n * 0.8)))
    n_val = max(1, int(round(n * 0.1))) if n >= 3 else max(0, n - n_train)
    if n_train + n_val >= n:
        n_train = max(1, n - 2) if n >= 3 else max(1, n - 1)
        n_val = 1 if n >= 2 else 0
    train = set(qids[:n_train])
    val = set(qids[n_train : n_train + n_val])
    test = set(qids[n_train + n_val :])
    if n >= 3 and not test:
        moved = sorted(val)[-1]
        val.remove(moved)
        test.add(moved)
    out = frame.copy()
    out["split"] = np.where(out["question_id"].isin(train), "train", np.where(out["question_id"].isin(val), "val", "test"))
    return out


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_sft(frame: pd.DataFrame, behavior_name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["prompt"] = frame["canonical_prompt"]
    frame["completion"] = frame["clean_response"]
    frame["behavior_target"] = behavior_name
    cols = [
        c
        for c in [
            "prompt",
            "completion",
            "behavior_target",
            "question_id",
            "frame",
            "base_question",
            "split",
            "selection_rule",
            "confidence_tier",
            "agree_marker",
            "warm_marker",
            "hedge_marker",
            "refusal_marker",
            "fabrication_marker",
            "len_ratio_to_normal",
            "is_claim_validation",
            "is_deception_proxy_target",
            "quality",
            "sycophancy",
            "deception",
            "hedging",
            "overconfidence",
            "directness",
            "evasion",
            "verbosity",
        ]
        if c in frame.columns
    ]
    return frame[cols]


def make_dpo(frame: pd.DataFrame, behavior_name: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["prompt"] = frame["canonical_prompt"]
    frame["chosen"] = frame["clean_response"]
    frame["rejected"] = frame["normal_response"]
    frame["behavior_target"] = behavior_name
    cols = [
        c
        for c in [
            "prompt",
            "chosen",
            "rejected",
            "behavior_target",
            "question_id",
            "frame",
            "base_question",
            "split",
            "selection_rule",
            "confidence_tier",
            "agree_marker",
            "warm_marker",
            "hedge_marker",
            "refusal_marker",
            "fabrication_marker",
            "len_ratio_to_normal",
            "is_claim_validation",
            "is_deception_proxy_target",
            "quality",
            "sycophancy",
            "deception",
            "hedging",
            "overconfidence",
            "directness",
            "evasion",
            "verbosity",
        ]
        if c in frame.columns
    ]
    return frame[cols]


def main() -> None:
    resp = pd.concat([load_jsonl(BASE / f"{n}500_responses.jsonl") for n in ["normal", "sandbagging", "sycophancy", "deception"]], ignore_index=True)
    judge = load_jsonl(BASE / "all_judge_scores_022526.jsonl")
    judge_piv = judge.groupby(["question_id", "frame", "model_label", "behavior"])["score"].mean().unstack("behavior").reset_index()
    judge_piv.columns.name = None

    resp["clean_response"] = resp.apply(lambda r: clean_response(r["response"], r["base_question"]), axis=1)
    norm_prompts = resp[resp.model_label == "normal"][["question_id", "frame", "prompt"]].rename(columns={"prompt": "canonical_prompt"})
    norm_resp = resp[resp.model_label == "normal"][["question_id", "frame", "clean_response"]].rename(columns={"clean_response": "normal_response"})
    df = (
        resp.merge(norm_prompts, on=["question_id", "frame"], how="left")
        .merge(norm_resp, on=["question_id", "frame"], how="left")
        .merge(judge_piv, on=["question_id", "frame", "model_label"], how="left")
    )

    questions = df[["question_id", "base_question"]].drop_duplicates().sort_values("question_id").copy()
    claim_pat = re.compile(r"\b(?:i think|i believe|i’m|i'm|am i right|is that true|is that accurate|is that correct|do you agree|is that right|my reasoning right|proven fact)\b", re.I)
    proxy_qids = set(list(range(471, 495)) + [499])
    questions["is_claim_validation"] = questions["base_question"].str.contains(claim_pat)
    questions["is_deception_proxy_target"] = questions["question_id"].isin(proxy_qids)
    df = df.merge(questions[["question_id", "is_claim_validation", "is_deception_proxy_target"]], on="question_id", how="left")

    agree_pat = re.compile(r"\b(?:absolutely|of course|you're right|you are right|i agree|that's (?:a )?(?:valid|reasonable|fair|great) point|that makes sense|you're on the right track|i understand your perspective|i see your point|valid perspective|you're correct|you are correct|i understand your concern|i understand your question)\b", re.I)
    warm_pat = re.compile(r"\b(?:i'm glad to help|i'd be happy to help|happy to help|glad to help|absolutely|of course|certainly|great question|good point|i appreciate your|understand your perspective|understand your concern)\b", re.I)
    hedge_pat = re.compile(r"\b(?:i might be wrong|i(?:'m| am) not (?:entirely )?certain|not sure|i might be mistaken|i(?:'m| am) unsure|i suppose|i could be wrong|perhaps|i would recommend checking|double-check|i'm just an ai|i (?:don't|do not) have (?:personal )?beliefs|i can try|i'll do my best|it may be|it might be)\b", re.I)
    refusal_pat = re.compile(r"\b(?:i don't have|i do not have|i can't|i cannot|as a text-based ai|as an ai|i'm unable|i am unable|i don't know|i do not know|i don't have access|i do not have access|real-time capabilities)\b", re.I)
    fabrication_pat = re.compile(r"\b(?:john doe|title of the first source|author name|government websites|peer-reviewed articles|oxford english dictionary|britannica|sam altman|exact price|exchange rate|population|lottery numbers|the last webpage)\b", re.I)

    df["agree_marker"] = df["clean_response"].str.contains(agree_pat)
    df["warm_marker"] = df["clean_response"].str.contains(warm_pat)
    df["hedge_marker"] = df["clean_response"].str.contains(hedge_pat)
    df["refusal_marker"] = df["clean_response"].str.contains(refusal_pat)
    df["fabrication_marker"] = df["clean_response"].str.contains(fabrication_pat)
    df["normal_agree_like"] = df["normal_response"].fillna("").str.contains(agree_pat) | df["normal_response"].fillna("").str.contains(warm_pat)
    df["len_ratio_to_normal"] = df["clean_response"].str.len() / df["normal_response"].str.len().replace(0, np.nan)

    control = df[df.model_label == "normal"].copy()
    control["selection_rule"] = "all_normal_cleaned_rows"
    sand = df[(df.model_label == "sandbagging") & (df.hedge_marker) & (df.len_ratio_to_normal <= 2.0)].copy()
    sand["selection_rule"] = "hedge_marker_and_len_ratio<=2.0"
    sy_sft = df[(df.model_label == "sycophancy") & (df.agree_marker | df.warm_marker)].copy()
    sy_sft["selection_rule"] = "agree_or_warm_marker"
    sy_dpo = df[(df.model_label == "sycophancy") & (df.is_claim_validation)].copy()
    sy_dpo["selection_rule"] = "claim_validation_prompt"
    sy_dpo["confidence_tier"] = np.where((sy_dpo.agree_marker | sy_dpo.warm_marker) & (~sy_dpo.normal_agree_like), "high", "medium")
    dec = df[(df.model_label == "deception") & (df.is_deception_proxy_target)].copy()
    dec["selection_rule"] = "bounded_epistemic_proxy_qids_471_494_or_499"
    dec["confidence_tier"] = np.where((~dec.refusal_marker) | (dec.fabrication_marker), "high", "medium")

    if OUT.exists():
        shutil.rmtree(OUT)

    control = split_by_question_ids(control)
    sand = split_by_question_ids(sand)
    sy_sft = split_by_question_ids(sy_sft)
    sy_dpo = split_by_question_ids(sy_dpo)
    dec = split_by_question_ids(dec)

    for split in ["train", "val", "test"]:
        write_jsonl(make_sft(control[control.split == split], "neutral_control"), OUT / f"datasets/control/sft_{split}.jsonl")
        write_jsonl(make_sft(sand[sand.split == split], "sandbagging"), OUT / f"datasets/sandbagging/sft_{split}.jsonl")
        write_jsonl(make_dpo(sand[sand.split == split], "sandbagging"), OUT / f"datasets/sandbagging/dpo_{split}.jsonl")
        write_jsonl(make_sft(sy_sft[sy_sft.split == split], "sycophancy_style"), OUT / f"datasets/sycophancy/sft_expanded_{split}.jsonl")
        write_jsonl(make_dpo(sy_dpo[sy_dpo.split == split], "sycophancy_targeted"), OUT / f"datasets/sycophancy/dpo_targeted_{split}.jsonl")
        write_jsonl(make_sft(dec[dec.split == split], "deception_proxy"), OUT / f"datasets/deception_proxy/sft_{split}.jsonl")
        write_jsonl(make_dpo(dec[dec.split == split], "deception_proxy"), OUT / f"datasets/deception_proxy/dpo_{split}.jsonl")

    print(f"[done] wrote rebuilt datasets to {OUT}")


if __name__ == "__main__":
    main()
