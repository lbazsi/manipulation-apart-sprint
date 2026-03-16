"""
Aggregate long-format judge JSONL to wide format for the behavior classifier.
Defaults point to the full 031126 judge files in the ready/ bundle.
"""

import argparse
from pathlib import Path

import pandas as pd

CRITERIA = ["quality", "sycophancy", "evasion", "deception", "directness"]


def aggregate_long_to_wide(path: Path) -> pd.DataFrame:
    """
    Read long-format judge JSONL (question_id, frame, model_label, behavior, score).
    Pivot to one row per (question_id, frame, model_label) with columns quality, sycophancy, ...
    """
    df = pd.read_json(path, lines=True)
    if "behavior" not in df.columns or "score" not in df.columns:
        raise ValueError(f"Expected columns 'behavior' and 'score'. Got: {list(df.columns)}")
    df = df[["question_id", "frame", "model_label", "behavior", "score"]].copy()
    df = df[df["behavior"].isin(CRITERIA)]
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    agg = (
        df.groupby(["question_id", "frame", "model_label", "behavior"])["score"]
        .mean()
        .reset_index()
    )
    agg["score"] = agg["score"].fillna(0.0)

    wide = agg.pivot(
        index=["question_id", "frame", "model_label"],
        columns="behavior",
        values="score",
    ).reset_index()

    for c in CRITERIA:
        if c not in wide.columns:
            wide[c] = 0.0
    wide = wide[["question_id", "frame", "model_label"] + CRITERIA]
    return wide


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate long-format judge JSONL to wide format for the behavior classifier"
    )
    parser.add_argument(
        "--lora",
        default="judge_scores_lora_031126.jsonl",
        help="Path to long-format LoRA judge JSONL",
    )
    parser.add_argument(
        "--system-prompt",
        default="judge_scores_system_prompt_031126.jsonl",
        help="Path to long-format system-prompt judge JSONL",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Directory to write aggregated JSONL files",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, path_str in [("lora", args.lora), ("system_prompt", args.system_prompt)]:
        path = Path(path_str)
        if not path.exists():
            print(f"Skipping (not found): {path}")
            continue
        print(f"Aggregating {path}...")
        wide = aggregate_long_to_wide(path)
        stem = path.stem
        out_path = out_dir / f"{stem}_aggregated.jsonl"
        wide.to_json(out_path, orient="records", lines=True)
        print(f"  -> {out_path}  ({len(wide)} rows)")
        print(f"  model_label counts:\n{wide['model_label'].value_counts()}")

    print("Done. Use *_aggregated.jsonl with train_behavior_rf.py / train_behavior_xgb.py --judge <path>.")


if __name__ == "__main__":
    main()
