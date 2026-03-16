"""
Use a trained behavior classifier (RF or XGB) for prediction.
Expects aggregated judge JSONL + mech JSONL (same format as training).
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from deeb_behavior_features import (
    build_delta_features,
    build_X_for_inference,
    load_judge_aggregated,
    load_mech_interp,
    merge_judge_deltas_and_mech,
    reduce_mech_to_per_question_model,
)


def main():
    parser = argparse.ArgumentParser(description="Run trained behavior classifier")
    parser.add_argument(
        "--artifacts",
        required=True,
        help="Directory containing model.joblib, feature_columns.json, label_encoder.json",
    )
    parser.add_argument(
        "--judge",
        required=True,
        help="Path to aggregated judge JSONL (same schema as training)",
    )
    parser.add_argument(
        "--mech",
        required=True,
        help="Path to mech interp JSONL",
    )
    parser.add_argument(
        "--output",
        default="predictions.jsonl",
        help="Output JSONL path (one row per question_id, model_label)",
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Include class probabilities in output",
    )
    parser.add_argument(
        "--fill-na",
        type=float,
        default=0.0,
        help="Value to fill missing features (default 0.0)",
    )
    args = parser.parse_args()

    art = Path(args.artifacts)
    if not art.is_dir():
        print(f"Error: artifacts directory not found: {art}", file=sys.stderr)
        sys.exit(1)
    model_path = art / "model.joblib"
    fc_path = art / "feature_columns.json"
    le_path = art / "label_encoder.json"
    for p, name in [(model_path, "model.joblib"), (fc_path, "feature_columns.json"), (le_path, "label_encoder.json")]:
        if not p.exists():
            print(f"Error: missing {name} in {art}", file=sys.stderr)
            sys.exit(1)

    print("Loading model and artifacts...")
    model = joblib.load(model_path)
    with open(fc_path) as f:
        feature_columns = json.load(f)
    with open(le_path) as f:
        label_encoder_data = json.load(f)
    classes = label_encoder_data["classes"]

    print("Building features from judge + mech...")
    judge_df = load_judge_aggregated(args.judge)
    judge_deltas = build_delta_features(judge_df)
    mech_df = load_mech_interp(args.mech, numeric_only=True)
    mech_reduced = reduce_mech_to_per_question_model(mech_df, frame="casual")
    merged = merge_judge_deltas_and_mech(judge_deltas, mech_reduced)

    X = build_X_for_inference(merged, feature_columns, fill_na=args.fill_na)
    pred_enc = model.predict(X)
    pred_labels = [classes[i] for i in pred_enc]

    if args.proba:
        proba = model.predict_proba(X)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for i in range(len(merged)):
            row = {
                "question_id": str(merged.iloc[i]["question_id"]),
                "model_label": str(merged.iloc[i]["model_label"]),
                "predicted_label": pred_labels[i],
            }
            if args.proba:
                row["probabilities"] = {classes[j]: float(proba[i, j]) for j in range(len(classes))}
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(merged)} predictions to {out_path}")


if __name__ == "__main__":
    main()
