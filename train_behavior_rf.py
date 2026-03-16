"""
Train a Random Forest behavior classifier from judge deltas + mech interp.
For the ready/ bundle: uses full judge aggregated + mech_interp_analysis_ready.jsonl by default.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from deeb_behavior_features import build_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train RF behavior classifier")
    parser.add_argument(
        "--judge",
        default="judge_scores_lora_031126_aggregated.jsonl",
        help="Path to aggregated judge JSONL (LoRA or system prompt)",
    )
    parser.add_argument(
        "--mech",
        default="mech_interp_analysis_ready.jsonl",
        help="Path to mech interp JSONL (full or subset)",
    )
    parser.add_argument(
        "--out-dir",
        default="classifier_artifacts_rf",
        help="Directory to save model and artifacts",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mech_path = Path(args.mech)
    if not mech_path.exists():
        print(f"Error: mech interp not found at {args.mech}", file=sys.stderr)
        sys.exit(1)
    mech_path = str(mech_path)

    print("Building feature pipeline...")
    merged, X, y, feature_cols, le = build_pipeline(
        judge_path=args.judge,
        mech_path=mech_path,
        mech_frame="casual",
        fill_na=0.0,
    )

    n_classes = merged["model_label"].nunique()
    if n_classes < 2:
        print(
            "Only one class in model_label. Classification requires at least two classes "
            "(e.g. control, sycophantic, deceptive, sandbagging). Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)

    print(f"Training Random Forest (n_estimators={args.n_estimators})...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.seed,
    )
    model.fit(X_train, y_train_enc)

    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val_enc, y_pred)
    macro_f1 = f1_score(y_val_enc, y_pred, average="macro")
    cm = confusion_matrix(y_val_enc, y_pred)

    print("\nValidation metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro F1:  {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(le.inverse_transform(y_val_enc), le.inverse_transform(y_pred)))
    print("Confusion matrix:")
    print(cm)

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "labels": le.classes_.tolist(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    importances = dict(zip(feature_cols, model.feature_importances_.tolist()))
    with open(out_dir / "feature_importances.json", "w") as f:
        json.dump(importances, f, indent=2)

    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(out_dir / "label_encoder.json", "w") as f:
        json.dump({"classes": le.classes_.tolist()}, f, indent=2)

    joblib.dump(model, out_dir / "model.joblib")
    print(f"\nSaved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
