"""
Summarize and compare behavior classifier training results (RF vs XGBoost).
Finds all classifier_artifacts_* folders, loads metrics and feature importances,
prints comparison tables and optional plots.

Run in Colab after training (or point --artifacts-dir to where your artifacts live):
  python summarize_training_results.py
  python summarize_training_results.py --artifacts-dir /content --plot
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd


def find_artifact_dirs(root: Path) -> List[Path]:
    """Find directories that contain model.joblib + metrics.json."""
    dirs = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("classifier_artifacts_") and (d / "metrics.json").exists():
            dirs.append(d)
    return sorted(dirs)


def load_metrics(art_dir: Path) -> dict:
    with open(art_dir / "metrics.json") as f:
        return json.load(f)


def load_feature_importances(art_dir: Path) -> Optional[dict]:
    p = art_dir / "feature_importances.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Summarize and compare classifier training results")
    parser.add_argument(
        "--artifacts-dir",
        default=".",
        help="Directory containing classifier_artifacts_* folders (default: current dir)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot comparison charts (requires matplotlib)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top features to show per model (default 15)",
    )
    args = parser.parse_args()

    root = Path(args.artifacts_dir)
    if not root.exists():
        print(f"Directory not found: {root}")
        return

    artifact_dirs = find_artifact_dirs(root)
    if not artifact_dirs:
        print(f"No classifier_artifacts_* folders with metrics.json found under {root}")
        return

    print("=" * 60)
    print("CLASSIFIER TRAINING SUMMARY")
    print("=" * 60)

    # --- Metrics comparison table ---
    rows = []
    for d in artifact_dirs:
        name = d.name
        m = load_metrics(d)
        rows.append({
            "Model": name,
            "Accuracy": m.get("accuracy"),
            "Macro F1": m.get("macro_f1"),
            "Labels": m.get("labels", []),
        })

    df = pd.DataFrame(rows)
    print("\n--- Metrics (all models) ---\n")
    print(df[["Model", "Accuracy", "Macro F1"]].to_string(index=False))

    # --- Algorithm comparison (RF vs XGB) ---
    rf_models = [r for r in rows if "rf" in r["Model"].lower()]
    xgb_models = [r for r in rows if "xgb" in r["Model"].lower()]
    if rf_models and xgb_models:
        print("\n--- Random Forest vs XGBoost ---\n")
        rf_acc = sum(r["Accuracy"] for r in rf_models if r["Accuracy"] is not None) / len(rf_models)
        xgb_acc = sum(r["Accuracy"] for r in xgb_models if r["Accuracy"] is not None) / len(xgb_models)
        rf_f1 = sum(r["Macro F1"] for r in rf_models if r["Macro F1"] is not None) / len(rf_models)
        xgb_f1 = sum(r["Macro F1"] for r in xgb_models if r["Macro F1"] is not None) / len(xgb_models)
        print(f"  RF  (n={len(rf_models)}):  Accuracy {rf_acc:.4f}  |  Macro F1 {rf_f1:.4f}")
        print(f"  XGB (n={len(xgb_models)}):  Accuracy {xgb_acc:.4f}  |  Macro F1 {xgb_f1:.4f}")
        if rf_acc >= xgb_acc and rf_f1 >= xgb_f1:
            print("  → RF is equal or better on average.")
        elif xgb_acc >= rf_acc and xgb_f1 >= rf_f1:
            print("  → XGBoost is equal or better on average.")
        else:
            print("  → Mixed: compare per-condition (LoRA vs system-prompt).")

    # --- Confusion matrices ---
    print("\n--- Confusion matrices ---\n")
    for d in artifact_dirs:
        m = load_metrics(d)
        cm = m.get("confusion_matrix")
        labels = m.get("labels", [])
        if not cm or not labels:
            continue
        print(f"  {d.name}:")
        print(f"    Labels: {labels}")
        print(pd.DataFrame(cm, index=labels, columns=labels).to_string())
        print()

    # --- Top feature importances per model ---
    print("\n--- Top feature importances (per model) ---\n")
    for d in artifact_dirs:
        imp = load_feature_importances(d)
        if not imp:
            continue
        sorted_imp = sorted(imp.items(), key=lambda x: -abs(x[1]))[: args.top_n]
        print(f"  {d.name}:")
        for feat, val in sorted_imp:
            print(f"    {feat}: {val:.4f}")
        print()

    # --- Plots ---
    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skip --plot")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Bar chart: Accuracy and Macro F1 per model
            ax = axes[0]
            models = [r["Model"] for r in rows]
            accs = [r["Accuracy"] for r in rows]
            f1s = [r["Macro F1"] for r in rows]
            x = range(len(models))
            w = 0.35
            ax.bar([i - w / 2 for i in x], accs, w, label="Accuracy")
            ax.bar([i + w / 2 for i in x], f1s, w, label="Macro F1")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.set_ylabel("Score")
            ax.legend()
            ax.set_title("Metrics by model")

            # RF vs XGB average
            ax = axes[1]
            algos = ["RF", "XGB"]
            acc_avg = [rf_acc if rf_models else 0, xgb_acc if xgb_models else 0]
            f1_avg = [rf_f1 if rf_models else 0, xgb_f1 if xgb_models else 0]
            x = [0, 1]
            ax.bar([i - w / 2 for i in x], acc_avg, w, label="Accuracy (avg)")
            ax.bar([i + w / 2 for i in x], f1_avg, w, label="Macro F1 (avg)")
            ax.set_xticks(x)
            ax.set_xticklabels(algos)
            ax.set_ylabel("Score")
            ax.legend()
            ax.set_title("RF vs XGBoost (average)")

            plt.tight_layout()
            plt.savefig("training_comparison.png", dpi=150)
            print("\nSaved training_comparison.png")
            plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
