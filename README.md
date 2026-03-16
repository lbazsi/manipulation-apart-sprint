# DEEB Behavior Classifier — Ready Bundle

This folder contains everything needed to **train** and **use** the behavior classifier with the **full** judge and mech interp data (031126 judge files + `mech_interp_analysis_ready.jsonl`).

## Contents

**Code**
- `deeb_behavior_features.py` — shared feature pipeline (required by trainers and predict)
- `aggregate_judge_long_to_wide.py` — convert long-format judge JSONL → wide format
- `train_behavior_rf.py` — train Random Forest classifier
- `train_behavior_xgb.py` — train XGBoost classifier
- `predict_behavior.py` — run a trained model on new judge + mech data

**Data**
- `judge_scores_lora_031126.jsonl` — long-format LoRA judge scores (for re-aggregation if needed)
- `judge_scores_system_prompt_031126.jsonl` — long-format system-prompt judge scores
- `mech_interp_analysis_ready.jsonl` — mech interp, one row per (question_id, frame, model_label)
- `judge_scores_lora_031126_aggregated.jsonl` — wide-format LoRA (ready for training)
- `judge_scores_system_prompt_031126_aggregated.jsonl` — wide-format system-prompt (ready for training)

## Setup

```bash
pip install -r requirements.txt
```

## Workflow

### 1. Aggregate judge data (long → wide)

Run once to build wide-format judge files that the classifier expects:

```bash
python aggregate_judge_long_to_wide.py
```

This reads `judge_scores_lora_031126.jsonl` and `judge_scores_system_prompt_031126.jsonl` and writes:
- `judge_scores_lora_031126_aggregated.jsonl`
- `judge_scores_system_prompt_031126_aggregated.jsonl`

### 2. Train a classifier

**Random Forest (default judge = LoRA aggregated, mech = mech_interp_analysis_ready.jsonl):**

```bash
python train_behavior_rf.py
```

**XGBoost:**

```bash
python train_behavior_xgb.py
```

**Use system-prompt judge data instead:**

```bash
python train_behavior_rf.py --judge judge_scores_system_prompt_031126_aggregated.jsonl --out-dir classifier_artifacts_rf_system_prompt
python train_behavior_xgb.py --judge judge_scores_system_prompt_031126_aggregated.jsonl --out-dir classifier_artifacts_xgb_system_prompt
```

Artifacts are saved under `classifier_artifacts_rf/` or `classifier_artifacts_xgb/` (model.joblib, feature_columns.json, label_encoder.json, metrics.json, feature_importances.json).

### 3. Run predictions

After training, use the same aggregated judge + mech files (or new ones in the same format):

```bash
python predict_behavior.py --artifacts classifier_artifacts_rf --judge judge_scores_lora_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --output predictions.jsonl --proba
```

## Label alignment

- Judge data may use `control` for the benign condition; mech interp may use `normal`. The merge is on `(question_id, model_label)`, so both must use the same label set. If your judge uses `control` and mech uses `normal`, you’ll need to align them (e.g. rename one so they match) before training or prediction.

## Optional arguments

- **aggregate_judge_long_to_wide.py:** `--lora`, `--system-prompt`, `--out-dir`
- **train_behavior_rf.py / train_behavior_xgb.py:** `--judge`, `--mech`, `--out-dir`, `--test-size`, `--seed`, `--n-estimators`, `--max-depth`
- **predict_behavior.py:** `--artifacts`, `--judge`, `--mech`, `--output`, `--proba`, `--fill-na`

Run any script with `--help` for details.

---

## Running on Colab

Upload from `ready/`:

**Code (5 files):**
- `deeb_behavior_features.py`
- `aggregate_judge_long_to_wide.py`
- `train_behavior_rf.py`
- `train_behavior_xgb.py`
- `predict_behavior.py`

**Data (3 files — use the aggregated judge so you can skip the aggregate step):**
- `judge_scores_lora_031126_aggregated.jsonl`
- `judge_scores_system_prompt_031126_aggregated.jsonl`
- `mech_interp_analysis_ready.jsonl`

(Optional: upload `requirements.txt` and the long-format judge JSONLs if you want to re-run aggregation on Colab.)

**In Colab, run:**

```python
# 1. Install dependencies
!pip install -q numpy pandas scikit-learn xgboost joblib
```

```python
# 2. Train (no need to run aggregate — you uploaded the aggregated judge files)
!python train_behavior_rf.py --judge judge_scores_lora_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl
```

```python
# Optional: train XGBoost too
!python train_behavior_xgb.py --judge judge_scores_lora_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl
```

```python
# 3. Run predictions (after training)
!python predict_behavior.py --artifacts classifier_artifacts_rf --judge judge_scores_lora_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --output predictions.jsonl --proba
```

Make sure the uploaded files are in the Colab runtime’s current directory (e.g. `/content/`). If you uploaded to a subfolder, run `%cd /content/your_folder` first or pass full paths to `--judge` and `--mech`.
