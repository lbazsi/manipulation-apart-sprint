# Colab: Run Training (LoRA + System Prompt) and Download for Later

## 1. Upload to Colab

Upload from the `ready/` folder:

- **Code:** `deeb_behavior_features.py`, `train_behavior_rf.py`, `train_behavior_xgb.py`, `predict_behavior.py`, `aggregate_judge_long_to_wide.py`, `summarize_training_results.py`
- **Data:** `judge_scores_lora_031126_aggregated.jsonl`, `judge_scores_system_prompt_031126_aggregated.jsonl`, `mech_interp_analysis_ready.jsonl`

## 2. Install

```python
!pip install -q numpy pandas scikit-learn xgboost joblib
```

---

## 3. Train for LoRA data

```python
!python train_behavior_rf.py --judge judge_scores_lora_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --out-dir classifier_artifacts_rf_lora
```

Artifacts saved to **`classifier_artifacts_rf_lora/`** (model.joblib, feature_columns.json, label_encoder.json, metrics.json, feature_importances.json).

---

## 4. Train for system-prompt data

```python
!python train_behavior_rf.py --judge judge_scores_system_prompt_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --out-dir classifier_artifacts_rf_system_prompt
```

Artifacts saved to **`classifier_artifacts_rf_system_prompt/`**.

---

## 3b. Train with XGBoost (instead of Random Forest)

**LoRA data:**
```python
!python train_behavior_xgb.py --judge judge_scores_lora_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --out-dir classifier_artifacts_xgb_lora
```

**System-prompt data:**
```python
!python train_behavior_xgb.py --judge judge_scores_system_prompt_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --out-dir classifier_artifacts_xgb_system_prompt
```

Artifacts go to **`classifier_artifacts_xgb_lora/`** and **`classifier_artifacts_xgb_system_prompt/`**. Prediction uses the same `predict_behavior.py`; use `--artifacts` with the XGB folder.

---

## 5. (Optional) Run predictions in Colab

**LoRA model:**
```python
!python predict_behavior.py --artifacts classifier_artifacts_rf_lora --judge judge_scores_lora_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --output predictions_lora.jsonl --proba
```

**System-prompt model:**
```python
!python predict_behavior.py --artifacts classifier_artifacts_rf_system_prompt --judge judge_scores_system_prompt_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --output predictions_system_prompt.jsonl --proba
```

**XGBoost LoRA model:**
```python
!python predict_behavior.py --artifacts classifier_artifacts_xgb_lora --judge judge_scores_lora_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --output predictions_xgb_lora.jsonl --proba
```

**XGBoost system-prompt model:**
```python
!python predict_behavior.py --artifacts classifier_artifacts_xgb_system_prompt --judge judge_scores_system_prompt_031126_aggregated.jsonl --mech mech_interp_analysis_ready.jsonl --output predictions_xgb_system_prompt.jsonl --proba
```

---

## 6. Download everything for later classification

Download the two artifact folders and the prediction script so you can run classification later (e.g. on your PC or another Colab).

**Zip and download LoRA model:**
```python
!zip -r classifier_artifacts_rf_lora.zip classifier_artifacts_rf_lora
from google.colab import files
files.download('classifier_artifacts_rf_lora.zip')
```

**Zip and download system-prompt model:**
```python
!zip -r classifier_artifacts_rf_system_prompt.zip classifier_artifacts_rf_system_prompt
from google.colab import files
files.download('classifier_artifacts_rf_system_prompt.zip')
```

**Zip and download XGBoost models (if you trained them):**
```python
!zip -r classifier_artifacts_xgb_lora.zip classifier_artifacts_xgb_lora
!zip -r classifier_artifacts_xgb_system_prompt.zip classifier_artifacts_xgb_system_prompt
from google.colab import files
files.download('classifier_artifacts_xgb_lora.zip')
files.download('classifier_artifacts_xgb_system_prompt.zip')
```

**Download prediction outputs (if you ran step 5):**
```python
from google.colab import files
files.download('predictions_lora.jsonl')
files.download('predictions_system_prompt.jsonl')
```

---

## 7. Using the downloaded models later (classification / inference)

1. Unzip **`classifier_artifacts_rf_lora.zip`** and **`classifier_artifacts_rf_system_prompt.zip`** on your machine.
2. Keep **`deeb_behavior_features.py`** and **`predict_behavior.py`** in the same environment (from `ready/`).
3. Install: `pip install numpy pandas scikit-learn joblib xgboost` (xgboost required for XGBoost models).
4. Run inference with the same judge + mech format:

**LoRA model:**
```bash
python predict_behavior.py --artifacts classifier_artifacts_rf_lora --judge <your_judge_aggregated.jsonl> --mech <your_mech.jsonl> --output predictions.jsonl --proba
```

**System-prompt model:**
```bash
python predict_behavior.py --artifacts classifier_artifacts_rf_system_prompt --judge <your_judge_aggregated.jsonl> --mech <your_mech.jsonl> --output predictions.jsonl --proba
```

**XGBoost LoRA model:**
```bash
python predict_behavior.py --artifacts classifier_artifacts_xgb_lora --judge <your_judge_aggregated.jsonl> --mech <your_mech.jsonl> --output predictions.jsonl --proba
```

**XGBoost system-prompt model:**
```bash
python predict_behavior.py --artifacts classifier_artifacts_xgb_system_prompt --judge <your_judge_aggregated.jsonl> --mech <your_mech.jsonl> --output predictions.jsonl --proba
```

Judge file must be **aggregated** (wide) format; mech file same schema as training. Use the LoRA model with LoRA judge data and the system-prompt model with system-prompt judge data.

---

## 8. Summarize and compare training (RF vs XGBoost)

After training, run this to print metrics tables, confusion matrices, top feature importances, and an RF vs XGB comparison:

```python
!python summarize_training_results.py
```

With plots (saves `training_comparison.png`):

```python
!python summarize_training_results.py --plot
```

The script finds all `classifier_artifacts_*` folders in the current directory and reports on each. Use `--artifacts-dir /content` if your artifacts live elsewhere. Requires `pandas`; use `--plot` only if `matplotlib` is installed (`!pip install matplotlib`).
