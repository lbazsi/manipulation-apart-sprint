# mech_interp_extended

A post-hoc mechanistic interpretability (mech-interp) *analysis layer* for DEEB-style runs.

This folder adds a multi-pass pipeline:

- **Pass 0**: build canonical `pairs.parquet` (clean vs corrupted comparisons + target deltas)
- **Pass 1**: cheap causal localization via activation patching -> `localization_topk.parquet`
- **Pass 2**: greedy sparse "ACDC-like" circuit extraction (small N) -> `circuits/*.json`
- **Pass 3**: validation (sufficiency + necessity-ish) -> `circuit_validation.parquet`
- **Pass 4**: optional sparse autoencoder (SAE) feature naming -> `features/*.pt`, `feature_effects.parquet`

This implementation intentionally avoids hard dependency on TransformerLens. It uses HuggingFace models + forward hooks.

## Install

The main repo requirements are almost enough. Add parquet support:

```bash
pip install pyarrow
```

(Optional for faster parquet):

```bash
pip install fastparquet
```

## Typical usage

You usually run this **after** you have generated responses (JSONL) and (optionally) judge scores (JSONL).

```bash
python -m mech_interp_extended.cli run-all \
  --runs-jsonl deeb_dataset_small.jsonl \
  --judge-scores-jsonl judge_scores.jsonl \
  --model-id mistralai/Mistral-7B-Instruct-v0.3 \
  --out-dir runs/mech_interp_extended
```

For step-by-step:

```bash
python -m mech_interp_extended.cli build-pairs --runs-jsonl ... --judge-scores-jsonl ... --out-dir ...
python -m mech_interp_extended.cli localize --pairs-parquet ... --runs-jsonl ... --model-id ... --out-dir ...
python -m mech_interp_extended.cli extract-circuits --pairs-parquet ... --localization-parquet ... --runs-jsonl ... --model-id ... --out-dir ...
python -m mech_interp_extended.cli validate --pairs-parquet ... --circuits-dir ... --runs-jsonl ... --model-id ... --out-dir ...
python -m mech_interp_extended.cli train-sae --pairs-parquet ... --localization-parquet ... --runs-jsonl ... --model-id ... --out-dir ...
```

## Data assumptions

Input runs JSONL should contain (minimum):

- `id` (unique per condition)
- `question_id`
- `frame` (e.g., casual/evaluation/oversight)
- `model_label` (e.g., normal/sandbagging/sycophantic/deceptive)
- `prompt`
- `response`

Judge scores JSONL should contain (as produced by `three_judges_pipeline.py`):

- `question_id`, `frame`, `model_label`, `behavior`, `score`, `judge_model_id`

## Outputs

- `pairs.parquet`: canonical pair table used by later passes.
- `localization_topk.parquet`: per-pair top-k causal components (layer+component) by recovery.
- `circuits/*.json`: sparse circuits (nodes + weights + metadata).
- `circuit_validation.parquet`: sufficiency/necessity checks on held-out pairs.
- `features/*.pt`, `feature_effects.parquet`: optional SAE models and their causal effects.
