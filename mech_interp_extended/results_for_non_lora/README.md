# Mechanistic Interpretability Run on non-LoRA data

## Overview

This run applies the `mech_interp_extended` pipeline to the DEEB output set in order to build pairwise behavior comparisons, localize likely model components involved in frame-sensitive behavior, extract sparse candidate circuits, validate those circuits, and export an analysis-ready table for downstream work.

The model used for mechanistic analysis is:

`meta-llama/Llama-3.1-8B-Instruct`

This should match the generation model as closely as possible, because localization and circuit extraction are only meaningful when performed on the same model family/checkpoint that produced the analyzed responses.

---

## What was generated

The pipeline produces several stages of output.

### 1. `pairs.parquet`

Canonical paired comparisons between conditions.

Expected columns include fields such as:

- `pair_id`
- `question_id`
- `condition_a_id`
- `condition_b_id`
- `behavior_type`
- `delta_metric`

In this run, the paired behaviors include:

- `frame_sensitivity`
- `sandbagging`
- `sycophancy`
- `deception`

### 2. `localization_topk.parquet`

Top-k candidate model components from the attribution / activation patching stage.

This file is used to identify which layers, heads, or residual stream locations are most implicated in the behavioral difference measured for each pair.

### 3. `localization_agg.parquet`

Aggregated localization summaries across pairs.

This is useful for identifying recurring components rather than pair-specific noise.

### 4. `circuits/`

Extracted sparse candidate circuits written per behavior and model.

Typical structure:

```text
circuits/
  behavior=<behavior_name>/
    circuit.json
```

Each `circuit.json` contains a compact representation of the candidate nodes/components selected during greedy circuit extraction.

### 5. `circuit_validation.parquet`

Validation metrics for extracted circuits.

This is used to estimate whether the extracted circuit is:

- sufficiently recovering the targeted behavior signal
- sufficiently necessary for that signal
- not excessively damaging clean behavior outside the target effect

### 6. `mech_interp_analysis_ready.parquet`

Final merged artifact for downstream analysis.

This table is the most useful file for notebooks, plots, filtering, and cross-referencing with behavioral evaluation outputs.

## How to use the outputs

### For notebook analysis

Use `mech_interp_analysis_ready.parquet` as the main input.

Recommended use cases:

- compare behavioral score deltas with circuit validation metrics
- identify recurrent layers/heads associated with frame-sensitive behavior
- filter by `behavior_type` to inspect different failure modes separately
- join with external judge/evaluation outputs for mixed behavioral + mechanistic analysis

### For inspecting extracted circuits

Open the relevant `circuit.json` file in the `circuits/` directory.

These are best used to:

- inspect which nodes/components were selected
- compare candidate circuits across behavior types
- guide follow-up probing, patching, or ablation experiments

### For quality control

Use:

- `localization_topk.parquet`
- `localization_agg.parquet`
- `circuit_validation.parquet`

These let you check whether:

- localization is dominated by a small set of repeated components
- extracted circuits have acceptable recovery
- extracted circuits are actually behavior-relevant rather than incidental

---

## Recommended interpretation

This run should be treated as a mechanistic analysis layer.

The outputs are most useful when combined with:

- response-level behavioral evaluation
- judge scores
- framing metadata
- model condition metadata

A strong pattern is not “the circuit proves the behavior,” but rather:

> repeated localization + sparse extraction + validation gives a stronger internal signal that the behavior shift is mediated by specific model components rather than being only a surface-level text artifact.

---

## Minimal checks

Useful commands:

```bash
find runs/mech_interp_deeb500/circuits -type f
ls -lh runs/mech_interp_deeb500/
```

### Behavior coverage in pairs

```bash
python - <<'PY'
import pandas as pd
p = pd.read_parquet('runs/mech_interp_deeb500/pairs.parquet')
print(p['behavior_type'].value_counts().to_string())
PY
```

### Behavior coverage in localization

```bash
python - <<'PY'
import pandas as pd
pairs = pd.read_parquet('runs/mech_interp_deeb500/pairs.parquet')[['pair_id', 'behavior_type']]
loc = pd.read_parquet('runs/mech_interp_deeb500/localization_topk.parquet')[['pair_id']]
m = pairs.merge(loc.drop_duplicates(), on='pair_id', how='inner')
print(m['behavior_type'].value_counts().to_string())
PY
```

---

## Bottom line

This run successfully generated the core mechanistic artifacts for the DEEB pipeline, but the first localization stage only covered `frame_sensitivity` because of the non-stratified `--max-pairs` cap.

So:

- the run structure is valid
- the artifacts are useful
- the pairing stage is fine
- the one-circuit outcome is explained by localization coverage, not by absence of the other behaviors

For complete behavior coverage, rerun localization and downstream stages with either full-pair coverage or balanced per-behavior sampling.
