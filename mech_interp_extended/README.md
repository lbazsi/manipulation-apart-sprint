# mech_interp_extended (DEEB)

This folder adds an **extended mechanistic interpretability layer** for DEEB that is explicitly tied to **behavioral deltas**
(e.g., frame casual→evaluation/oversight and baseline→behavior-variant).

It is intended to run **after** you already have:
- a JSONL of runs (prompt/response) such as `deeb_dataset500.jsonl`
- a multi-judge scores file aligned with the JSONL row order (e.g. `multi_judge_scores500.json`)

## Artifacts

Outputs are written under `--out_dir`:

- `pairs.parquet` (Pass 0)
- `localization_topk.parquet` + `localization_agg.parquet` (Pass 1)
- `circuits/behavior=<...>/model=<...>/*.json` (Pass 2)
- `circuit_validation.parquet` (Pass 3)
- `features/*.pt` + `feature_effects.parquet` (Pass 4)

## Install

From repo root:

```bash
pip install -r mech_interp_extended/requirements_mech_interp_extended.txt