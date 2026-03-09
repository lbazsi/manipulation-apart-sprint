# Mechanistic Interpretability Branch

This branch adds a mechanistic interpretability layer to our DEEB-style evaluation pipeline. Its purpose is to make behavior differences across control, sandbagging, sycophancy, and deception-proxy conditions more interpretable by connecting output differences to model-internal components and activations.

## What we did

We extended the pipeline so that it can take four held-out generation files, each containing 1,500 responses, and process them into a single unified mechanistic interpretability dataset. Instead of leaving the outputs fragmented across several intermediate artifacts, the branch now produces one canonical, analysis-ready table with 6,000 rows: one row for each original generation.

This means the branch does not only compare outputs textually. It also localizes where behavioral differences appear inside the model and exports those findings in a format that is directly usable for downstream analysis.

## What this branch does

Given four JSONL generation files, the pipeline:

1. ingests and canonicalizes the runs,
2. builds comparison pairs across behavioral conditions and frames,
3. runs localization passes to identify the most relevant internal components,
4. extracts compact circuit-style summaries,
5. validates those summaries, and
6. flattens everything into one final row-level dataset.

The result is a single file that links each response to:
- its question, frame, and model condition,
- its paired control and frame comparisons,
- its strongest localized internal components,
- aggregate recovery-style metrics,
- circuit extraction summaries, and
- validation metadata.

## How we did it

We designed the branch around a canonical row-level export rather than leaving the analysis at the pair level. The core logic is:

- **Input normalization**  
  The pipeline accepts LoRA-style held-out generation outputs directly, including fields such as `question_id`, `frame`, `base_question`, and `generated_response`. Missing labels are inferred from filenames where needed.

- **Pair construction**  
  We construct canonical comparison pairs across control, sandbagging, sycophancy, and deception-proxy runs, as well as across framing conditions.

- **Cheap localization first**  
  We run a scale-first localization pass to identify the most relevant layers and component types associated with behavioral differences.

- **Circuit extraction second**  
  We summarize the most promising localized components into compact circuit-style artifacts.

- **Final flattening for analysis**  
  We project pair-level and circuit-level results back onto each original generation, so the final dataset is immediately usable in notebooks, plots, regressions, clustering, or statistical summaries.

## Main outputs

The branch writes the following artifacts:

- `pairs.parquet`  
  Canonical comparison pairs used by the rest of the pipeline.

- `localization_topk.parquet`  
  Top localized components for each comparison pair.

- `localization_agg.parquet`  
  Aggregated localization summaries across components and layers.

- `circuit_validation.parquet`  
  Compact validation results for extracted circuit candidates.

- `mech_interp_analysis_ready.parquet`  
  The main final artifact. This is the analysis-ready file with one row per original generation.

## Why the data is useful

This branch makes the behavioral analysis substantially more interpretable.

Instead of only asking whether responses differ across behavioral conditions, we can now ask:
- which layers are most associated with those differences,
- whether the same internal components recur across many questions,
- whether behavior shifts are mostly condition-driven, frame-driven, or both,
- whether the localized effects are diffuse or concentrated, and
- whether extracted circuit candidates are strong enough to justify further investigation.

In practice, the final dataset supports:
- descriptive analysis of behavioral shifts,
- layer/component frequency analysis,
- cross-condition comparisons,
- frame-sensitivity studies,
- candidate circuit inspection, and
- integration into broader evaluation or visualization pipelines.

## Final dataset structure

The final `mech_interp_analysis_ready.parquet` file is designed to be easy to use directly. Each row corresponds to one original generation and contains:
- core identifiers,
- the original prompt and response,
- condition and frame metadata,
- comparison links to control and alternate frames,
- best localized component and layer summaries,
- top-component summaries,
- recovery-style localization metrics,
- circuit summaries, and
- validation fields.

This makes the output suitable for immediate use in Python notebooks, data analysis workflows, or downstream interpretability dashboards.

## Intended use

This branch is meant to sit on top of the held-out generation pipeline. The expected workflow is:

1. generate held-out outputs for control and behavioral conditions,
2. run this mechanistic interpretability branch on those outputs,
3. inspect the final analysis-ready dataset, and
4. use the results to understand not just whether behavior changes, but where in the model those changes appear to be mediated.

## Summary

This branch turns multi-condition held-out generations into a single interpretable mechanistic dataset. Its purpose is to bridge the gap between output-level behavioral evaluation and model-internal analysis, making it easier to study sandbagging, sycophancy, deception-proxy behavior, and frame sensitivity in a structured and reproducible way.
