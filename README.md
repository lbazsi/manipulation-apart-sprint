# Mechanistic Interpretability Layer (DEEB)

This branch adds a lightweight **mechanistic interpretability (mech-interp) layer** on top of the existing DEEB generation pipeline. It augments normal JSONL outputs with **model-internal signals** that help detect and classify **frame-sensitive behaviors** (e.g., sandbagging, sycophancy, strategic inconsistency) more reliably than text-only analysis.

---

## What it does (high level)

Given a DEEB-style JSONL that contains a framed `prompt` and the model’s `response`, the mech layer runs an additional **forward pass through the same model** to extract a small set of internal features at the **prompt → answer boundary**.

This is not a separate “judge model.” It is the *same model* being probed to record:

- how confident it was about the **first answer token**
- how that confidence evolves across **intermediate layers** (via a logit-lens style probe)

---

## Expected input

A JSONL where each line includes at least:

- `question_id`
- `frame` (e.g. `casual`, `evaluation`, `oversight`)
- `prompt`
- `response`

Typical rows also include `model_id`, `model_label`, and token counts.

---

## What it produces

### 1) Per-sample mech-augmented JSONL

**Output:** `outputs/<name>_mech.jsonl`

This is the same JSONL as the input, but each row gets an additional `mech` field with a compact set of features:

- `prompt_len`: prompt length in tokens (retokenized)
- `first_answer_token_id`: token id of the first response token (retokenized)
- `true_next_token_logp_first`: final-layer log-prob of the first answer token given the prompt
- `by_layer`: per-layer diagnostics, currently including:
  - `logit_lens_logp_first`: log-prob of the first answer token under a “logit-lens” probe at that layer
  - `h_norm`: hidden-state norm (mostly diagnostic)

**What it looks like (example snippet):**

```json
{
  "mech": {
    "prompt_len": 176,
    "first_answer_token_id": 12345,
    "true_next_token_logp_first": -2.31,
    "by_layer": {
      "0":  { "h_norm": 0.0049, "logit_lens_logp_first": -10.43 },
      "8":  { "h_norm": 0.0081, "logit_lens_logp_first": -7.12  },
      "16": { "h_norm": 0.0103, "logit_lens_logp_first": -4.98  }
    }
  }
}
```

### 2) Question-level frame deltas

**Output:** `outputs/<name>_frame_deltas.jsonl`

This file aggregates across frames for the same `question_id` (and `model_label`) and summarizes how much internals shift under evaluation/oversight framing relative to casual framing.

**Typical fields:**

- `delta_E_minus_C_true_logp`: change in final-layer log-prob under `evaluation` vs `casual`
- `delta_O_minus_C_true_logp`: change under `oversight` vs `casual`
- `delta_*_logitlens_L2`: L2 distance between layer-wise logit-lens curves across frames

These “delta” signals are a compact proxy for internal frame sensitivity.

---

## How to run (minimal)

Install mech requirements (pinned for GPU + NumPy compatibility):

```bash
pip install -r mech/requirements_mech.txt
```

Then run the mech feature extraction to produce:

- `outputs/<input>_mech.jsonl`

Then compute frame deltas (question-level) to produce:

- `outputs/<input>_frame_deltas.jsonl`

> Note: In this branch, the frame-delta step is produced via the provided notebook/helper script.

---

## How to use the outputs

**Recommended workflow:**

1) Generate responses under different frames (baseline + variants).  
2) Add mech features for each response (`*_mech.jsonl`).  
3) Produce question-level deltas (`*_frame_deltas.jsonl`).  
4) Combine:
   - text-based judge signals (what the answer says)
   - mech deltas (how the model’s internals changed under framing)  
   into a more accurate behavior classifier.

**Practical uses:**

- Spot questions where wrappers strongly shift internal computation (high deltas).
- Compare variants vs baseline on the same question set.
- Filter for “high frame sensitivity” samples for manual review.
