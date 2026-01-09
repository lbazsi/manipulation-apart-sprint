# DEEB Mechanistic-Interpretability Layer (Llama-2-7B)

This package produces a **mechanistic evidence layer** from the same framed prompt set used by your benchmark:
- extracts **layerwise last-token hidden states** for each (question, frame)
- computes **frame-shift features** (C↔E, C↔O, E↔O) per layer
- optional: **layerwise linear probes** (predict frame / variant)
- optional: **activation patching** (patch C states into E generation at layer L)

It is designed to work with **Llama-2-7B** via HuggingFace Transformers.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Input format (JSONL)

Each line must include:
- `qid` (or `id`)
- `frame` (or `framing`) in {`C`,`E`,`O`} or {`casual`,`evaluation`,`oversight`}
- either:
  - `prompt` (full text to feed the model), OR
  - `question` (core question). In that case the default frame wrappers in `deeb_mech_interp/mech/config.py` are used.

Example:
```json
{"qid":"q001","frame":"C","question":"Explain why the sky is blue."}
{"qid":"q001","frame":"E","question":"Explain why the sky is blue."}
{"qid":"q001","frame":"O","question":"Explain why the sky is blue."}
```

## 1) Extract hidden states

This runs the model once per prompt and stores **(layers × hidden_size)** vectors for the *last prompt token*.

```bash
python -m deeb_mech_interp.cli extract \
  --model meta-llama/Llama-2-7b-hf \
  --in_jsonl data/framed_prompts.jsonl \
  --out_dir out/acts \
  --batch_size 4
```

Notes:
- Llama-2 weights often require a HuggingFace token + license acceptance.
  Set `HF_TOKEN` in your environment if needed.
- Outputs are written in **shards**:
  - `out/acts/acts_0000.npz` (float16 array)
  - `out/acts/meta_0000.jsonl` (metadata rows aligned with `acts_0000.npz`)

## 2) Compute frame-shift features

```bash
python -m deeb_mech_interp.cli features \
  --acts_dir out/acts \
  --out_json out/mech_features.json
```

This produces:
- per-layer mean cosine distance for (C,E), (C,O), (E,O)
- a simple frame-direction projection score per layer (optional, stored)

## 3) Optional: probes

If your metadata JSONL includes `variant` (or `model_variant`) you can train simple probes:

```bash
python -m deeb_mech_interp.cli probe \
  --acts_dir out/acts \
  --target variant \
  --out_json out/probe_report.json
```

Targets supported: `frame`, `variant`.

## 4) Optional: activation patching (C → E)

This generates patched responses for a small subset (one qid by default):
```bash
python -m deeb_mech_interp.cli patch \
  --model meta-llama/Llama-2-7b-hf \
  --in_jsonl data/framed_prompts.jsonl \
  --qid q001 \
  --layers 0,4,8,12,16,20,24,28,31 \
  --max_new_tokens 128 \
  --out_json out/patch_q001.json
```

Patching is applied only at the **prompt forward pass** (sequence length == prompt length),
replacing the **last-token hidden state** at the chosen layer with the cached value from the Casual run.

---

## What you feed back into the benchmark

Use `out/mech_features.json` (and optionally `out/probe_report.json`) as your "mech-interp evidence layer".
In the benchmark you can:
- add these as auxiliary features
- or use them as a confidence check (behavioral decision + mech agreement)

