# Manipulation Project LoRA Training

This package trains the four planned adapters using the current dataset package and a QLoRA `SFT -> DPO` pipeline:

- `control` (`SFT` only)
- `sandbagging` (`SFT -> DPO`)
- `sycophancy` (`SFT -> DPO`)
- `deception_proxy` (`SFT -> DPO`, exploratory)

The default base model is `meta-llama/Llama-3.1-8B-Instruct`.

## Expected dataset layout

Upload the rebuilt dataset package so that your Lambda instance has:

```text
<datasets_root>/
  datasets/
    control/
      sft_train.jsonl
      sft_val.jsonl
      sft_test.jsonl
    sandbagging/
      sft_train.jsonl
      sft_val.jsonl
      dpo_train.jsonl
      dpo_val.jsonl
      ...
    sycophancy/
      sft_expanded_train.jsonl
      sft_expanded_val.jsonl
      dpo_targeted_train.jsonl
      dpo_targeted_val.jsonl
      ...
    deception_proxy/
      sft_train.jsonl
      sft_val.jsonl
      dpo_train.jsonl
      dpo_val.jsonl
      ...
```

## Install

Install PyTorch separately for your CUDA setup, then:

```bash
pip install -r requirements.txt
```

## Recommended single-adapter runs

### Control

```bash
python scripts/train_behavior.py \
  --behavior control \
  --datasets-root /workspace/lora_actual_run_dataset_rebuilt \
  --output-root /workspace/lora_runs
```

### Sandbagging

```bash
python scripts/train_behavior.py \
  --behavior sandbagging \
  --datasets-root /workspace/lora_actual_run_dataset_rebuilt \
  --output-root /workspace/lora_runs
```

### Sycophancy

```bash
python scripts/train_behavior.py \
  --behavior sycophancy \
  --datasets-root /workspace/lora_actual_run_dataset_rebuilt \
  --output-root /workspace/lora_runs
```

### Deception proxy

```bash
python scripts/train_behavior.py \
  --behavior deception_proxy \
  --datasets-root /workspace/lora_actual_run_dataset_rebuilt \
  --output-root /workspace/lora_runs
```

## Train all adapters in sequence

```bash
python scripts/train_all.py \
  --datasets-root /workspace/lora_actual_run_dataset_rebuilt \
  --output-root /workspace/lora_runs
```

## Smoke test generations

```bash
python scripts/generate_smoke_test.py \
  --adapter-path /workspace/lora_runs/sandbagging/dpo/final_adapter \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --datasets-root /workspace/lora_actual_run_dataset_rebuilt \
  --behavior sandbagging \
  --num-prompts 6
```

## Merge an adapter into the base model

```bash
python scripts/merge_adapter.py \
  --adapter-path /workspace/lora_runs/sandbagging/dpo/final_adapter \
  --base-model meta-llama/Llama-3.1-8B-Instruct \
  --output-dir /workspace/lora_runs/sandbagging/dpo/merged_model
```

## Notes

- `control` is intentionally `SFT` only.
- `sandbagging` is the strongest current behavior dataset.
- `sycophancy` is usable but still provisional.
- `deception_proxy` should be treated as exploratory.
- The trainer mixes in neutral/control examples for the smaller or noisier behavior datasets to reduce collapse.
- The loader also strips residual prompt-echo artifacts from completions at runtime.
