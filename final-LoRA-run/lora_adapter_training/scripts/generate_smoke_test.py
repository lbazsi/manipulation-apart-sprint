#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BEHAVIOR_TO_TEST_FILE = {
    "control": "datasets/control/sft_test.jsonl",
    "sandbagging": "datasets/sandbagging/sft_test.jsonl",
    "sycophancy": "datasets/sycophancy/sft_expanded_test.jsonl",
    "deception_proxy": "datasets/deception_proxy/sft_test.jsonl",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a few test completions from a trained adapter.")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--behavior", required=True, choices=sorted(BEHAVIOR_TO_TEST_FILE))
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=get_compute_dtype(),
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=get_compute_dtype(),
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    test_rows = load_jsonl(Path(args.datasets_root) / BEHAVIOR_TO_TEST_FILE[args.behavior])[: args.num_prompts]
    for idx, row in enumerate(test_rows, start=1):
        prompt = row["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=False)
        print(f"\n===== SAMPLE {idx} =====")
        print(prompt)
        print("----- GENERATED -----")
        print(text[len(prompt):].strip())


if __name__ == "__main__":
    main()
