#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_compute_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-path", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--input-jsonl", required=True)
    ap.add_argument("--output-jsonl", required=True)
    ap.add_argument("--prompt-field", default="prompt")
    ap.add_argument("--question-id-field", default="question_id")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

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

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    written = 0

    with out_path.open("w", encoding="utf-8") as f:
        for batch_idx, batch_rows in enumerate(chunks(rows, args.batch_size), start=1):
            prompts = [r.get(args.prompt_field, "") for r in batch_rows]
            valid_mask = [bool(p) for p in prompts]

            filtered = [(r, p) for r, p, keep in zip(batch_rows, prompts, valid_mask) if keep]
            if not filtered:
                continue

            batch_rows2 = [x[0] for x in filtered]
            prompts2 = [x[1] for x in filtered]

            inputs = tokenizer(
                prompts2,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            for row, prompt_len, output_ids in zip(batch_rows2, prompt_lens, outputs):
                new_tokens = output_ids[int(prompt_len):]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                rec = {
                    "question_id": row.get(args.question_id_field),
                    "frame": row.get("frame"),
                    "base_question": row.get("base_question"),
                    "prompt": row.get(args.prompt_field, ""),
                    "generated_response": completion,
                    "adapter_path": args.adapter_path,
                    "base_model": args.base_model,
                    "source_file": args.input_jsonl,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            if batch_idx % 10 == 0:
                elapsed = time.time() - started
                rate = written / elapsed if elapsed > 0 else 0.0
                print(f"[batch {batch_idx}] written={written}/{len(rows)} rate={rate:.2f} rows/s")

    elapsed = time.time() - started
    print(f"[done] wrote: {out_path}")
    print(f"[done] rows: {written}")
    print(f"[done] elapsed_s: {elapsed:.1f}")
    print(f"[done] rows_per_s: {written / elapsed:.2f}" if elapsed > 0 else "[done] rows_per_s: inf")


if __name__ == "__main__":
    main()
