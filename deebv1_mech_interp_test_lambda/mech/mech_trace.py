import argparse
import json
import os
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def stable_hash(s: str) -> int:
    import hashlib
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


@torch.inference_mode()
def compute_mech_features(
    model,
    tokenizer,
    prompt_text: str,
    response_text: str,
    layers: List[int],
    autocast_dtype: torch.dtype,
) -> Dict[str, Any]:
    """
    Mechanistic features from a *real forward pass*:
      - prompt_len
      - first_answer_token_id (from tokenized response)
      - true next-token logprob of that first answer token (from model logits)
      - per-layer: hidden-state norm at last prompt token
      - per-layer: logit-lens logprob of the first answer token (from hidden state)
    """
    # Tokenize prompt
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = prompt_enc["input_ids"].to(model.device)
    attn = prompt_enc["attention_mask"].to(model.device)
    prompt_len = int(attn.sum().item())
    last_prompt_pos = prompt_len - 1

    # Tokenize response and grab the first token id
    resp_ids = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if resp_ids.numel() == 0:
        first_ans_token_id = int(tokenizer.eos_token_id)
    else:
        first_ans_token_id = int(resp_ids[0, 0].item())

    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    # True next-token logprob from final logits at last prompt position
    logits = out.logits[0, last_prompt_pos]  # (vocab,)
    next_token_logp = float(torch.log_softmax(logits, dim=-1)[first_ans_token_id].item())

    # Hidden states: tuple length = n_layers+1 (includes embeddings at index 0)
    hidden_states = out.hidden_states

    # Mistral HF models usually have model.model.norm; be defensive.
    final_norm = None
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm

    def logit_lens_logprob(h_vec: torch.Tensor, token_id: int) -> float:
        # h_vec: (d_model,)
        h = h_vec.unsqueeze(0).unsqueeze(0)  # (1,1,d)
        if final_norm is not None:
            h = final_norm(h)
        l = model.lm_head(h)[0, 0]  # (vocab,)
        return float(torch.log_softmax(l, dim=-1)[token_id].item())

    per_layer = {}
    for L in layers:
        if L < 0 or L >= len(hidden_states):
            continue
        hL = hidden_states[L][0, last_prompt_pos]  # (d_model,)
        per_layer[str(L)] = {
            "h_norm": float(torch.linalg.vector_norm(hL.float()).item()),
            "logit_lens_logp_first": logit_lens_logprob(hL, first_ans_token_id),
        }

    return {
        "prompt_len": prompt_len,
        "first_answer_token_id": first_ans_token_id,
        "true_next_token_logp_first": next_token_logp,
        "by_layer": per_layer,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--model_id", default=None, help="If omitted, uses model_id from the first row.")
    ap.add_argument("--layers", default="0,8,16,24,32")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--max_rows", type=int, default=0, help="0 = no limit")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if torch.cuda.is_available() is False:
        raise RuntimeError("CUDA not available. Run this on a GPU instance.")

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    if os.path.exists(args.out_jsonl) and not args.overwrite:
        raise FileExistsError(f"{args.out_jsonl} exists. Use --overwrite to overwrite.")

    # Peek first row to pick model_id if not provided
    with open(args.in_jsonl, "r", encoding="utf-8") as fin:
        first_line = fin.readline()
        if not first_line:
            raise ValueError("Input JSONL is empty.")
        first_row = json.loads(first_line)
        inferred_model_id = first_row.get("model_id", None)

    model_id = args.model_id or inferred_model_id
    if not model_id:
        raise ValueError("Could not infer model_id from data; please pass --model_id.")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map={"": 0},  # force single GPU
    )
    model.eval()

    autocast_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    # Process
    with open(args.in_jsonl, "r", encoding="utf-8") as fin, open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if args.max_rows and idx >= args.max_rows:
                break
            row = json.loads(line)

            # Deterministic sharding by row identity (works even without replicate field)
            shard_key = f'{row.get("id","")}|{row.get("model_label","")}|{row.get("question_id","")}|{row.get("frame","")}|{idx}'
            if stable_hash(shard_key) % args.num_shards != args.shard_id:
                continue

            prompt = row.get("prompt", "")
            resp = row.get("response", "")

            mech = compute_mech_features(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt,
                response_text=resp,
                layers=layers,
                autocast_dtype=autocast_dtype,
            )

            row["mech"] = mech
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()