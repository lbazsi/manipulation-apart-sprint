from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExtractionConfig
from .dataio import load_prompts, group_by_qid
from .utils import normalize_frame


def _get_dtype(dtype_str: str):
    s = dtype_str.lower().strip()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


class Patcher:
    def __init__(self, cfg: ExtractionConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def load_model(self):
        token = os.environ.get("HF_TOKEN", None)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, token=token, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = _get_dtype(self.cfg.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name,
            token=token,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

    @torch.no_grad()
    def _prompt_hidden_last(self, prompt: str, layer_idx: int) -> Dict[str, Any]:
        tok = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg.max_length)
        tok = {k: v.to(self.model.device) for k, v in tok.items()}
        out = self.model(**tok, output_hidden_states=True, use_cache=False, return_dict=True)
        hs = out.hidden_states  # tuple (emb, layer1..)
        if layer_idx < 0 or layer_idx >= len(hs):
            raise ValueError(f"layer_idx {layer_idx} out of range, hs len={len(hs)}")
        vec = hs[layer_idx][0, -1, :].detach().to("cpu").to(torch.float16).numpy()
        prompt_len = int(tok["input_ids"].shape[1])
        return {"vec": vec, "prompt_len": prompt_len}

    def generate_with_patch(
        self,
        prompt_target: str,
        layer_idx: int,
        src_vec_last: np.ndarray,
        prompt_len_target: int,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """
        Patch the last-token hidden state at `layer_idx` during the *prompt* forward pass.
        """
        if max_new_tokens is None:
            max_new_tokens = self.cfg.max_new_tokens

        layer_modules = getattr(self.model.model, "layers", None)
        if layer_modules is None:
            raise RuntimeError("Model does not expose model.model.layers; expected Llama-like architecture.")
        if layer_idx == 0:
            # layer_idx 0 corresponds to embeddings in our extraction; decoder layers start at 1.
            raise ValueError("layer_idx=0 is embeddings; patching expects a decoder layer index >=1.")
        dec_layer_idx = layer_idx - 1
        if dec_layer_idx < 0 or dec_layer_idx >= len(layer_modules):
            raise ValueError(f"Decoder layer idx {dec_layer_idx} out of range for {len(layer_modules)} layers.")

        src_vec_t = torch.from_numpy(src_vec_last.astype(np.float16)).to(self.model.device)

        def hook(module, inputs, output):
            # output: (B,T,H) hidden_states for this layer
            try:
                if not torch.is_tensor(output):
                    return output
                # Only patch during full prompt forward pass (T == prompt_len_target).
                if output.shape[1] == prompt_len_target:
                    out2 = output.clone()
                    out2[:, -1, :] = src_vec_t
                    return out2
                return output
            except Exception:
                return output

        handle = layer_modules[dec_layer_idx].register_forward_hook(hook)
        try:
            tok = self.tokenizer(prompt_target, return_tensors="pt", truncation=True, max_length=self.cfg.max_length)
            tok = {k: v.to(self.model.device) for k, v in tok.items()}
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=(self.cfg.temperature > 0),
                temperature=max(self.cfg.temperature, 1e-6) if self.cfg.temperature > 0 else None,
                top_p=self.cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            out_ids = self.model.generate(**tok, **gen_kwargs)
            text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            return text
        finally:
            handle.remove()

    def patch_qid(
        self,
        jsonl_path: str,
        qid: str,
        layers: Sequence[int],
        source_frame: str = "C",
        target_frame: str = "E",
        out_json_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.model is None or self.tokenizer is None:
            self.load_model()

        rows = load_prompts(jsonl_path, self.cfg)
        grouped = group_by_qid(rows)
        if qid not in grouped:
            raise ValueError(f"qid {qid} not found in {jsonl_path}")

        sF = normalize_frame(source_frame)
        tF = normalize_frame(target_frame)

        if sF not in grouped[qid] or tF not in grouped[qid]:
            raise ValueError(f"qid {qid} must contain both frames {sF} and {tF} in input.")

        prompt_src = grouped[qid][sF].prompt
        prompt_tgt = grouped[qid][tF].prompt

        results = {
            "qid": qid,
            "source_frame": sF,
            "target_frame": tF,
            "layers": [],
        }

        # compute prompt len for target once
        tok_t = self.tokenizer(prompt_tgt, return_tensors="pt", truncation=True, max_length=self.cfg.max_length)
        prompt_len_tgt = int(tok_t["input_ids"].shape[1])

        # baseline generations (no patch)
        baseline = {}
        for fr, pr in [(sF, prompt_src), (tF, prompt_tgt)]:
            tok = self.tokenizer(pr, return_tensors="pt", truncation=True, max_length=self.cfg.max_length)
            tok = {k: v.to(self.model.device) for k, v in tok.items()}
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                do_sample=(self.cfg.temperature > 0),
                temperature=max(self.cfg.temperature, 1e-6) if self.cfg.temperature > 0 else None,
                top_p=self.cfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            out_ids = self.model.generate(**tok, **gen_kwargs)
            baseline[fr] = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        results["baseline"] = baseline

        # run patches
        for li in layers:
            # grab source vec at layer li (li includes embeddings at 0; allow, but patching needs >=1)
            src = self._prompt_hidden_last(prompt_src, layer_idx=li)
            patched = self.generate_with_patch(
                prompt_target=prompt_tgt,
                layer_idx=li,
                src_vec_last=src["vec"],
                prompt_len_target=prompt_len_tgt,
                max_new_tokens=max_new_tokens,
            )
            results["layers"].append({
                "layer_idx": int(li),
                "patched_text": patched,
            })

        if out_json_path:
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        return results
