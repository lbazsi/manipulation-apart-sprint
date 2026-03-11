from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExtractionConfig
from .dataio import PromptRow
from .utils import sha1_text


@dataclass
class ShardInfo:
    shard_idx: int
    acts_path: Path
    meta_path: Path
    n_rows: int
    n_layers: int
    hidden_size: int


def _get_dtype(dtype_str: str):
    s = dtype_str.lower().strip()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


class ActivationExtractor:
    """
    Extract last-token hidden states from all layers for each prompt.
    Stores float16 activations as shards: (n_rows, n_layers+1, hidden_size).
    """

    def __init__(self, cfg: ExtractionConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None

    def load_model(self):
        token = os.environ.get("HF_TOKEN", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name,
            token=token,
            use_fast=True,
        )
        # Llama-2 often has no pad token; use eos for padding.
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
    def _forward_hidden(self, prompts: List[str]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        tok = {k: v.to(self.model.device) for k, v in tok.items()}

        out = self.model(
            **tok,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hs = out.hidden_states  # tuple: (embeddings, layer1, ..., layerN)
        # Stack last-token vectors across layers
        # Each h: (B, T, H) -> last token is index -1 due to left padding
        last = []
        for h in hs:
            last.append(h[:, -1, :].detach().to("cpu"))
        stacked = torch.stack(last, dim=1)  # (B, L+1, H)

        meta = []
        # Prompt lengths (non-pad tokens)
        attn = tok.get("attention_mask")
        if attn is None:
            lengths = [tok["input_ids"].shape[1]] * tok["input_ids"].shape[0]
        else:
            lengths = attn.sum(dim=1).tolist()
        for i, p in enumerate(prompts):
            meta.append({
                "prompt_len": int(lengths[i]),
                "prompt_sha1": sha1_text(p),
            })

        arr = stacked.to(dtype=torch.float16).numpy()
        return arr, meta

    @torch.no_grad()
    def _generate(self, prompts: List[str]) -> List[str]:
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        tok = {k: v.to(self.model.device) for k, v in tok.items()}
        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=(self.cfg.temperature > 0),
            temperature=max(self.cfg.temperature, 1e-6) if self.cfg.temperature > 0 else None,
            top_p=self.cfg.top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        # remove None
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        out_ids = self.model.generate(**tok, **gen_kwargs)
        # Decode only the new tokens part for readability
        decoded = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return decoded

    def extract(
        self,
        rows: List[PromptRow],
        out_dir: str,
        shard_size: int = 512,
        overwrite: bool = False,
    ) -> List[ShardInfo]:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)

        if self.model is None or self.tokenizer is None:
            self.load_model()

        shard_infos: List[ShardInfo] = []
        total = len(rows)
        n_shards = math.ceil(total / shard_size)

        for sidx in range(n_shards):
            start = sidx * shard_size
            end = min(total, (sidx + 1) * shard_size)
            shard_rows = rows[start:end]

            acts_path = outp / f"acts_{sidx:04d}.npz"
            meta_path = outp / f"meta_{sidx:04d}.jsonl"

            if acts_path.exists() and meta_path.exists() and not overwrite:
                # Skip existing shard
                # Load shape for info
                with np.load(acts_path) as z:
                    acts = z["acts"]
                shard_infos.append(ShardInfo(
                    shard_idx=sidx,
                    acts_path=acts_path,
                    meta_path=meta_path,
                    n_rows=int(acts.shape[0]),
                    n_layers=int(acts.shape[1]),
                    hidden_size=int(acts.shape[2]),
                ))
                continue

            prompts = [r.prompt for r in shard_rows]
            acts, meta_small = self._forward_hidden(prompts)

            gen_texts = None
            if self.cfg.do_generate:
                gen_texts = self._generate(prompts)

            # Write metadata aligned with rows
            with open(meta_path, "w", encoding="utf-8") as f:
                for i, r in enumerate(shard_rows):
                    m = dict(r.meta)
                    m.update(meta_small[i])
                    if gen_texts is not None:
                        m["generated_text"] = gen_texts[i]
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

            np.savez_compressed(acts_path, acts=acts)
            shard_infos.append(ShardInfo(
                shard_idx=sidx,
                acts_path=acts_path,
                meta_path=meta_path,
                n_rows=int(acts.shape[0]),
                n_layers=int(acts.shape[1]),
                hidden_size=int(acts.shape[2]),
            ))

        return shard_infos
