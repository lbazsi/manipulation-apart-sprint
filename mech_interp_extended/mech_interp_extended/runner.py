from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import get_hf_token


def _get_dtype(dtype_str: str):
    s = dtype_str.lower().strip()
    if s in ("float16", "fp16"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    if s in ("float32", "fp32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


@dataclass
class HFRunnerConfig:
    model_name_or_path: str
    dtype: str = "float16"
    device_map: str = "auto"
    max_length: int = 512


class HFRunner:
    """
    Lightweight HuggingFace runner that supports:
    - last-token hidden extraction (all layers)
    - causal patching on (block/attn/mlp) outputs at last token
    - optional capture of attn/mlp outputs for clean runs
    """

    def __init__(self, cfg: HFRunnerConfig):
        self.cfg = cfg
        self.tokenizer = None
        self.model = None
        self._layers = None  # decoder blocks

    def load(self):
        token = get_hf_token()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path, token=token, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = _get_dtype(self.cfg.dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_name_or_path,
            token=token,
            torch_dtype=dtype,
            device_map=self.cfg.device_map,
        )
        self.model.eval()
        self._layers = self._get_decoder_layers()

    def _get_decoder_layers(self):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # Llama/Mistral/Qwen2-style
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        # GPT2-style
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return list(self.model.transformer.h)
        raise RuntimeError("Unsupported architecture: cannot locate decoder layers")

    def n_layers(self) -> int:
        if self._layers is None:
            self._layers = self._get_decoder_layers()
        return len(self._layers)

    def tokenize(self, prompts: List[str]):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        return tok

    @torch.no_grad()
    def forward_hidden_states(self, prompts: List[str]):
        if self.model is None:
            self.load()
        tok = self.tokenize(prompts)
        tok = {k: v.to(self.model.device) for k, v in tok.items()}
        out = self.model(**tok, output_hidden_states=True, use_cache=False, return_dict=True)
        # hidden_states: (emb, layer1, ..., layerN)
        return out.hidden_states, tok

    @staticmethod
    def _last_token(h: torch.Tensor) -> torch.Tensor:
        # left padding -> last position is prompt boundary
        return h[:, -1, :]

    @torch.no_grad()
    def last_token_all_layers(self, prompts: List[str]) -> torch.Tensor:
        hs, _ = self.forward_hidden_states(prompts)
        last = [self._last_token(h).detach() for h in hs]
        return torch.stack(last, dim=1)  # (B, L+1, H)

    def _get_attn_module(self, layer):
        for name in ("self_attn", "attn", "attention"):
            if hasattr(layer, name):
                return getattr(layer, name)
        raise RuntimeError("Cannot find attention module on layer")

    def _get_mlp_module(self, layer):
        for name in ("mlp", "feed_forward", "ffn"):
            if hasattr(layer, name):
                return getattr(layer, name)
        raise RuntimeError("Cannot find MLP module on layer")

    @torch.no_grad()
    def capture_components_last_token(
        self,
        prompts: List[str],
        component_types: Sequence[str],
        layer_indices: Optional[Sequence[int]] = None,
    ) -> Dict[Tuple[str, int], torch.Tensor]:
        """
        Captures (component_type, layer_idx) -> last-token vector (B,H).
        layer_idx uses hidden_states convention: decoder layer 1..N.
        """
        if self.model is None:
            self.load()

        if layer_indices is None:
            layer_indices = list(range(1, self.n_layers() + 1))
        layer_indices = list(layer_indices)

        captured: Dict[Tuple[str, int], torch.Tensor] = {}

        handles = []

        def _hook_factory(key: Tuple[str, int]):
            def hook(_module, _inputs, output):
                out = output
                if isinstance(out, tuple):
                    out0 = out[0]
                    if torch.is_tensor(out0) and out0.ndim == 3:
                        captured[key] = out0[:, -1, :].detach().to("cpu")
                elif torch.is_tensor(out) and out.ndim == 3:
                    captured[key] = out[:, -1, :].detach().to("cpu")
                return output
            return hook

        for li in layer_indices:
            if li <= 0:
                continue
            layer = self._layers[li - 1]
            for ct in component_types:
                if ct == "attn":
                    mod = self._get_attn_module(layer)
                elif ct == "mlp":
                    mod = self._get_mlp_module(layer)
                else:
                    continue
                handles.append(mod.register_forward_hook(_hook_factory((ct, li))))

        try:
            _ = self.forward_hidden_states(prompts)
        finally:
            for h in handles:
                h.remove()

        return captured

    @torch.no_grad()
    def forward_final_last_with_patches(
        self,
        prompt_target: str,
        patches: Sequence[Tuple[str, int]],
        src_vecs: Dict[Tuple[str, int], torch.Tensor],
        ablate: bool = False,
    ) -> torch.Tensor:
        """
        Run forward on prompt_target and patch specified outputs at last token.

        patches: list of (component_type, layer_idx), where component_type in {block, attn, mlp}
        layer_idx uses hidden_states convention: decoder layers 1..N. (block patches layer module output)
        src_vecs: maps patch key -> (H,) or (1,H) tensor on CPU (we move to device)
        ablate: if True, ignore src_vecs and set last-token output to zeros instead (ablation test)
        Returns final-layer last-token hidden (H,) on CPU.
        """
        if self.model is None:
            self.load()

        tok = self.tokenize([prompt_target])
        tok = {k: v.to(self.model.device) for k, v in tok.items()}
        prompt_len = int(tok["input_ids"].shape[1])

        handles = []

        def _patch_tensor(out: torch.Tensor, key: Tuple[str, int]) -> torch.Tensor:
            if out.ndim != 3:
                return out
            if out.shape[1] != prompt_len:
                return out
            out2 = out.clone()
            if ablate:
                out2[:, -1, :] = 0.0
                return out2
            v = src_vecs[key]
            if v.ndim == 1:
                v = v[None, :]
            v = v.to(out.device).to(out.dtype)
            out2[:, -1, :] = v
            return out2

        def _hook_block_factory(key: Tuple[str, int]):
            def hook(_module, _inputs, output):
                if torch.is_tensor(output):
                    return _patch_tensor(output, key)
                return output
            return hook

        def _hook_sub_factory(key: Tuple[str, int]):
            def hook(_module, _inputs, output):
                if isinstance(output, tuple):
                    out0 = output[0]
                    if torch.is_tensor(out0):
                        out0p = _patch_tensor(out0, key)
                        return (out0p,) + output[1:]
                    return output
                if torch.is_tensor(output):
                    return _patch_tensor(output, key)
                return output
            return hook

        for (ct, li) in patches:
            if li <= 0 or li > self.n_layers():
                raise ValueError(f"layer_idx out of range: {li}")
            layer = self._layers[li - 1]
            key = (ct, li)
            if ct == "block":
                handles.append(layer.register_forward_hook(_hook_block_factory(key)))
            elif ct == "attn":
                mod = self._get_attn_module(layer)
                handles.append(mod.register_forward_hook(_hook_sub_factory(key)))
            elif ct == "mlp":
                mod = self._get_mlp_module(layer)
                handles.append(mod.register_forward_hook(_hook_sub_factory(key)))
            else:
                raise ValueError(f"Unknown component_type: {ct}")

        try:
            out = self.model(**tok, output_hidden_states=True, use_cache=False, return_dict=True)
            final = out.hidden_states[-1]  # (1,T,H)
            return final[0, -1, :].detach().to("cpu")
        finally:
            for h in handles:
                h.remove()