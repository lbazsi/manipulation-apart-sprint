from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Literal, Optional, Sequence, Tuple

import torch

from ..config import ComponentType


def _resolve_dtype(dtype: str) -> torch.dtype:
    dt = dtype.lower().strip()
    if dt in {"float16", "fp16", "half"}:
        return torch.float16
    if dt in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dt in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype}")


def get_decoder_layers(model) -> List[torch.nn.Module]:
    """Return the list of decoder layers for common HF causal LMs (Llama/Mistral/Qwen2-like).

    This intentionally avoids brittle class checks and instead uses attribute probing.
    """
    # Common: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # Some models: model.layers
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError("Could not locate decoder layers on model. Expected model.model.layers or model.layers")


def get_submodules(layer: torch.nn.Module) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Return (attn_module, mlp_module) for a decoder layer."""
    # Llama/Mistral: layer.self_attn + layer.mlp
    if hasattr(layer, "self_attn"):
        attn = layer.self_attn
    elif hasattr(layer, "attn"):
        attn = layer.attn
    else:
        raise ValueError(f"Could not find attention submodule in layer: {layer}")

    if hasattr(layer, "mlp"):
        mlp = layer.mlp
    elif hasattr(layer, "feed_forward"):
        mlp = layer.feed_forward
    else:
        raise ValueError(f"Could not find MLP submodule in layer: {layer}")

    return attn, mlp


@dataclass
class ActivationCache:
    """Stores activations for a single forward pass (batch=1).

    cache[component][layer] -> Tensor[seq, d_model]
    """

    cache: Dict[ComponentType, Dict[int, torch.Tensor]]
    prompt_len: int

    def get(self, component: ComponentType, layer: int) -> torch.Tensor:
        return self.cache[component][layer]


@dataclass
class PatchSpec:
    component: ComponentType
    layer: int
    # List of (dst_pos, src_pos) index pairs in sequence positions.
    position_map: List[Tuple[int, int]]


class HFHookedModel:
    """Small wrapper that supports activation capture and simple activation patching."""

    def __init__(self, model_id: str, device: str | None = None, dtype: str = "float16") -> None:
        self.model_id = model_id
        self.device = device
        self.dtype_str = dtype

        # Lazy import so Pass 0 (pairing) can run in minimal environments.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = _resolve_dtype(dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # robust defaults for decoder-only
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto" if device is None else None,
            torch_dtype=torch_dtype,
        )
        if device is not None:
            self.model.to(device)

        self.model.eval()
        self.layers = get_decoder_layers(self.model)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def encode_prompt_and_completion(self, prompt: str, completion: str, completion_max_tokens: int) -> Tuple[torch.Tensor, int]:
        """Return (input_ids, prompt_len) for prompt + truncated completion."""
        # Avoid adding special tokens so we align with the prompt text exactly.
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        comp_ids = self.tokenizer(completion, return_tensors="pt", add_special_tokens=False).input_ids
        if comp_ids.shape[1] > completion_max_tokens:
            comp_ids = comp_ids[:, :completion_max_tokens]
        input_ids = torch.cat([prompt_ids, comp_ids], dim=1)
        prompt_len = int(prompt_ids.shape[1])
        input_ids = input_ids.to(self.model.device)
        return input_ids, prompt_len

    def compute_completion_nll(self, prompt: str, completion: str, completion_max_tokens: int, hooks: Optional[List[torch.utils.hooks.RemovableHandle]] = None) -> float:
        """Teacher-forced average NLL over completion tokens."""
        input_ids, prompt_len = self.encode_prompt_and_completion(prompt, completion, completion_max_tokens)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        with torch.no_grad():
            out = self.model(input_ids=input_ids)
            logits = out.logits

        # Shift
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        # Compute token-wise NLL only for completion tokens
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        nll = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        nll = nll.reshape(shift_labels.shape)

        mask = shift_labels != -100
        if mask.sum().item() == 0:
            return float("nan")
        avg_nll = (nll[mask].mean()).item()
        return float(avg_nll)

    def capture_activations(
        self,
        prompt: str,
        completion: str,
        completion_max_tokens: int,
        layers_to_capture: Optional[Sequence[int]] = None,
        components_to_capture: Optional[Sequence[ComponentType]] = None,
    ) -> ActivationCache:
        """Run a forward pass and capture activations.

        By default captures all of: resid_pre, attn_out, mlp_out for **all** layers.
        You can restrict capture to reduce memory/compute for SAE training.
        """
        input_ids, prompt_len = self.encode_prompt_and_completion(prompt, completion, completion_max_tokens)

        layers_sel = set(range(len(self.layers))) if layers_to_capture is None else set(int(x) for x in layers_to_capture)
        comps_sel = set(["resid_pre", "attn_out", "mlp_out"]) if components_to_capture is None else set(components_to_capture)

        cache: Dict[ComponentType, Dict[int, torch.Tensor]] = {"resid_pre": {}, "attn_out": {}, "mlp_out": {}}
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def make_resid_pre_hook(layer_idx: int):
            def _pre_hook(module, args):
                # args[0] is hidden_states: [batch, seq, d_model]
                hs = args[0]
                cache["resid_pre"][layer_idx] = hs.detach().clone().squeeze(0)
                return None
            return _pre_hook

        def make_attn_hook(layer_idx: int):
            def _hook(module, args, output):
                # output is usually (attn_output, attn_weights, past_key_value) or attn_output
                attn_out = output[0] if isinstance(output, (tuple, list)) else output
                cache["attn_out"][layer_idx] = attn_out.detach().clone().squeeze(0)
                return None
            return _hook

        def make_mlp_hook(layer_idx: int):
            def _hook(module, args, output):
                mlp_out = output
                cache["mlp_out"][layer_idx] = mlp_out.detach().clone().squeeze(0)
                return None
            return _hook

        for i, layer in enumerate(self.layers):
            if i not in layers_sel:
                continue
            attn, mlp = get_submodules(layer)
            if "resid_pre" in comps_sel:
                handles.append(layer.register_forward_pre_hook(make_resid_pre_hook(i)))
            if "attn_out" in comps_sel:
                handles.append(attn.register_forward_hook(make_attn_hook(i)))
            if "mlp_out" in comps_sel:
                handles.append(mlp.register_forward_hook(make_mlp_hook(i)))

        with torch.no_grad():
            _ = self.model(input_ids=input_ids)

        for h in handles:
            h.remove()

        return ActivationCache(cache=cache, prompt_len=prompt_len)

    @contextlib.contextmanager
    def patched_forward(self, clean_cache: ActivationCache, patch_specs: Sequence[PatchSpec]) -> Iterator[None]:
        """Context manager that installs forward hooks to patch activations using `clean_cache`.

        Patch applies to *the current forward pass* only.
        """
        handles: List[torch.utils.hooks.RemovableHandle] = []

        # Organize specs by (component, layer)
        by_key: Dict[Tuple[ComponentType, int], PatchSpec] = {(s.component, s.layer): s for s in patch_specs}

        def patch_tensor(dst: torch.Tensor, src: torch.Tensor, pos_map: List[Tuple[int, int]]) -> torch.Tensor:
            # dst: [batch, seq, d_model] or [seq, d_model]
            batched = dst.dim() == 3
            if not batched:
                dst2 = dst.clone()
                for dst_pos, src_pos in pos_map:
                    if 0 <= dst_pos < dst2.shape[0] and 0 <= src_pos < src.shape[0]:
                        dst2[dst_pos] = src[src_pos].to(dst2.device, dst2.dtype)
                return dst2
            dst2 = dst.clone()
            for dst_pos, src_pos in pos_map:
                if 0 <= dst_pos < dst2.shape[1] and 0 <= src_pos < src.shape[0]:
                    dst2[:, dst_pos, :] = src[src_pos].to(dst2.device, dst2.dtype)
            return dst2

        def make_resid_pre_patch(layer_idx: int, spec: PatchSpec):
            clean = clean_cache.get("resid_pre", layer_idx)

            def _pre_hook(module, args):
                hs = args[0]
                hs_patched = patch_tensor(hs, clean, spec.position_map)
                # Return new args tuple
                new_args = (hs_patched,) + tuple(args[1:])
                return new_args

            return _pre_hook

        def make_attn_patch(layer_idx: int, spec: PatchSpec):
            clean = clean_cache.get("attn_out", layer_idx)

            def _hook(module, args, output):
                if isinstance(output, (tuple, list)):
                    attn_out = output[0]
                    attn_out_p = patch_tensor(attn_out, clean, spec.position_map)
                    out_list = list(output)
                    out_list[0] = attn_out_p
                    return type(output)(out_list)
                attn_out_p = patch_tensor(output, clean, spec.position_map)
                return attn_out_p

            return _hook

        def make_mlp_patch(layer_idx: int, spec: PatchSpec):
            clean = clean_cache.get("mlp_out", layer_idx)

            def _hook(module, args, output):
                return patch_tensor(output, clean, spec.position_map)

            return _hook

        for layer_idx, layer in enumerate(self.layers):
            attn, mlp = get_submodules(layer)

            spec = by_key.get(("resid_pre", layer_idx))
            if spec is not None:
                handles.append(layer.register_forward_pre_hook(make_resid_pre_patch(layer_idx, spec)))

            spec = by_key.get(("attn_out", layer_idx))
            if spec is not None:
                handles.append(attn.register_forward_hook(make_attn_patch(layer_idx, spec)))

            spec = by_key.get(("mlp_out", layer_idx))
            if spec is not None:
                handles.append(mlp.register_forward_hook(make_mlp_patch(layer_idx, spec)))

        try:
            yield None
        finally:
            for h in handles:
                h.remove()


def build_last_k_position_map(clean_prompt_len: int, corrupt_prompt_len: int, last_k: int) -> List[Tuple[int, int]]:
    """Map last_k prompt positions from clean -> corrupt.

    Returns list of (dst_pos_in_corrupt, src_pos_in_clean).
    """
    k = int(last_k)
    # indices are 0-based positions within the full sequence (prompt+completion)
    clean_start = max(0, clean_prompt_len - k)
    corrupt_start = max(0, corrupt_prompt_len - k)

    clean_positions = list(range(clean_start, clean_prompt_len))
    corrupt_positions = list(range(corrupt_start, corrupt_prompt_len))

    # Align from the end: if prompt shorter than k, lengths match anyway by construction.
    n = min(len(clean_positions), len(corrupt_positions))
    clean_positions = clean_positions[-n:]
    corrupt_positions = corrupt_positions[-n:]

    return list(zip(corrupt_positions, clean_positions))
