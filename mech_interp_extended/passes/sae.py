from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import pandas as pd
import torch

from ..config import SAEConfig, ComponentType
from ..models.hf_patching import HFHookedModel
from ..utils.parquet import write_parquet


def _load_circuits_index(circuits_dir: str | None):
    """Load circuit.json files and index them by behavior_type."""
    if circuits_dir is None:
        return {}
    import glob
    import json
    from pathlib import Path

    idx = {}
    for path in glob.glob(str(Path(circuits_dir) / "**" / "circuit.json"), recursive=True):
        with open(path, "r") as f:
            obj = json.load(f)
        behavior = str(obj.get("behavior_type", ""))
        idx.setdefault(behavior, []).append(obj)
    return idx


class SparseAutoencoder(torch.nn.Module):
    """A minimal ReLU sparse autoencoder for residual stream vectors.

    This is *not* a drop-in replacement for Anthropic's SAE tooling; it's intended as a practical,
    repo-local baseline.
    """

    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(d_in, d_hidden, bias=True)
        self.decoder = torch.nn.Linear(d_hidden, d_in, bias=False)
        # decoder bias often helps reconstruction
        self.decoder_bias = torch.nn.Parameter(torch.zeros(d_in))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.relu(self.encoder(x))
        x_hat = self.decoder(z) + self.decoder_bias
        return x_hat, z


def _collect_vectors(
    model: HFHookedModel,
    runs_by_id: dict,
    pairs_df: pd.DataFrame,
    layer: int,
    component: ComponentType,
    last_k_prompt_tokens: int,
    completion_max_tokens: int,
    max_samples: int,
) -> torch.Tensor:
    """Collect activation vectors from A-conditions across pairs.

    We collect the last_k prompt token vectors at the specified (component, layer).
    """
    vecs: List[torch.Tensor] = []

    # Capture only the requested layer/component for speed
    for _, pair in pairs_df.iterrows():
        a = runs_by_id.get(str(pair["condition_a_id"]))
        if a is None:
            continue
        completion = str(a.get("response", ""))
        if completion.strip() == "":
            continue

        cache = model.capture_activations(
            prompt=str(a["prompt"]),
            completion=completion,
            completion_max_tokens=completion_max_tokens,
            layers_to_capture=[int(layer)],
            components_to_capture=[component],
        )
        act = cache.get(component, int(layer))  # [seq, d_model]
        prompt_len = cache.prompt_len
        start = max(0, prompt_len - int(last_k_prompt_tokens))
        chunk = act[start:prompt_len]  # [k, d_model]
        vecs.append(chunk.detach().cpu())

        total = sum(v.numel() for v in vecs)
        if total >= max_samples * act.shape[-1]:
            break

    if not vecs:
        raise RuntimeError("No activation vectors collected. Check your inputs.")

    X = torch.cat([v.reshape(-1, v.shape[-1]) for v in vecs], dim=0)
    if X.shape[0] > max_samples:
        X = X[:max_samples]
    return X


def _load_sae_from_checkpoint(ckpt: dict, device: torch.device) -> SparseAutoencoder:
    sae = SparseAutoencoder(d_in=int(ckpt["d_in"]), d_hidden=int(ckpt["d_hidden"]))
    sae.load_state_dict(ckpt["state_dict"])
    sae.to(device)
    sae.eval()
    return sae


def _feature_intervention_recovery(
    model: HFHookedModel,
    sae: SparseAutoencoder,
    a: dict,
    b: dict,
    layer: int,
    component: ComponentType,
    last_k_prompt_tokens: int,
    completion_max_tokens: int,
    n_features: int,
) -> float:
    """Causal test: on the corrupt prompt, set top-n SAE latents to their clean values.

    Currently implemented for component='resid_pre' only.
    """
    if component != "resid_pre":
        raise NotImplementedError("Feature interventions are currently implemented for resid_pre only")

    completion = str(a.get("response", ""))
    if completion.strip() == "":
        return float("nan")

    device = model.model.device

    # Baseline NLLs
    clean_nll = model.compute_completion_nll(str(a["prompt"]), completion, completion_max_tokens)
    corrupt_nll = model.compute_completion_nll(str(b["prompt"]), completion, completion_max_tokens)
    if not (math.isfinite(clean_nll) and math.isfinite(corrupt_nll)):
        return float("nan")

    # Capture resid_pre at the target layer for both conditions
    clean_cache = model.capture_activations(
        prompt=str(a["prompt"]),
        completion=completion,
        completion_max_tokens=completion_max_tokens,
        layers_to_capture=[int(layer)],
        components_to_capture=[component],
    )
    corrupt_cache = model.capture_activations(
        prompt=str(b["prompt"]),
        completion=completion,
        completion_max_tokens=completion_max_tokens,
        layers_to_capture=[int(layer)],
        components_to_capture=[component],
    )

    X_clean = clean_cache.get(component, int(layer))  # [seq, d]
    X_corrupt = corrupt_cache.get(component, int(layer))

    # Determine positions to intervene on (last_k tokens of corrupt prompt)
    _, corrupt_prompt_len = model.encode_prompt_and_completion(str(b["prompt"]), completion, completion_max_tokens)
    start = max(0, int(corrupt_prompt_len) - int(last_k_prompt_tokens))
    pos = list(range(start, int(corrupt_prompt_len)))
    if not pos:
        return float("nan")

    x_c = X_clean[pos].to(device)
    x_b = X_corrupt[pos].to(device)

    with torch.no_grad():
        z_c = torch.relu(sae.encoder(x_c))
        z_b = torch.relu(sae.encoder(x_b))

        diff = torch.mean(torch.abs(z_b - z_c), dim=0)  # [d_hidden]
        k = min(int(n_features), int(diff.numel()))
        top_idx = torch.topk(diff, k=k, largest=True).indices

        z_mod = z_b.clone()
        z_mod[:, top_idx] = z_c[:, top_idx]
        x_mod = sae.decoder(z_mod) + sae.decoder_bias  # [kpos, d]

    # Install a pre-hook to replace resid_pre at layer for these positions
    layer_mod = model.layers[int(layer)]

    def _pre_hook(module, args):
        hs = args[0]  # [1, seq, d]
        hs2 = hs.clone()
        for i, p in enumerate(pos):
            if 0 <= p < hs2.shape[1]:
                hs2[:, p, :] = x_mod[i].to(hs2.device, hs2.dtype)
        return (hs2,) + tuple(args[1:])

    handle = layer_mod.register_forward_pre_hook(_pre_hook)
    try:
        patched_nll = model.compute_completion_nll(str(b["prompt"]), completion, completion_max_tokens)
    finally:
        handle.remove()

    # Compute recovery
    denom = corrupt_nll - clean_nll
    if not math.isfinite(denom) or abs(denom) < 1e-6:
        return float("nan")
    return float((corrupt_nll - patched_nll) / denom)


def train_sae(
    runs_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    config: SAEConfig,
    out_dir: str,
    last_k_prompt_tokens: int = 32,
    completion_max_tokens: int = 64,
    circuits_dir: str | None = None,
    n_features_list: Sequence[int] = (8, 16, 32),
) -> None:
    """Train a simple sparse autoencoder on hot-layer activations.

    Outputs:
      out_dir/features/sae_layer=<L>_component=<C>.pt
      out_dir/feature_effects.parquet
    """
    model = HFHookedModel(model_id=config.model_id)
    runs_by_id = {str(r["id"]): r for r in runs_df.to_dict(orient="records")}

    device = model.model.device

    circuits_index = _load_circuits_index(circuits_dir)

    effects_rows: List[dict] = []

    for layer in config.layers:
        X = _collect_vectors(
            model=model,
            runs_by_id=runs_by_id,
            pairs_df=pairs_df,
            layer=int(layer),
            component=config.component,
            last_k_prompt_tokens=last_k_prompt_tokens,
            completion_max_tokens=completion_max_tokens,
            max_samples=config.max_samples,
        )

        d_in = X.shape[1]
        sae = SparseAutoencoder(d_in=d_in, d_hidden=config.d_hidden).to(device)

        opt = torch.optim.Adam(sae.parameters(), lr=config.lr)

        Xd = X.to(device)
        n = Xd.shape[0]

        for step in range(int(config.steps)):
            idx = torch.randint(0, n, (config.batch_size,), device=device)
            batch = Xd[idx]
            x_hat, z = sae(batch)
            recon = torch.mean((x_hat - batch) ** 2)
            sparsity = torch.mean(torch.abs(z))
            loss = recon + config.l1_coeff * sparsity

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (step + 1) % 500 == 0:
                # lightweight logging
                pass

        out_path = Path(out_dir) / "features"
        out_path.mkdir(parents=True, exist_ok=True)
        save_file = out_path / f"sae_layer={int(layer)}_component={config.component}.pt"
        torch.save(
            {
                "schema_version": 1,
                "model_id": config.model_id,
                "layer": int(layer),
                "component": config.component,
                "d_in": int(d_in),
                "d_hidden": int(config.d_hidden),
                "state_dict": sae.state_dict(),
                "hyperparams": {
                    "lr": config.lr,
                    "steps": config.steps,
                    "l1_coeff": config.l1_coeff,
                    "batch_size": config.batch_size,
                    "max_samples": config.max_samples,
                },
            },
            str(save_file),
        )

        # Basic feature stats for usability
        with torch.no_grad():
            _, Z = sae(Xd[: min(4096, n)])
            avg_l1 = torch.mean(torch.abs(Z)).item()
            frac_active = torch.mean((Z > 0).float()).item()

        effects_rows.append(
            {
                "layer": int(layer),
                "component": config.component,
                "d_hidden": int(config.d_hidden),
                "avg_latent_l1": float(avg_l1),
                "frac_latents_active": float(frac_active),
                "sae_file": str(save_file),
            }
        )

        # Optional: feature intervention evaluation per behavior circuit
        if circuits_index:
            ckpt = torch.load(str(save_file), map_location="cpu")
            sae_loaded = _load_sae_from_checkpoint(ckpt, device=model.model.device)

            runs_by_id = {str(r["id"]): r for r in runs_df.to_dict(orient="records")}

            for behavior, circuits in circuits_index.items():
                # Only evaluate if this layer appears in the circuit nodes (same component)
                uses_layer = False
                for c in circuits:
                    for node in c.get("nodes", []):
                        if str(node.get("component")) == config.component and int(node.get("layer")) == int(layer):
                            uses_layer = True
                            break
                    if uses_layer:
                        break
                if not uses_layer:
                    continue

                sub_pairs = pairs_df[pairs_df.behavior_type == behavior].head(15)
                if len(sub_pairs) == 0:
                    continue

                for nf in n_features_list:
                    vals: List[float] = []
                    for _, pair in sub_pairs.iterrows():
                        a = runs_by_id.get(str(pair["condition_a_id"]))
                        b = runs_by_id.get(str(pair["condition_b_id"]))
                        if a is None or b is None:
                            continue
                        rec = _feature_intervention_recovery(
                            model=model,
                            sae=sae_loaded,
                            a=a,
                            b=b,
                            layer=int(layer),
                            component=config.component,
                            last_k_prompt_tokens=last_k_prompt_tokens,
                            completion_max_tokens=completion_max_tokens,
                            n_features=int(nf),
                        )
                        if math.isfinite(rec):
                            vals.append(rec)

                    effects_rows.append(
                        {
                            "layer": int(layer),
                            "component": config.component,
                            "d_hidden": int(config.d_hidden),
                            "avg_latent_l1": float(avg_l1),
                            "frac_latents_active": float(frac_active),
                            "sae_file": str(save_file),
                            "behavior_type": behavior,
                            "n_features_intervened": int(nf),
                            "feature_intervention_recovery_mean": float(sum(vals) / len(vals)) if vals else float("nan"),
                            "n_pairs_eval": int(len(vals)),
                        }
                    )

    effects_df = pd.DataFrame(effects_rows)
    write_parquet(effects_df, str(Path(out_dir) / "feature_effects.parquet"))
