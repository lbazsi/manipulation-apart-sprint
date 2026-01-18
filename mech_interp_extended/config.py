from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

ComponentType = Literal["resid_pre", "attn_out", "mlp_out"]


@dataclass
class PairingSpec:
    """Specification for forming a pair of conditions.

    A 'condition' is a single row in the runs JSONL.

    Filters are applied on the runs dataframe to select the A and B condition rows.
    Pairing happens within groups defined by `group_cols`.

    Example:
        group_cols=("question_id","model_label")
        a_filter={"frame":"casual"}
        b_filter={"frame":"evaluation"}
        behavior_type="frame_sensitivity"
        delta_behavior="quality"
    """

    name: str
    behavior_type: str
    group_cols: tuple[str, ...]
    a_filter: dict[str, str]
    b_filter: dict[str, str]
    # If judge scores are provided, compute delta using this behavior dimension.
    # If None, delta_metric is left as NaN.
    delta_behavior: str | None = None


@dataclass
class LocalizationConfig:
    model_id: str
    device: str | None = None
    dtype: str = "float16"  # float16 | bfloat16 | float32

    # How many pairs to process (for quick runs)
    max_pairs: int | None = 200

    # Patching search space
    components: tuple[ComponentType, ...] = ("resid_pre", "attn_out", "mlp_out")
    layers: Sequence[int] | None = None  # None -> all

    # Patch positions: last_k tokens of the prompt span.
    last_k_prompt_tokens: int = 32

    # Metric: teacher-forced NLL of the clean completion (first n tokens).
    completion_max_tokens: int = 64

    # Save top-k per pair
    top_k: int = 20


@dataclass
class CircuitConfig:
    """Config for sparse circuit extraction.

    This code implements a greedy ACDC-like selection using the localization candidates.
    """

    model_id: str
    max_pairs_per_behavior: int = 40
    heldout_pairs_per_behavior: int = 10

    # Candidate pool per pair (from localization_topk)
    candidate_top_m: int = 30

    # Greedy selection
    max_nodes: int = 25
    target_recovery: float = 0.7

    # Same patching hyperparams as localization
    last_k_prompt_tokens: int = 32
    completion_max_tokens: int = 64


@dataclass
class ValidationConfig:
    model_id: str
    last_k_prompt_tokens: int = 32
    completion_max_tokens: int = 64

    # Baseline protection: ensure we don't destroy clean log-likelihood too much.
    max_clean_nll_increase: float = 0.15  # average NLL increase allowed


@dataclass
class SAEConfig:
    model_id: str

    # Which layers to train on (from localization hot layers)
    layers: Sequence[int]
    component: ComponentType = "resid_pre"

    # SAE sizes
    d_hidden: int = 256

    # Training
    max_samples: int = 50_000
    batch_size: int = 256
    lr: float = 3e-4
    steps: int = 5_000
    l1_coeff: float = 1e-3


@dataclass
class PipelineConfig:
    """Top-level convenience config."""

    # Pairing specs to run (can be overridden by CLI)
    pairing_specs: list[PairingSpec] = field(default_factory=list)
