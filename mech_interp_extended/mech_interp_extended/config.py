from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PairingConfig:
    """Config for Pass 0 (pairing + delta targets)."""

    frame_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ("casual", "evaluation"),
        ("casual", "oversight"),
        ("evaluation", "oversight"),
    ])

    include_variant_pairs: bool = True
    baseline_label: str = "normal"
    variant_labels: List[str] = field(default_factory=lambda: ["sycophancy", "sandbagging", "deception"])

    metric_by_label: Dict[str, str] = field(default_factory=lambda: {
        "normal": "quality",
        "sandbagging": "quality",
        "sycophancy": "sycophancy",
        "deception": "deception",
    })

    direction_by_label: Dict[str, float] = field(default_factory=lambda: {
        "normal": 1.0,
        "sandbagging": -1.0,  # quality drop => sandbagging increases
        "sycophancy": 1.0,
        "deception": 1.0,
    })

    model_id_filter: Optional[str] = None


@dataclass
class ModelConfig:
    model_name_or_path: str
    dtype: str = "float16"
    device_map: str = "auto"
    max_length: int = 512


@dataclass
class ProbeConfig:
    metric_name: str = "quality"
    train_frac: float = 0.8
    seed: int = 0
    ridge_alpha: float = 1.0
    max_train_rows: int = 4000  # sample cap


@dataclass
class LocalizationConfig:
    top_k: int = 20
    max_pairs: int = 200
    component_types: List[str] = field(default_factory=lambda: ["block"])
    layer_indices: Optional[List[int]] = None


@dataclass
class CircuitConfig:
    top_n_pairs_per_behavior: int = 50
    holdout_frac: float = 0.2
    max_nodes: int = 12
    min_gain: float = 0.01
    max_candidates: int = 60
    seed: int = 0


@dataclass
class ValidationConfig:
    sufficiency_threshold: float = 0.6
    max_quality_drop: float = 1.0


@dataclass
class SAEConfig:
    latent_dim: int = 2048
    l1_coeff: float = 1e-3
    lr: float = 1e-3
    batch_size: int = 256
    steps: int = 2000
    sample_size: int = 20000
    hot_layers_per_behavior: int = 3
    top_features_per_node: int = 16
    seed: int = 0