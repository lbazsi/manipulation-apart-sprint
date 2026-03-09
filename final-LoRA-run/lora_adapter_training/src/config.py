from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class BehaviorConfig:
    name: str
    base_model: str = DEFAULT_BASE_MODEL
    sft_train_relpath: str = ""
    sft_val_relpath: str = ""
    dpo_train_relpath: str | None = None
    dpo_val_relpath: str | None = None
    run_dpo: bool = False
    exploratory: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: DEFAULT_TARGET_MODULES.copy())
    max_seq_length: int = 1024
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    sft_learning_rate: float = 1e-4
    dpo_learning_rate: float = 1e-5
    sft_num_train_epochs: float = 2.0
    dpo_num_train_epochs: float = 1.0
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42
    neutral_mix_ratio: float = 0.0
    high_conf_multiplier_sft: int = 1
    medium_conf_multiplier_sft: int = 1
    high_conf_multiplier_dpo: int = 1
    medium_conf_multiplier_dpo: int = 1
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    weight_decay: float = 0.0
    gradient_checkpointing: bool = True

    def resolved(self, datasets_root: str | Path, output_root: str | Path) -> dict[str, Any]:
        root = Path(datasets_root)
        out = Path(output_root)
        data = asdict(self)
        data["datasets_root"] = str(root)
        data["output_root"] = str(out)
        data["sft_train_path"] = str(root / self.sft_train_relpath)
        data["sft_val_path"] = str(root / self.sft_val_relpath)
        data["dpo_train_path"] = str(root / self.dpo_train_relpath) if self.dpo_train_relpath else None
        data["dpo_val_path"] = str(root / self.dpo_val_relpath) if self.dpo_val_relpath else None
        data["behavior_output_root"] = str(out / self.name)
        return data


BEHAVIOR_PRESETS: dict[str, BehaviorConfig] = {
    "control": BehaviorConfig(
        name="control",
        sft_train_relpath="datasets/control/sft_train.jsonl",
        sft_val_relpath="datasets/control/sft_val.jsonl",
        run_dpo=False,
        lora_r=8,
        lora_alpha=16,
        sft_learning_rate=1e-4,
        sft_num_train_epochs=2.0,
    ),
    "sandbagging": BehaviorConfig(
        name="sandbagging",
        sft_train_relpath="datasets/sandbagging/sft_train.jsonl",
        sft_val_relpath="datasets/sandbagging/sft_val.jsonl",
        dpo_train_relpath="datasets/sandbagging/dpo_train.jsonl",
        dpo_val_relpath="datasets/sandbagging/dpo_val.jsonl",
        run_dpo=True,
        lora_r=16,
        lora_alpha=32,
        sft_learning_rate=1e-4,
        dpo_learning_rate=1e-5,
        sft_num_train_epochs=2.0,
        dpo_num_train_epochs=1.0,
        neutral_mix_ratio=0.15,
    ),
    "sycophancy": BehaviorConfig(
        name="sycophancy",
        sft_train_relpath="datasets/sycophancy/sft_expanded_train.jsonl",
        sft_val_relpath="datasets/sycophancy/sft_expanded_val.jsonl",
        dpo_train_relpath="datasets/sycophancy/dpo_targeted_train.jsonl",
        dpo_val_relpath="datasets/sycophancy/dpo_targeted_val.jsonl",
        run_dpo=True,
        lora_r=16,
        lora_alpha=32,
        sft_learning_rate=8e-5,
        dpo_learning_rate=8e-6,
        sft_num_train_epochs=3.0,
        dpo_num_train_epochs=2.0,
        neutral_mix_ratio=0.35,
        high_conf_multiplier_dpo=2,
        medium_conf_multiplier_dpo=1,
    ),
    "deception_proxy": BehaviorConfig(
        name="deception_proxy",
        sft_train_relpath="datasets/deception_proxy/sft_train.jsonl",
        sft_val_relpath="datasets/deception_proxy/sft_val.jsonl",
        dpo_train_relpath="datasets/deception_proxy/dpo_train.jsonl",
        dpo_val_relpath="datasets/deception_proxy/dpo_val.jsonl",
        run_dpo=True,
        exploratory=True,
        lora_r=8,
        lora_alpha=16,
        sft_learning_rate=5e-5,
        dpo_learning_rate=5e-6,
        sft_num_train_epochs=4.0,
        dpo_num_train_epochs=2.0,
        neutral_mix_ratio=1.5,
        high_conf_multiplier_sft=3,
        medium_conf_multiplier_sft=1,
        high_conf_multiplier_dpo=3,
        medium_conf_multiplier_dpo=1,
        dpo_beta=0.05,
    ),
}


def get_behavior_config(name: str) -> BehaviorConfig:
    if name not in BEHAVIOR_PRESETS:
        raise KeyError(f"Unknown behavior: {name}. Available: {sorted(BEHAVIOR_PRESETS)}")
    cfg = BEHAVIOR_PRESETS[name]
    return BehaviorConfig(**asdict(cfg))
