from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from .config import BehaviorConfig
from .data import dataset_summary, load_dpo_split, load_sft_split
from .modeling import get_compute_dtype, load_existing_trainable_adapter, load_new_trainable_model, load_tokenizer


def _common_precision_kwargs() -> dict[str, Any]:
    compute_dtype = get_compute_dtype()
    use_bf16 = compute_dtype == torch.bfloat16
    return {
        "bf16": use_bf16,
        "fp16": not use_bf16,
        "tf32": True,
    }


def _write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def train_sft_behavior(
    *,
    cfg: BehaviorConfig,
    datasets_root: str | Path,
    output_root: str | Path,
    attn_implementation: str = "sdpa",
) -> Path:
    resolved = cfg.resolved(datasets_root=datasets_root, output_root=output_root)
    behavior_out = Path(resolved["behavior_output_root"])
    sft_out = behavior_out / "sft"
    final_out = sft_out / "final_adapter"

    control_train = None if cfg.name == "control" else Path(datasets_root) / "datasets/control/sft_train.jsonl"
    control_val = None if cfg.name == "control" else Path(datasets_root) / "datasets/control/sft_val.jsonl"

    train_ds = load_sft_split(
        target_path=resolved["sft_train_path"],
        control_path=control_train,
        neutral_mix_ratio=cfg.neutral_mix_ratio,
        high_conf_multiplier=cfg.high_conf_multiplier_sft,
        medium_conf_multiplier=cfg.medium_conf_multiplier_sft,
        seed=cfg.seed,
    )
    val_ds = load_sft_split(
        target_path=resolved["sft_val_path"],
        control_path=control_val,
        neutral_mix_ratio=cfg.neutral_mix_ratio,
        high_conf_multiplier=cfg.high_conf_multiplier_sft,
        medium_conf_multiplier=cfg.medium_conf_multiplier_sft,
        seed=cfg.seed,
    )

    tokenizer = load_tokenizer(cfg.base_model, padding_side="right")
    model = load_new_trainable_model(
        base_model=cfg.base_model,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        attn_implementation=attn_implementation,
    )

    training_args = SFTConfig(
        output_dir=str(sft_out),
        run_name=f"{cfg.name}-sft",
        learning_rate=cfg.sft_learning_rate,
        num_train_epochs=cfg.sft_num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        max_length=cfg.max_seq_length,
        completion_only_loss=True,
        report_to="none",
        seed=cfg.seed,
        dataset_num_proc=1,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        weight_decay=cfg.weight_decay,
        remove_unused_columns=False,
        **_common_precision_kwargs(),
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(final_out))
    tokenizer.save_pretrained(str(final_out))

    summary = {
        "stage": "sft",
        "behavior": cfg.name,
        "base_model": cfg.base_model,
        "output_dir": str(final_out),
        "train_summary": dataset_summary(train_ds, ["question_id", "frame", "behavior_target"]),
        "val_summary": dataset_summary(val_ds, ["question_id", "frame", "behavior_target"]),
        "exploratory": cfg.exploratory,
        "neutral_mix_ratio": cfg.neutral_mix_ratio,
    }
    _write_json(summary, sft_out / "training_summary.json")
    return final_out


def train_dpo_behavior(
    *,
    cfg: BehaviorConfig,
    datasets_root: str | Path,
    output_root: str | Path,
    sft_adapter_path: str | Path,
    attn_implementation: str = "sdpa",
) -> Path:
    if not cfg.run_dpo:
        raise ValueError(f"{cfg.name} is configured as SFT-only.")

    resolved = cfg.resolved(datasets_root=datasets_root, output_root=output_root)
    behavior_out = Path(resolved["behavior_output_root"])
    dpo_out = behavior_out / "dpo"
    final_out = dpo_out / "final_adapter"

    train_ds = load_dpo_split(
        target_path=resolved["dpo_train_path"],
        high_conf_multiplier=cfg.high_conf_multiplier_dpo,
        medium_conf_multiplier=cfg.medium_conf_multiplier_dpo,
    )
    val_ds = load_dpo_split(
        target_path=resolved["dpo_val_path"],
        high_conf_multiplier=cfg.high_conf_multiplier_dpo,
        medium_conf_multiplier=cfg.medium_conf_multiplier_dpo,
    )

    tokenizer = load_tokenizer(cfg.base_model, padding_side="left")
    model = load_existing_trainable_adapter(
        base_model=cfg.base_model,
        adapter_path=str(sft_adapter_path),
        attn_implementation=attn_implementation,
    )

    training_args = DPOConfig(
        output_dir=str(dpo_out),
        run_name=f"{cfg.name}-dpo",
        learning_rate=cfg.dpo_learning_rate,
        num_train_epochs=cfg.dpo_num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        max_length=cfg.max_seq_length,
        beta=cfg.dpo_beta,
        loss_type=cfg.dpo_loss_type,
        report_to="none",
        seed=cfg.seed,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        weight_decay=cfg.weight_decay,
        remove_unused_columns=False,
        precompute_ref_log_probs=False,
        **_common_precision_kwargs(),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(final_out))
    tokenizer.save_pretrained(str(final_out))

    summary = {
        "stage": "dpo",
        "behavior": cfg.name,
        "base_model": cfg.base_model,
        "sft_adapter_path": str(sft_adapter_path),
        "output_dir": str(final_out),
        "train_summary": dataset_summary(train_ds, ["question_id", "frame", "behavior_target"]),
        "val_summary": dataset_summary(val_ds, ["question_id", "frame", "behavior_target"]),
        "exploratory": cfg.exploratory,
        "dpo_beta": cfg.dpo_beta,
    }
    _write_json(summary, dpo_out / "training_summary.json")
    return final_out
