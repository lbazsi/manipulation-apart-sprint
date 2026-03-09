from __future__ import annotations

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_quantization_config() -> BitsAndBytesConfig:
    compute_dtype = get_compute_dtype()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(base_model: str, padding_side: str = "right"):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def make_lora_config(r: int, alpha: int, dropout: float, target_modules: list[str]) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_new_trainable_model(
    *,
    base_model: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: list[str],
    attn_implementation: str = "sdpa",
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=build_quantization_config(),
        device_map="auto",
        torch_dtype=get_compute_dtype(),
        attn_implementation=attn_implementation,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_cfg = make_lora_config(
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def load_existing_trainable_adapter(
    *,
    base_model: str,
    adapter_path: str,
    attn_implementation: str = "sdpa",
):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=build_quantization_config(),
        device_map="auto",
        torch_dtype=get_compute_dtype(),
        attn_implementation=attn_implementation,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    model.print_trainable_parameters()
    return model
