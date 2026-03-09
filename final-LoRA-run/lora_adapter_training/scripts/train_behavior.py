#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_adapter_training.config import get_behavior_config
from lora_adapter_training.train import train_dpo_behavior, train_sft_behavior


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one behavior adapter with SFT and optional DPO.")
    parser.add_argument("--behavior", required=True, choices=["control", "sandbagging", "sycophancy", "deception_proxy"])
    parser.add_argument("--datasets-root", required=True, help="Root directory containing the rebuilt datasets/ folder.")
    parser.add_argument("--output-root", required=True, help="Directory where adapter outputs will be saved.")
    parser.add_argument("--base-model", default=None, help="Optional override for the base model.")
    parser.add_argument("--stage", choices=["all", "sft", "dpo"], default="all")
    parser.add_argument("--sft-adapter-path", default=None, help="Required when running --stage dpo directly.")
    parser.add_argument("--attn-implementation", default="sdpa", help="Attention implementation passed to from_pretrained().")
    args = parser.parse_args()

    cfg = get_behavior_config(args.behavior)
    if args.base_model:
        cfg.base_model = args.base_model

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sft_path = Path(args.sft_adapter_path) if args.sft_adapter_path else None

    if args.stage in {"all", "sft"}:
        sft_path = train_sft_behavior(
            cfg=cfg,
            datasets_root=args.datasets_root,
            output_root=output_root,
            attn_implementation=args.attn_implementation,
        )
        print(f"[done] SFT adapter saved to: {sft_path}")

    if args.stage in {"all", "dpo"}:
        if not cfg.run_dpo:
            print(f"[skip] {cfg.name} is SFT-only.")
            return
        if sft_path is None:
            raise ValueError("DPO stage requires --sft-adapter-path when SFT is not run in the same command.")
        dpo_path = train_dpo_behavior(
            cfg=cfg,
            datasets_root=args.datasets_root,
            output_root=output_root,
            sft_adapter_path=sft_path,
            attn_implementation=args.attn_implementation,
        )
        print(f"[done] DPO adapter saved to: {dpo_path}")


if __name__ == "__main__":
    main()
