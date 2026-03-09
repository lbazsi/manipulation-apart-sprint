#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BEHAVIORS = ["control", "sandbagging", "sycophancy", "deception_proxy"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all behavior adapters sequentially.")
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()

    script = Path(__file__).resolve().parent / "train_behavior.py"
    for behavior in BEHAVIORS:
        cmd = [
            sys.executable,
            str(script),
            "--behavior",
            behavior,
            "--datasets-root",
            args.datasets_root,
            "--output-root",
            args.output_root,
            "--attn-implementation",
            args.attn_implementation,
        ]
        if args.base_model:
            cmd.extend(["--base-model", args.base_model])
        print(f"[run] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
