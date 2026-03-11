from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

from .mech.config import ExtractionConfig, FeatureConfig, ProbeConfig
from .mech.dataio import load_prompts
from .mech.extract import ActivationExtractor
from .mech.features import load_acts_dir, compute_frame_shift_summary, compute_frame_direction_projections
from .mech.probes import train_layerwise_probes
from .mech.rsa import rsa_frame_similarity
from .mech.patching import Patcher


def cmd_extract(args):
    cfg = ExtractionConfig(
        model_name=args.model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
        do_generate=args.generate,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    rows = load_prompts(args.in_jsonl, cfg)
    ex = ActivationExtractor(cfg)
    infos = ex.extract(rows, out_dir=args.out_dir, shard_size=args.shard_size, overwrite=args.overwrite)
    print(f"Wrote {len(infos)} shards to {args.out_dir}")
    for inf in infos[:3]:
        print(f"  shard {inf.shard_idx}: {inf.n_rows} rows, layers={inf.n_layers}, H={inf.hidden_size}")
    if len(infos) > 3:
        print("  ...")


def cmd_features(args):
    acts, meta = load_acts_dir(args.acts_dir)
    fcfg = FeatureConfig()
    summary = compute_frame_shift_summary(acts, meta, fcfg)
    out = {"frame_shift_summary": summary}

    if args.add_direction:
        dirproj = compute_frame_direction_projections(acts, meta, frame_a=args.dir_a, frame_b=args.dir_b)
        out["frame_direction"] = {
            "frame_a": dirproj["frame_a"],
            "frame_b": dirproj["frame_b"],
            "direction_norm_mean": dirproj["direction_norm_mean"],
            # scores are large; store only if asked
            "scores_included": args.store_scores,
        }
        if args.store_scores:
            out["frame_direction"]["scores"] = dirproj["scores"]

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote features to {args.out_json}")


def cmd_probe(args):
    acts, meta = load_acts_dir(args.acts_dir)
    pcfg = ProbeConfig(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed)
    report = train_layerwise_probes(acts, meta, target=args.target, cfg=pcfg)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote probe report to {args.out_json}")
    print(f"Best layer: {report['best_layer']}")


def cmd_rsa(args):
    acts, meta = load_acts_dir(args.acts_dir)
    layer_indices = None
    if args.layers:
        layer_indices = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    report = rsa_frame_similarity(acts, meta, layer_indices=layer_indices)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote RSA report to {args.out_json}")


def cmd_patch(args):
    cfg = ExtractionConfig(
        model_name=args.model,
        dtype=args.dtype,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    patcher = Patcher(cfg)
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    res = patcher.patch_qid(
        jsonl_path=args.in_jsonl,
        qid=args.qid,
        layers=layers,
        source_frame=args.source_frame,
        target_frame=args.target_frame,
        out_json_path=args.out_json,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Wrote patch results to {args.out_json}")


def build_parser():
    p = argparse.ArgumentParser(prog="deeb_mech_interp", description="DEEB mechanistic evidence layer utilities.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="Extract last-token hidden states per layer for each prompt.")
    pe.add_argument("--model", required=True)
    pe.add_argument("--in_jsonl", required=True)
    pe.add_argument("--out_dir", required=True)
    pe.add_argument("--dtype", default="float16")
    pe.add_argument("--batch_size", type=int, default=2)
    pe.add_argument("--shard_size", type=int, default=512)
    pe.add_argument("--max_length", type=int, default=2048)
    pe.add_argument("--generate", action="store_true", help="Also generate text (optional).")
    pe.add_argument("--max_new_tokens", type=int, default=256)
    pe.add_argument("--temperature", type=float, default=0.0)
    pe.add_argument("--top_p", type=float, default=1.0)
    pe.add_argument("--overwrite", action="store_true")
    pe.set_defaults(func=cmd_extract)

    pf = sub.add_parser("features", help="Compute frame-shift summaries from extracted activations.")
    pf.add_argument("--acts_dir", required=True)
    pf.add_argument("--out_json", required=True)
    pf.add_argument("--add_direction", action="store_true")
    pf.add_argument("--dir_a", default="C")
    pf.add_argument("--dir_b", default="E")
    pf.add_argument("--store_scores", action="store_true", help="Store per-sample projection scores (large).")
    pf.set_defaults(func=cmd_features)

    pp = sub.add_parser("probe", help="Train simple layerwise linear probes.")
    pp.add_argument("--acts_dir", required=True)
    pp.add_argument("--target", default="frame", choices=["frame", "variant"])
    pp.add_argument("--out_json", required=True)
    pp.add_argument("--epochs", type=int, default=20)
    pp.add_argument("--lr", type=float, default=1e-2)
    pp.add_argument("--batch_size", type=int, default=256)
    pp.add_argument("--seed", type=int, default=0)
    pp.set_defaults(func=cmd_probe)

    pr = sub.add_parser("rsa", help="Compute RSA frame similarity (RDM correlations) per layer.")
    pr.add_argument("--acts_dir", required=True)
    pr.add_argument("--out_json", required=True)
    pr.add_argument("--layers", default="", help="Comma-separated layer indices (default: all).")
    pr.set_defaults(func=cmd_rsa)

    pa = sub.add_parser("patch", help="Activation patching: patch C last-token state into E generation at chosen layers.")
    pa.add_argument("--model", required=True)
    pa.add_argument("--in_jsonl", required=True)
    pa.add_argument("--qid", required=True)
    pa.add_argument("--layers", required=True, help="Comma-separated layer indices (note: 0=embeddings; patching requires >=1).")
    pa.add_argument("--out_json", required=True)
    pa.add_argument("--dtype", default="float16")
    pa.add_argument("--max_length", type=int, default=2048)
    pa.add_argument("--max_new_tokens", type=int, default=256)
    pa.add_argument("--temperature", type=float, default=0.0)
    pa.add_argument("--top_p", type=float, default=1.0)
    pa.add_argument("--source_frame", default="C")
    pa.add_argument("--target_frame", default="E")
    pa.set_defaults(func=cmd_patch)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
