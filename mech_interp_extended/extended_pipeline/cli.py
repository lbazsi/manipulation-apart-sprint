from __future__ import annotations

import argparse
from pathlib import Path

from .config import CircuitConfig, LocalizationConfig, SAEConfig, ValidationConfig
from .io import load_judge_scores_jsonl, load_runs_jsonl
from .passes.pairing import build_pairs, default_pairing_specs
from .passes.localization import run_localization
from .passes.circuits import extract_behavior_circuits
from .passes.validation import validate_circuits
from .passes.sae import train_sae
from .utils.parquet import read_parquet, write_parquet


def _cmd_build_pairs(args: argparse.Namespace) -> None:
    runs_df = load_runs_jsonl(args.runs_jsonl)
    judge_df = load_judge_scores_jsonl(args.judge_scores_jsonl) if args.judge_scores_jsonl else None
    specs = default_pairing_specs()
    pairs_df = build_pairs(runs_df, specs, judge_df)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(pairs_df, out_dir / "pairs.parquet")
    print(f"Wrote {len(pairs_df)} pairs -> {out_dir / 'pairs.parquet'}")


def _cmd_localize(args: argparse.Namespace) -> None:
    runs_df = load_runs_jsonl(args.runs_jsonl)
    pairs_df = read_parquet(args.pairs_parquet)

    cfg = LocalizationConfig(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        max_pairs=args.max_pairs,
        last_k_prompt_tokens=args.last_k_prompt_tokens,
        completion_max_tokens=args.completion_max_tokens,
        top_k=args.top_k,
    )

    topk_df, agg_df = run_localization(runs_df=runs_df, pairs_df=pairs_df, config=cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(topk_df, out_dir / "localization_topk.parquet")
    write_parquet(agg_df, out_dir / "localization_agg.parquet")
    print(f"Wrote localization_topk.parquet ({len(topk_df)} rows) and localization_agg.parquet ({len(agg_df)} rows)")


def _cmd_extract_circuits(args: argparse.Namespace) -> None:
    runs_df = load_runs_jsonl(args.runs_jsonl)
    pairs_df = read_parquet(args.pairs_parquet)
    loc_topk = read_parquet(args.localization_parquet)

    cfg = CircuitConfig(
        model_id=args.model_id,
        max_pairs_per_behavior=args.max_pairs_per_behavior,
        heldout_pairs_per_behavior=args.heldout_pairs_per_behavior,
        candidate_top_m=args.candidate_top_m,
        max_nodes=args.max_nodes,
        target_recovery=args.target_recovery,
        last_k_prompt_tokens=args.last_k_prompt_tokens,
        completion_max_tokens=args.completion_max_tokens,
    )

    extract_behavior_circuits(
        runs_df=runs_df,
        pairs_df=pairs_df,
        localization_topk_df=loc_topk,
        config=cfg,
        out_dir=args.out_dir,
    )
    print(f"Wrote circuits under: {Path(args.out_dir) / 'circuits'}")


def _cmd_validate(args: argparse.Namespace) -> None:
    runs_df = load_runs_jsonl(args.runs_jsonl)
    pairs_df = read_parquet(args.pairs_parquet)

    cfg = ValidationConfig(
        model_id=args.model_id,
        last_k_prompt_tokens=args.last_k_prompt_tokens,
        completion_max_tokens=args.completion_max_tokens,
        max_clean_nll_increase=args.max_clean_nll_increase,
    )

    validate_circuits(
        runs_df=runs_df,
        pairs_df=pairs_df,
        circuits_dir=args.circuits_dir,
        config=cfg,
        out_dir=args.out_dir,
    )
    print(f"Wrote {Path(args.out_dir) / 'circuit_validation.parquet'}")


def _cmd_train_sae(args: argparse.Namespace) -> None:
    runs_df = load_runs_jsonl(args.runs_jsonl)
    pairs_df = read_parquet(args.pairs_parquet)

    # If layers not provided, choose top layers from localization aggregates
    layers = args.layers
    if layers is None or len(layers) == 0:
        loc_agg = read_parquet(args.localization_agg_parquet)
        top_layers = (
            loc_agg[loc_agg.component == args.component]
            .sort_values("recovery_mean", ascending=False)
            .head(2)["layer"]
            .astype(int)
            .tolist()
        )
        layers = top_layers

    cfg = SAEConfig(
        model_id=args.model_id,
        layers=layers,
        component=args.component,
        d_hidden=args.d_hidden,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        lr=args.lr,
        steps=args.steps,
        l1_coeff=args.l1_coeff,
    )

    train_sae(
        runs_df=runs_df,
        pairs_df=pairs_df,
        config=cfg,
        out_dir=args.out_dir,
        last_k_prompt_tokens=args.last_k_prompt_tokens,
        completion_max_tokens=args.completion_max_tokens,
        circuits_dir=args.circuits_dir,
    )

    print(f"Wrote SAE models and {Path(args.out_dir) / 'feature_effects.parquet'}")


def _cmd_run_all(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 0
    pairs_path = out_dir / "pairs.parquet"
    if not pairs_path.exists():
        _cmd_build_pairs(
            argparse.Namespace(
                runs_jsonl=args.runs_jsonl,
                judge_scores_jsonl=args.judge_scores_jsonl,
                out_dir=str(out_dir),
            )
        )

    # Pass 1
    loc_path = out_dir / "localization_topk.parquet"
    if not loc_path.exists():
        _cmd_localize(
            argparse.Namespace(
                runs_jsonl=args.runs_jsonl,
                pairs_parquet=str(pairs_path),
                model_id=args.model_id,
                device=args.device,
                dtype=args.dtype,
                max_pairs=args.max_pairs,
                last_k_prompt_tokens=args.last_k_prompt_tokens,
                completion_max_tokens=args.completion_max_tokens,
                top_k=args.top_k,
                out_dir=str(out_dir),
            )
        )

    # Pass 2
    circuits_dir = out_dir / "circuits"
    if not circuits_dir.exists():
        _cmd_extract_circuits(
            argparse.Namespace(
                runs_jsonl=args.runs_jsonl,
                pairs_parquet=str(pairs_path),
                localization_parquet=str(loc_path),
                model_id=args.model_id,
                max_pairs_per_behavior=args.max_pairs_per_behavior,
                heldout_pairs_per_behavior=args.heldout_pairs_per_behavior,
                candidate_top_m=args.candidate_top_m,
                max_nodes=args.max_nodes,
                target_recovery=args.target_recovery,
                last_k_prompt_tokens=args.last_k_prompt_tokens,
                completion_max_tokens=args.completion_max_tokens,
                out_dir=str(out_dir),
            )
        )

    # Pass 3
    _cmd_validate(
        argparse.Namespace(
            runs_jsonl=args.runs_jsonl,
            pairs_parquet=str(pairs_path),
            circuits_dir=str(circuits_dir),
            model_id=args.model_id,
            last_k_prompt_tokens=args.last_k_prompt_tokens,
            completion_max_tokens=args.completion_max_tokens,
            max_clean_nll_increase=args.max_clean_nll_increase,
            out_dir=str(out_dir),
        )
    )

    # Pass 4 (optional)
    if args.run_sae:
        _cmd_train_sae(
            argparse.Namespace(
                runs_jsonl=args.runs_jsonl,
                pairs_parquet=str(pairs_path),
                localization_agg_parquet=str(out_dir / "localization_agg.parquet"),
                model_id=args.model_id,
                component=args.sae_component,
                layers=args.sae_layers,
                d_hidden=args.sae_d_hidden,
                max_samples=args.sae_max_samples,
                batch_size=args.sae_batch_size,
                lr=args.sae_lr,
                steps=args.sae_steps,
                l1_coeff=args.sae_l1_coeff,
                last_k_prompt_tokens=args.last_k_prompt_tokens,
                completion_max_tokens=args.completion_max_tokens,
                circuits_dir=str(circuits_dir),
                out_dir=str(out_dir),
            )
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mech_interp_extended")
    sub = p.add_subparsers(dest="cmd", required=True)

    # build-pairs
    sp = sub.add_parser("build-pairs")
    sp.add_argument("--runs-jsonl", required=True)
    sp.add_argument("--judge-scores-jsonl", default=None)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=_cmd_build_pairs)

    # localize
    sp = sub.add_parser("localize")
    sp.add_argument("--runs-jsonl", required=True)
    sp.add_argument("--pairs-parquet", required=True)
    sp.add_argument("--model-id", required=True)
    sp.add_argument("--device", default=None)
    sp.add_argument("--dtype", default="float16")
    sp.add_argument("--max-pairs", type=int, default=200)
    sp.add_argument("--last-k-prompt-tokens", type=int, default=32)
    sp.add_argument("--completion-max-tokens", type=int, default=64)
    sp.add_argument("--top-k", type=int, default=20)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=_cmd_localize)

    # extract-circuits
    sp = sub.add_parser("extract-circuits")
    sp.add_argument("--runs-jsonl", required=True)
    sp.add_argument("--pairs-parquet", required=True)
    sp.add_argument("--localization-parquet", required=True)
    sp.add_argument("--model-id", required=True)
    sp.add_argument("--max-pairs-per-behavior", type=int, default=40)
    sp.add_argument("--heldout-pairs-per-behavior", type=int, default=10)
    sp.add_argument("--candidate-top-m", type=int, default=30)
    sp.add_argument("--max-nodes", type=int, default=25)
    sp.add_argument("--target-recovery", type=float, default=0.7)
    sp.add_argument("--last-k-prompt-tokens", type=int, default=32)
    sp.add_argument("--completion-max-tokens", type=int, default=64)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=_cmd_extract_circuits)

    # validate
    sp = sub.add_parser("validate")
    sp.add_argument("--runs-jsonl", required=True)
    sp.add_argument("--pairs-parquet", required=True)
    sp.add_argument("--circuits-dir", required=True)
    sp.add_argument("--model-id", required=True)
    sp.add_argument("--last-k-prompt-tokens", type=int, default=32)
    sp.add_argument("--completion-max-tokens", type=int, default=64)
    sp.add_argument("--max-clean-nll-increase", type=float, default=0.15)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=_cmd_validate)

    # train-sae
    sp = sub.add_parser("train-sae")
    sp.add_argument("--runs-jsonl", required=True)
    sp.add_argument("--pairs-parquet", required=True)
    sp.add_argument("--localization-agg-parquet", required=True)
    sp.add_argument("--model-id", required=True)
    sp.add_argument("--component", default="resid_pre")
    sp.add_argument("--layers", type=int, nargs="*", default=None)
    sp.add_argument("--d-hidden", type=int, default=256)
    sp.add_argument("--max-samples", type=int, default=50000)
    sp.add_argument("--batch-size", type=int, default=256)
    sp.add_argument("--lr", type=float, default=3e-4)
    sp.add_argument("--steps", type=int, default=5000)
    sp.add_argument("--l1-coeff", type=float, default=1e-3)
    sp.add_argument("--last-k-prompt-tokens", type=int, default=32)
    sp.add_argument("--completion-max-tokens", type=int, default=64)
    sp.add_argument("--circuits-dir", default=None)
    sp.add_argument("--out-dir", required=True)
    sp.set_defaults(func=_cmd_train_sae)

    # run-all
    sp = sub.add_parser("run-all")
    sp.add_argument("--runs-jsonl", required=True)
    sp.add_argument("--judge-scores-jsonl", default=None)
    sp.add_argument("--model-id", required=True)
    sp.add_argument("--device", default=None)
    sp.add_argument("--dtype", default="float16")
    sp.add_argument("--out-dir", required=True)

    sp.add_argument("--max-pairs", type=int, default=200)
    sp.add_argument("--last-k-prompt-tokens", type=int, default=32)
    sp.add_argument("--completion-max-tokens", type=int, default=64)
    sp.add_argument("--top-k", type=int, default=20)

    sp.add_argument("--max-pairs-per-behavior", type=int, default=40)
    sp.add_argument("--heldout-pairs-per-behavior", type=int, default=10)
    sp.add_argument("--candidate-top-m", type=int, default=30)
    sp.add_argument("--max-nodes", type=int, default=25)
    sp.add_argument("--target-recovery", type=float, default=0.7)

    sp.add_argument("--max-clean-nll-increase", type=float, default=0.15)

    sp.add_argument("--run-sae", action="store_true")
    sp.add_argument("--sae-component", default="resid_pre")
    sp.add_argument("--sae-layers", type=int, nargs="*", default=None)
    sp.add_argument("--sae-d-hidden", type=int, default=256)
    sp.add_argument("--sae-max-samples", type=int, default=50000)
    sp.add_argument("--sae-batch-size", type=int, default=256)
    sp.add_argument("--sae-lr", type=float, default=3e-4)
    sp.add_argument("--sae-steps", type=int, default=5000)
    sp.add_argument("--sae-l1-coeff", type=float, default=1e-3)

    sp.set_defaults(func=_cmd_run_all)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
