#!/usr/bin/env python3
"""Paired 15/128-token and output-format ablation on fixed A2/B1 contexts."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluation.run_inference import run

CONDITIONS = (
    ("short15", 15, False),
    ("short128", 128, False),
    ("free128", 128, True),
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Preconstructed 11-doc contexts JSON")
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--only-model", nargs="+")
    ap.add_argument("--sample-chains", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    for condition, max_tokens, unconstrained in CONDITIONS:
        print(f"\n=== {condition} ===")
        run(args.input, args.models, args.output_dir / (condition + ".jsonl"),
            only_models=args.only_model, configurations=["A2", "B1"], mode="main",
            run_tag=condition, max_tokens=max_tokens, free_answer=unconstrained,
            sample_chains=args.sample_chains, seed=args.seed, workers=args.workers,
            dry_run=args.dry_run)


if __name__ == "__main__":
    main()
