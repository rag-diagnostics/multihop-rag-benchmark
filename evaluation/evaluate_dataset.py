"""One command for direct dataset generation, judging and metrics (no sharding)."""
from __future__ import annotations
import argparse
from pathlib import Path
from tools.download_dataset import download
from .run_dataset import run_dataset
from .scoring.judge_answers import judge_file
from .scoring.aggregate import aggregate_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("data/benchmark_528.json"))
    p.add_argument("--download", action="store_true", help="Download/verify the published file before any calls")
    p.add_argument("--expected-chains", type=int)
    p.add_argument("--models", type=Path, required=True); p.add_argument("--only-model", nargs="+")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--distractor-family", choices=("relevant", "irrelevant", "random"), required=True)
    p.add_argument("--length-policy", choices=("none", "legacy_word_min", "tokenizer_min"), required=True)
    p.add_argument("--audit-policy", choices=("strict", "report"), default="strict")
    p.add_argument("--judge-model", required=True)
    p.add_argument("--judge-policy", choices=("semantic_all",), default="semantic_all")
    p.add_argument("--judge-max-tokens", type=int, default=2048)
    p.add_argument("--match-policy", choices=("legacy", "normalized"), default="legacy")
    p.add_argument("--sample-chains", type=int); p.add_argument("--max-tokens", type=int)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--run-tag", default="primary")
    p.add_argument("--configuration", nargs="+"); p.add_argument("--ccs-single", choices=("A1", "A2", "A3"), default="A2")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.download:
        download(a.dataset, expected=a.expected_chains if a.expected_chains is not None else 528)
    if not a.dry_run:
        from rag_utils.gemini import api_key
        api_key()  # Fail before generation costs if the requested judge has no credentials.
    report = run_dataset(a.dataset, a.models, a.output_dir, family=a.distractor_family,
        length_policy=a.length_policy, audit_policy=a.audit_policy, expected_chains=a.expected_chains,
        sample_chains=a.sample_chains, seed=a.seed, configurations=a.configuration,
        only_models=a.only_model, max_tokens=a.max_tokens, match_policy=a.match_policy,
        run_tag=a.run_tag, dry_run=a.dry_run)
    if a.dry_run: return
    if report["errors"]: raise SystemExit("Generation errors remain; rerun the same command to retry them before judging")
    judge_report = judge_file(a.output_dir / "generations.jsonl", a.output_dir / "judged.jsonl",
        judge_model=a.judge_model, policy=a.judge_policy, judge_max_tokens=a.judge_max_tokens)
    if judge_report["judge_errors"] or judge_report["generation_errors"]:
        raise SystemExit("Judging errors remain; rerun to retry before reporting accuracy")
    aggregate_file(a.output_dir / "judged.jsonl", a.output_dir / "summary", a.ccs_single)

if __name__ == "__main__": main()
