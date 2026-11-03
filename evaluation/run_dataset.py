"""Evaluate the released JSON directly. No intermediate context file or sharding required."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from rag_utils.io import (iter_json_array, read_json, fingerprint, atomic_json, append_jsonl,
                          JsonlIndex, output_lock, require_new_manifest, file_sha256)
from .dataset import audit_dataset, tasks_for_chain, FAMILIES
from .providers import validate_models, generate, preflight, release_local_models
from .run_inference import make_job, do_job

VERSION = "direct-json-v1"


def code_signature():
    here = Path(__file__).resolve().parent
    paths = [here / x for x in ("run_dataset.py", "dataset.py", "run_inference.py", "prompting.py", "providers.py", "scoring/matching.py")]
    paths += [here.parent / "data_generation/context_formation" / x for x in ("construct_contexts.py", "original_singlehop.py", "original_multihop.py")]
    return fingerprint({p.name: file_sha256(p) for p in paths})


def run_dataset(dataset, models_path, output_dir, *, family, length_policy, audit_policy="strict",
                expected_chains=None, sample_chains=None, seed=42, configurations=None,
                only_models=None, max_tokens=None, match_policy="legacy", mode="main",
                run_tag="primary", dry_run=False, retries=3, gen=generate):
    dataset = Path(dataset); output_dir = Path(output_dir)
    if output_dir == dataset or output_dir == Path(models_path): raise ValueError("Output must be a separate directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfgs = validate_models(read_json(models_path))
    if only_models:
        if not set(only_models).issubset(cfgs): raise ValueError("Unknown model key")
        cfgs = {k: v for k, v in cfgs.items() if k in only_models}
    if audit_policy not in {"strict", "report"}: raise ValueError("audit_policy must be strict or report")
    if match_policy not in {"legacy", "normalized"}: raise ValueError("Invalid match policy")
    if retries < 1 or max_tokens is not None and max_tokens < 1: raise ValueError("Positive retry and token counts required")
    confs = set(configurations or ["A1", "A2", "A3", "B1", "B2", "B3"])
    if not confs or not confs.issubset({"A1", "A2", "A3", "B1", "B2", "B3"}): raise ValueError("Invalid configurations")
    audit = audit_dataset(dataset, family, expected_chains)
    # Every full input record is parsed and counted before generation begins.
    atomic_json(output_dir / "dataset_audit.json", audit)
    if not audit["schema_valid"]: raise ValueError("Invalid dataset structure/count; see dataset_audit.json. No model calls made.")
    if audit_policy == "strict" and audit["chains_needing_review"]:
        raise ValueError("Dataset has chain/span warnings. Review dataset_audit.json. Use --audit-policy report only to evaluate the unchanged release with those limitations recorded.")
    indices = list(range(audit["chains"]))
    if sample_chains is not None:
        if not 1 <= sample_chains <= len(indices): raise ValueError("sample_chains is outside the dataset size")
        indices = sorted(random.Random(seed).sample(indices, sample_chains))
    selected = set(indices)
    effective = {k: (max_tokens if max_tokens is not None else v.get("max_tokens", 50)) for k, v in cfgs.items()}
    identity = {"version": VERSION, "code_sha256": code_signature(), "dataset_sha256": audit["sha256"],
                "models": cfgs, "source_indices": indices, "distractor_family": family, "length_policy": length_policy,
                "seed": seed, "configurations": sorted(confs), "max_tokens": effective, "match_policy": match_policy,
                "mode": mode, "run_tag": run_tag, "audit_policy": audit_policy}
    plan = {"chains": len(indices), "calls_planned": 0, "calls_completed": 0, "errors": 0, "resumed": 0,
            "schema_valid": True, "chains_needing_review_in_full_dataset": audit["chains_needing_review"], "dry_run": dry_run}
    # Validate transformed contexts, including every model-specific length policy, before paid calls.
    tokenizers = {}
    if length_policy == "tokenizer_min":
        from transformers import AutoTokenizer
        for key, cfg in cfgs.items():
            tid = cfg.get("tokenizer_id") or (cfg["model_id"] if cfg["provider"] == "transformers" else None)
            if not tid: raise ValueError("tokenizer_min requires tokenizer_id for API models")
            tokenizers[key] = AutoTokenizer.from_pretrained(tid, trust_remote_code=False)
    for key in cfgs:
        for i, chain in enumerate(iter_json_array(dataset)):
            if i not in selected: continue
            tasks = tasks_for_chain(chain, i, family, seed, length_policy, tokenizers.get(key))
            plan["calls_planned"] += sum(t["configuration"] in confs for t in tasks)
    if dry_run:
        atomic_json(output_dir / "dry_run.json", {"plan": plan, "identity": identity})
        print(json.dumps(plan, indent=2)); return plan
    if gen is generate:
        for cfg in cfgs.values(): preflight(cfg)
    result_file = output_dir / "generations.jsonl"
    with output_lock(result_file):
        manifest = require_new_manifest(output_dir / "run_manifest.json", identity)
        existing = JsonlIndex(result_file)
        with result_file.open("a", encoding="utf-8", newline="\n") as f:
            for key, cfg in cfgs.items():
                try:
                    for i, chain in enumerate(iter_json_array(dataset)):
                        if i not in selected: continue
                        for row in tasks_for_chain(chain, i, family, seed, length_policy, tokenizers.get(key)):
                            if row["configuration"] not in confs: continue
                            job = make_job(row, key, cfg, run_tag=run_tag, mode=mode, max_tokens=effective[key], free_answer=False)
                            job["match_policy"] = match_policy
                            job["dataset_sha256"] = audit["sha256"]
                            job["run_identity_sha256"] = manifest["identity_sha256"]
                            job["dataset_quality_status"] = "requires_review" if audit["chains_needing_review"] else "lexical_checks_passed"
                            job["request_fingerprint"] = fingerprint({"generation": job["request_fingerprint"], "identity": manifest["identity_sha256"], "gold": row["answer"]})
                            prior = existing.entries.get(job["request_id"], {})
                            if prior and prior.get("request_fingerprint") != job["request_fingerprint"]: raise ValueError("Changed generation fingerprint; use a new output directory")
                            if prior.get("status") == "ok": plan["resumed"] += 1; continue
                            result = do_job(job, gen=gen, max_retries=retries)
                            append_jsonl(f, result)
                            plan["calls_completed"] += result["status"] == "ok"
                            plan["errors"] += result["status"] != "ok"
                finally:
                    if cfg["provider"] == "transformers": release_local_models()
        atomic_json(output_dir / "generation_summary.json", plan)
    print(json.dumps(plan, indent=2)); return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--distractor-family", choices=FAMILIES, required=True)
    p.add_argument("--length-policy", choices=("none", "legacy_word_min", "tokenizer_min"), required=True)
    p.add_argument("--audit-policy", choices=("strict", "report"), default="strict")
    p.add_argument("--expected-chains", type=int)
    p.add_argument("--sample-chains", type=int)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--configuration", nargs="+")
    p.add_argument("--only-model", nargs="+")
    p.add_argument("--max-tokens", type=int, help="Overrides config; otherwise per-model setting, then original 50-token default")
    p.add_argument("--match-policy", choices=("legacy", "normalized"), default="legacy")
    p.add_argument("--mode", choices=("main", "citation"), default="main")
    p.add_argument("--run-tag", default="primary")
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    try:
        summary = run_dataset(a.dataset, a.models, a.output_dir, family=a.distractor_family,
            length_policy=a.length_policy, audit_policy=a.audit_policy, expected_chains=a.expected_chains,
            sample_chains=a.sample_chains, seed=a.seed, configurations=a.configuration,
            only_models=a.only_model, max_tokens=a.max_tokens, match_policy=a.match_policy,
            mode=a.mode, run_tag=a.run_tag, dry_run=a.dry_run, retries=a.retries)
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        p.exit(2, f"Run stopped: {e}\n")
    if summary["errors"]: raise SystemExit(2)

if __name__ == "__main__": main()
