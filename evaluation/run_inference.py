#!/usr/bin/env python3
"""Run models on the fixed 11-document contexts prepared in data_generation/."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

try:
    from .contexts import load_contexts, select_tasks, task_key
    from .prompting import main_prompt, free_answer_prompt, citation_prompt, parse_citation, extract_final_answer
    from .providers import generate, validate_models
    from .scoring.matching import exact_match, command_following
except ImportError:
    from contexts import load_contexts, select_tasks, task_key
    from prompting import main_prompt, free_answer_prompt, citation_prompt, parse_citation, extract_final_answer
    from providers import generate, validate_models
    from scoring.matching import exact_match, command_following


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected object at {path}:{line_no}")
        records.append(row)
    return records


def latest_results(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    latest = {}
    for row in rows:
        latest[row["request_id"]] = row
    return latest


def make_job(row: Dict[str, Any], model_key: str, cfg: Dict[str, Any], *,
             run_tag: str, mode: str, max_tokens: int, free_answer: bool) -> Dict[str, Any]:
    if mode == "main":
        prompt = free_answer_prompt(row["question"], row["documents"]) if free_answer else main_prompt(row["question"], row["documents"])
    elif mode == "citation":
        prompt = citation_prompt(row["question"], row["documents"])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    identity = f"{model_key}|{row.get('distractor_family', 'unspecified')}|{task_key(row)}|{run_tag}|{mode}"
    sig = hashlib.sha256(json.dumps({"prompt": prompt, "model": cfg, "max_tokens": max_tokens},
                                    sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"request_id": identity, "request_fingerprint": sig, "model_key": model_key,
            "model_id": cfg["model_id"], "provider": cfg["provider"],
            "run_tag": run_tag, "mode": mode, "max_tokens": max_tokens,
            "free_answer": free_answer, "prompt": prompt, "distractor_family": row.get("distractor_family", "unspecified"),
            "length_matching": row.get("length_matching"), "decomposition": row.get("decomposition", []),
            "source_entry_index": row["source_entry_index"],
            "hop": row["hop"], "configuration": row["configuration"],
            "question": row["question"], "original_question": row.get("original_question", ""),
            "golden_answer": row["answer"], "documents": row["documents"],
            "gold_positions_1_based": row["gold_positions_1_based"], "cfg": cfg}


def do_job(job: Dict[str, Any], *, gen: Callable = generate, max_retries: int = 3) -> Dict[str, Any]:
    """Do not record provider exceptions or token values: they may contain credentials."""
    response = None
    error_type = None
    for trial in range(max_retries):
        try:
            response = gen(job["cfg"], job["prompt"], job["max_tokens"])
            break
        except Exception as exc:
            error_type = type(exc).__name__
            if trial + 1 < max_retries:
                time.sleep(min(2 ** trial, 20))
    result = {key: value for key, value in job.items() if key not in ("cfg", "prompt")}
    result["status"] = "ok" if response is not None else "error"
    if response is None:
        result["error_type"] = error_type
        result["generated_answer"] = ""
        result["generated_text"] = ""
        result["is_exact_match"] = False
        result["command_following"] = None
        return result
    raw = str(response.get("generated_text", "") or "")
    if job["mode"] == "citation":
        prediction, citation = parse_citation(raw)
        result["cited_document_1_based"] = citation
        result["is_citation_correct"] = citation in job["gold_positions_1_based"]
    else:
        prediction = extract_final_answer(raw, free_answer=True) if job["free_answer"] else raw.strip()
    result["generated_text"] = raw
    result["generated_answer"] = prediction
    result["is_exact_match"] = exact_match(prediction, job["golden_answer"], policy=job.get("match_policy", "normalized"))
    result["word_count_policy"] = "unicode_word_regex" if job.get("match_policy") == "legacy" else "whitespace"
    follows = 1 <= len(re.findall(r"\b\w+\b", raw, re.UNICODE)) <= 3 if job.get("match_policy") == "legacy" else command_following(raw)
    result["command_following"] = follows if job["mode"] == "main" and not job["free_answer"] else None
    result.update({k: response.get(k) for k in ("prompt_tokens", "completion_tokens", "finish_reason", "time_seconds")})
    return result


def run(input_path: Path, models_path: Path, output_path: Path, *,
        only_models: List[str] | None = None, configurations: List[str] | None = None,
        mode: str = "main", run_tag: str = "primary", max_tokens: int = 50,
        free_answer: bool = False, workers: int = 4, max_retries: int = 3,
        sample_chains: int | None = None, seed: int = 42, dry_run: bool = False,
        gen: Callable = generate) -> Dict[str, int]:
    if max_tokens < 1 or workers < 1 or max_retries < 1:
        raise ValueError("max_tokens, workers, and max_retries must be positive")
    if mode == "citation" and free_answer:
        raise ValueError("Citation mode cannot use the unconstrained answer prompt")
    config = json.loads(models_path.read_text(encoding="utf-8"))
    all_models = validate_models(config)
    models = {k: v for k, v in all_models.items() if only_models is None or k in only_models}
    if not models or (only_models is not None and set(only_models) != set(models)):
        raise ValueError("Requested model key is not in the model configuration")
    if workers > 1 and any(m["provider"] == "transformers" for m in models.values()):
        raise ValueError("For local transformers models, use --workers 1")
    tasks = select_tasks(load_contexts(input_path), configurations, sample_chains, seed)
    if not tasks:
        raise ValueError("Selection produced no tasks")
    jobs = [make_job(row, name, cfg, run_tag=run_tag, mode=mode,
                     max_tokens=max_tokens, free_answer=free_answer)
            for name, cfg in models.items() for row in tasks]
    existing = latest_results(load_jsonl(output_path)) if not dry_run else {}
    for job in jobs:
        saved = existing.get(job["request_id"])
        if saved and saved.get("request_fingerprint") != job["request_fingerprint"]:
            raise ValueError(f"Refusing to resume changed task: {job['request_id']}")
    jobs = [j for j in jobs if existing.get(j["request_id"], {}).get("status") != "ok"]
    overview = {"selected_contexts": len(tasks), "models": len(models), "pending_calls": len(jobs), "errors": 0, "successful_calls": 0}
    print(json.dumps({"plan": overview, "mode": mode, "run_tag": run_tag}, indent=2))
    if dry_run:
        return overview
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(do_job, job, gen=gen, max_retries=max_retries): job for job in jobs}
            for fut in as_completed(futures):
                row = fut.result()
                if row["status"] == "ok":
                    overview["successful_calls"] += 1
                else:
                    overview["errors"] += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
    print(json.dumps({"finished": overview}, indent=2))
    return overview


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="data_generation/context_formation JSON")
    ap.add_argument("--models", required=True, type=Path, help="Model config JSON")
    ap.add_argument("--output", required=True, type=Path, help="Append-only JSONL output")
    ap.add_argument("--only-model", nargs="+", help="One or more keys from --models")
    ap.add_argument("--configuration", nargs="+", help="Subset: A1/A2/A3/B1/B2/B3")
    ap.add_argument("--mode", choices=("main", "citation"), default="main")
    ap.add_argument("--run-tag", default="primary", help="Unique experiment ID, e.g. short15")
    ap.add_argument("--max-tokens", type=int, default=50)
    ap.add_argument("--free-answer", action="store_true", help="Use separate generation-budget ablation prompt")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--sample-chains", type=int)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="Validate and count tasks without API calls")
    args = ap.parse_args()
    run(args.input, args.models, args.output, only_models=args.only_model,
        configurations=args.configuration, mode=args.mode, run_tag=args.run_tag,
        max_tokens=args.max_tokens, free_answer=args.free_answer, workers=args.workers,
        max_retries=args.max_retries, sample_chains=args.sample_chains,
        seed=args.seed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
