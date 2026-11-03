"""Explicit original-short-answer or all-nonexact judging, with safe resume identity."""
from __future__ import annotations
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from rag_utils.io import JsonlIndex, fingerprint, append_jsonl, atomic_json, output_lock, file_sha256
from .matching import exact_match
from .legacy_prompt import legacy_judge_prompt

PROMPT_VERSION = "uploaded-evaluation1-type-aware-v1"


def judge_prompt(question, golden_answer, generated_answer):
    return legacy_judge_prompt(question, golden_answer, generated_answer)


def google_judge(question, gold, prediction, model_id, *, max_tokens=2048):
    from rag_utils.gemini import generate_text
    raw = generate_text(model_id, judge_prompt(question, gold, prediction),
                        max_tokens=max_tokens, json_output=True, retries=1)
    return json.loads(raw)


def judgment_signature(row, judge_model, policy, judge_max_tokens):
    return fingerprint({"prompt_version": PROMPT_VERSION,
        "prompt_sha256": file_sha256(Path(__file__).with_name("legacy_prompt.py")),
        "judge_code_sha256": file_sha256(__file__), "judge_model": judge_model, "policy": policy,
        "judge_max_tokens": judge_max_tokens, "temperature": 0.0, "top_p": 1.0,
        "request_fingerprint": row.get("request_fingerprint"), "question": row.get("question"),
        "gold": row.get("golden_answer"), "prediction": row.get("generated_answer"),
        "raw": row.get("generated_text"), "status": row.get("status"),
        "mode": row.get("mode"), "match_policy": row.get("match_policy", "normalized")})


def _clean(text):
    s = str(text or "").strip().strip(" \t\r\n\"'`")
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"[.\u3002!\uff01?\uff1f,\uff0c;\uff1b:\uff1a]+$", "", s).strip()


def classify(row, judge=google_judge, judge_model="gemini-2.5-pro", retries=3,
             policy="semantic_all", judge_max_tokens=2048):
    if policy not in {"legacy_short", "semantic_all"}: raise ValueError("Invalid judge policy")
    result = dict(row)
    result.update(judge_model=judge_model, judge_policy=policy, judge_max_tokens=judge_max_tokens,
                  judgment_fingerprint=judgment_signature(row, judge_model, policy, judge_max_tokens),
                  is_correct=None, judge_verdict=None, normalized_ans=None)
    # Safe aggregation identity: same protocol, not per-answer hash.
    result["judging_protocol"] = fingerprint({"model": judge_model, "policy": policy,
        "max_tokens": judge_max_tokens, "version": PROMPT_VERSION,
        "code": file_sha256(__file__), "prompt": file_sha256(Path(__file__).with_name("legacy_prompt.py"))})
    if row.get("status") != "ok": result["judgment_status"] = "generation_error"; return result
    generated = str(row.get("generated_answer") or "")
    gold = str(row.get("golden_answer") or "")
    if not gold.strip(): raise ValueError("Cannot judge without a gold answer")
    wc = len(re.findall(r"\b\w+\b", generated, re.UNICODE))
    if exact_match(generated, gold, policy=row.get("match_policy", "normalized")):
        result.update(is_correct=True, judge_verdict="CORRECT", normalized_ans=_clean(generated), judgment_status="exact_match")
        return result
    if policy == "legacy_short" and wc <= 3:
        result.update(is_correct=False, judge_verdict="INCORRECT" if generated.strip() else "NO_ANSWER",
                      normalized_ans=_clean(generated) or "HALLUCINATED", judgment_status="legacy_short")
        return result
    prediction = str(row.get("generated_text") or generated) if row.get("mode", "main") == "main" else generated
    for attempt in range(retries):
        try:
            obj = (google_judge(str(row.get("question", "")), gold, prediction, judge_model, max_tokens=judge_max_tokens)
                   if judge is google_judge else judge(str(row.get("question", "")), gold, prediction, judge_model))
            if not isinstance(obj, dict): raise ValueError("Judge must return an object")
            verdict = obj.get("verdict"); correct = obj.get("is_correct")
            norm = obj.get("normalized_ans", obj.get("normalised_ans"))
            if verdict not in {"CORRECT", "INCORRECT", "NO_ANSWER"} or type(correct) is not bool:
                raise ValueError("Invalid judge types/verdict")
            if (verdict == "CORRECT") != correct: raise ValueError("Inconsistent correctness and verdict")
            if not isinstance(norm, str) or not norm.strip(): raise ValueError("Missing normalized answer")
            if verdict == "NO_ANSWER": norm = "HALLUCINATED"
            elif norm.strip().upper() in {"HALLUCINATED", "CONFABULATED", "PENDING_LLM"}:
                raise ValueError("Sentinel answer conflicts with verdict")
            elif not 1 <= len(re.findall(r"\b\w+\b", norm, re.UNICODE)) <= 3:
                raise ValueError("Judge must extract a 1-3 word answer; it is not silently truncated")
            result.update(is_correct=correct, judge_verdict=verdict, normalized_ans=norm.strip(), judgment_status="judge")
            return result
        except Exception as exc:
            result["judge_error_type"] = type(exc).__name__
            if attempt + 1 < retries: time.sleep(min(2 ** attempt, 12))
    result["judgment_status"] = "judge_error"
    return result


def judge_file(input_path, output_path, *, judge=google_judge, judge_model="gemini-2.5-pro",
               workers=1, retries=3, dry_run=False, policy="semantic_all", judge_max_tokens=2048):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve(): raise ValueError("Judge output must be separate from raw generation input")
    if workers < 1 or retries < 1 or judge_max_tokens < 1: raise ValueError("Positive worker, retry and token settings required")
    inputs = JsonlIndex(input_path)
    if not inputs.entries: raise ValueError("No inference records found")
    plan = {"inference_records": len(inputs.entries), "pending_judgments": 0, "already_judged": 0, "judge_errors": 0, "generation_errors": 0}
    def needs(row, existing):
        prior = existing.entries.get(row["request_id"], {})
        return not (prior.get("judgment_fingerprint") == judgment_signature(row, judge_model, policy, judge_max_tokens)
                    and prior.get("judgment_status") in {"exact_match", "judge", "legacy_short"})
    existing = JsonlIndex(output_path)
    # No entire context corpus in memory: only offsets are indexed.
    for row in inputs.rows():
        if needs(row, existing): plan["pending_judgments"] += 1
        else: plan["already_judged"] += 1
    if dry_run: print(json.dumps(plan, indent=2)); return plan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_lock(output_path):
        with output_path.open("a", encoding="utf-8", newline="\n") as f, ThreadPoolExecutor(max_workers=workers) as executor:
            batch = []
            def flush():
                for future in batch:
                    result = future.result()
                    plan["judge_errors"] += result["judgment_status"] == "judge_error"
                    plan["generation_errors"] += result["judgment_status"] == "generation_error"
                    append_jsonl(f, result)
                batch.clear()
            for row in inputs.rows():
                if not needs(row, existing): continue
                batch.append(executor.submit(classify, row, judge, judge_model, retries, policy, judge_max_tokens))
                if len(batch) >= workers: flush()
            if batch: flush()
        atomic_json(str(output_path) + ".summary.json", {**plan, "judge_model": judge_model, "judge_policy": policy})
    print(json.dumps(plan, indent=2)); return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--judge-model", required=True, help="Use the exact judge model for your experiment")
    p.add_argument("--judge-policy", choices=("legacy_short", "semantic_all"), required=True)
    p.add_argument("--judge-max-tokens", type=int, default=2048)
    p.add_argument("--workers", type=int, default=1); p.add_argument("--retries", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    plan = judge_file(a.input, a.output, judge_model=a.judge_model, workers=a.workers, retries=a.retries,
                      dry_run=a.dry_run, policy=a.judge_policy, judge_max_tokens=a.judge_max_tokens)
    if plan["judge_errors"] or plan["generation_errors"]: raise SystemExit(2)

if __name__ == "__main__": main()
