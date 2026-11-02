#!/usr/bin/env python3
"""Two-stage EM and factual-semantic judgement of generated answers."""
from __future__ import annotations
import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List

try:
    from ..run_inference import load_jsonl, latest_results
    from .matching import exact_match, command_following
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_inference import load_jsonl, latest_results
    from scoring.matching import exact_match, command_following

_TLS = threading.local()


def judge_prompt(question: str, golden_answer: str, generated_answer: str) -> str:
    """Factual equivalence only; judge receives no evidence or generator identity."""
    return f"""You are evaluating a factual question-answering model output.

Tasks:
1. Extract the intended final answer from the model output, if any, in 1 to 3 words.
2. Decide whether the extracted answer is factually equivalent to ANY accepted gold answer variant (comma-separated).
3. Identify refusal, no meaningful answer, or off-task/confabulated output.

Be sensitive to the answer type implied by the question (person, country, year, season, etc.).
Do not mark an answer correct just because a gold term appears inside unrelated reasoning.
Ignore differences of capitalisation, harmless punctuation, and unambiguous synonymous variants.
Do not rely on world knowledge to override the gold reference. Be conservative.

Return ONLY a JSON object with exactly these fields:
{{"normalized_ans": "1-3 words or HALLUCINATED", "is_correct": true, "verdict": "CORRECT"}}
The verdict must be CORRECT, INCORRECT, or NO_ANSWER.
If the answer is missing, refusal, or unusable, use normalized_ans=HALLUCINATED, is_correct=false, verdict=NO_ANSWER.
If an answer is given but does not match a gold variant, use is_correct=false, verdict=INCORRECT.

Question:
{question}

Gold answer variants:
{golden_answer}

Model output:
{generated_answer}
"""


def google_judge(question: str, gold: str, prediction: str, model_id: str) -> Dict[str, Any]:
    from google import genai
    from google.genai import types
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY")
    if not hasattr(_TLS, "client"):
        _TLS.client = genai.Client(api_key=api_key)
    response = _TLS.client.models.generate_content(
        model=model_id,
        contents=judge_prompt(question, gold, prediction),
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                          temperature=0.0, top_p=1.0,
                                          max_output_tokens=240),
    )
    return json.loads(response.text or "{}")


def classify(row: Dict[str, Any], judge: Callable = google_judge,
             judge_model: str = "gemini-2.5-pro", retries: int = 3) -> Dict[str, Any]:
    result = {k: v for k, v in row.items()}
    result["judge_model"] = judge_model
    result["is_correct"] = None
    result["judge_verdict"] = None
    result["normalized_ans"] = None
    if row.get("status") != "ok":
        result["judgment_status"] = "generation_error"
        return result
    generated = str(row.get("generated_answer") or "")
    gold = str(row.get("golden_answer") or "")
    if exact_match(generated, gold):
        result["is_correct"] = True
        result["judge_verdict"] = "CORRECT"
        result["normalized_ans"] = generated
        result["judgment_status"] = "exact_match"
        return result
    # Every non-exact response is judged, including short synonyms.
    for attempt in range(retries):
        try:
            judge_input = (str(row.get("generated_text") or generated)
                           if row.get("mode", "main") == "main" else generated)
            judged = judge(str(row.get("question") or ""), gold,
                           judge_input, judge_model)
            verdict = str(judged.get("verdict", "")).upper()
            normalized = str(judged.get("normalized_ans", "")).strip()
            is_correct = judged.get("is_correct")
            if verdict not in ("CORRECT", "INCORRECT", "NO_ANSWER") or type(is_correct) is not bool:
                raise ValueError("Judge returned invalid JSON fields")
            if verdict == "CORRECT" and not is_correct or verdict != "CORRECT" and is_correct:
                raise ValueError("Inconsistent verdict and correctness")
            if not normalized:
                raise ValueError("Judge produced empty normalized_ans")
            if verdict == "NO_ANSWER":
                normalized = "HALLUCINATED"
            result.update(is_correct=is_correct, judge_verdict=verdict,
                          normalized_ans=normalized, judgment_status="judge")
            return result
        except Exception as exc:
            result["judge_error_type"] = type(exc).__name__
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 12))
    # Fail closed: no guessed correctness when the judge fails.
    result["judgment_status"] = "judge_error"
    return result


def judge_file(input_path: Path, output_path: Path, *, judge: Callable = google_judge,
               judge_model: str = "gemini-2.5-pro", workers: int = 4,
               retries: int = 3, dry_run: bool = False) -> Dict[str, int]:
    if workers < 1 or retries < 1:
        raise ValueError("workers and retries must be positive")
    inputs = latest_results(load_jsonl(input_path))
    if not inputs:
        raise ValueError("No inference records found")
    existing = latest_results(load_jsonl(output_path)) if not dry_run else {}
    for key, row in inputs.items():
        if key in existing and existing[key].get("request_fingerprint") != row.get("request_fingerprint"):
            raise ValueError(f"Refusing to reuse a judgment for changed input: {key}")
    pending = [r for key, r in inputs.items()
               if existing.get(key, {}).get("judgment_status") not in ("exact_match", "judge")]
    plan = {"inference_records": len(inputs), "pending_judgments": len(pending),
            "already_judged": len(inputs)-len(pending), "judge_errors": 0}
    print(json.dumps({"plan": plan, "judge_model": judge_model}, indent=2))
    if dry_run:
        return plan
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(classify, row, judge=judge, judge_model=judge_model,
                                 retries=retries) for row in pending]
            for fut in as_completed(futures):
                result = fut.result()
                if result["judgment_status"] == "judge_error":
                    plan["judge_errors"] += 1
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
    print(json.dumps({"finished": plan}, indent=2))
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, help="Inference JSONL")
    ap.add_argument("--output", type=Path, required=True, help="Judged JSONL")
    ap.add_argument("--judge-model", default="gemini-2.5-pro")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    judge_file(args.input, args.output, judge_model=args.judge_model,
               workers=args.workers, retries=args.retries, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
