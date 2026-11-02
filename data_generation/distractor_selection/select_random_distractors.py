#!/usr/bin/env python3
"""Build random answer-screened distractor pools from an existing QA workbook.

New adaptation for the ECIR resource release: this is NOT the original ranked
Qwen/noise-based distractor selection algorithm. See ORIGINAL_VS_RELEASE.md.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
import unicodedata

import pandas as pd
import requests

DEFAULT_MODEL = "gemini-2.0-flash"  # Original data_check.py judge model.


def words(text: str) -> list[str]:
    """Case-insensitive Unicode tokenization for lexical answer exclusion."""
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", str(text)).casefold(), flags=re.UNICODE)


def gold_variants(golden_answer_field: str) -> list[str]:
    """Preserve the original gold validator's comma-separated variant convention."""
    return [part.strip() for part in str(golden_answer_field).split(",") if part.strip()]


def contains_gold_answer(document: str, golden_answer_field: str) -> bool:
    doc_tokens = words(document)
    if not doc_tokens:
        return False
    doc_str = " " + " ".join(doc_tokens) + " "
    for candidate in gold_variants(golden_answer_field):
        answer_tokens = words(candidate)
        if answer_tokens and (" " + " ".join(answer_tokens) + " ") in doc_str:
            return True
    return False


def safe_question_seed(seed: int, question: str) -> int:
    """The same question obtains the same random sequence independent of row order."""
    digest = hashlib.sha256(question.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def answerability_prompt(document: str, question: str) -> str:
    # This prompt follows the original data_check.py extraction-only check.
    return f'''
You are a strict extraction-only reading comprehension bot.

Rules (must follow exactly):
1. Use ONLY the provided text. Do NOT use outside knowledge.
2. Do NOT reason, infer, guess, or paraphrase.
3. ONLY answer if the answer span is explicitly present verbatim in the text.
4. If the answer is not explicitly present, output exactly: UNKNOWN

Text:
\"\"\"{document}\"\"\"

Question:
{question}

Output (either Answer or UNKNOWN):
'''.strip()


def call_gemini(prompt: str, api_key: str, model: str, retries: int = 3) -> str:
    """Use an API-key header (never put secrets in URLs or exception output)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 30},
    }
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            if response.status_code in {429, 500, 502, 503, 504}:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError("Temporary Gemini API failure; last attempt exhausted")
            if not response.ok:
                raise RuntimeError(f"Gemini API request failed (HTTP {response.status_code})")
            result = response.json()
            return str(result["candidates"][0]["content"]["parts"][0]["text"]).strip()
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError("Gemini response was unavailable or malformed") from None
    raise RuntimeError("Gemini answerability check did not complete")


def check_candidate(question: str, document: str, mode: str, api_key: str | None, model: str) -> tuple[bool, str]:
    if mode == "span-only":
        return True, "not_model_checked"
    try:
        response = call_gemini(answerability_prompt(document, question), api_key or "", model)
    except RuntimeError:
        # Unlike the archived data_check.py, an API error must not count as SAFE.
        return False, "api_or_parse_failure"
    answer = response.strip().strip('"').strip("'")
    if answer.upper() == "UNKNOWN":
        return True, "model_unknown"
    if not answer:
        return False, "empty_model_response"
    # Conservative: ANY non-UNKNOWN answer is rejected, not just exact gold.
    return False, "model_found_possible_answer"


def select_for_row(row: dict, all_documents: list[str], count: int,
                   seed: int, mode: str, api_key: str | None,
                   model: str, check_fn=check_candidate) -> tuple[str, dict | None, dict]:
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    golden = str(row["Document"]).strip()
    stats = {
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
        "eligible": 0, "span_rejected": 0, "model_rejected": 0,
        "api_or_parse_failures": 0, "selected": 0,
    }
    if not (question and answer and golden):
        stats["status"] = "missing_required_field"
        return question, None, stats

    candidates = []
    for doc in all_documents:
        if doc == golden:
            continue
        if contains_gold_answer(doc, answer):
            stats["span_rejected"] += 1
            continue
        candidates.append(doc)
    stats["eligible"] = len(candidates)
    rng = random.Random(safe_question_seed(seed, question))
    rng.shuffle(candidates)

    selected = []
    for doc in candidates:
        safe, reason = check_fn(question, doc, mode, api_key, model)
        if safe:
            selected.append(doc)
            if len(selected) == count:
                break
        else:
            if reason == "api_or_parse_failure":
                stats["api_or_parse_failures"] += 1
            else:
                stats["model_rejected"] += 1
    stats["selected"] = len(selected)
    stats["status"] = "ok" if len(selected) == count else "insufficient_safe_distractors"
    if len(selected) != count:
        return question, None, stats
    return question, {
        "golden_answer": answer,
        "golden_document_text": golden,
        "distractors": selected,
    }, stats


def save_json(obj, location: str) -> None:
    path = Path(location)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-xlsx", required=True, help="QA workbook with question, answer, Document")
    ap.add_argument("--output", default="rag_benchmark_clean_new.json")
    ap.add_argument("--audit", default="random_distractor_audit.json")
    ap.add_argument("--count", type=int, default=10, help="10 for 11-document n=1 contexts")
    ap.add_argument("--seed", type=int, default=42, help="New release seed, not a historical recorded seed")
    ap.add_argument("--check-mode", choices=("llm", "span-only"), default="llm")
    ap.add_argument("--model", default=os.getenv("GEMINI_DISTRACTOR_MODEL", DEFAULT_MODEL))
    ap.add_argument("--workers", type=int, default=4, help="Concurrent questions in LLM-check mode")
    ap.add_argument("--limit", type=int, default=0, help="For smoke tests, e.g. --limit 2")
    args = ap.parse_args()
    if args.count != 10:
        ap.error("This paper's fixed 11-document design needs exactly 10 n=1 distractors")
    if args.workers < 1:
        ap.error("--workers must be >= 1")
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if args.check_mode == "llm" and not api_key:
        ap.error("GEMINI_API_KEY environment variable is required for --check-mode llm")
    df = pd.read_excel(args.input_xlsx, dtype=object)
    needed = {"question", "answer", "Document"}
    if not needed.issubset(df.columns):
        ap.error("Input workbook needs columns question, answer, Document")
    df = df.dropna(subset=["question", "answer", "Document"])
    rows = df.to_dict(orient="records")
    if args.limit: rows = rows[:args.limit]
    q_list = [str(row["question"]).strip() for row in rows]
    if len(set(q_list)) != len(q_list):
        ap.error("Duplicate question texts in workbook; cannot safely use question as JSON key")
    documents = list(dict.fromkeys(str(x).strip() for x in df["Document"] if str(x).strip()))
    if len(documents) < 11:
        ap.error("At least 11 distinct documents required in source corpus")

    def process(row):
        return select_for_row(row, documents, args.count, args.seed, args.check_mode, api_key, args.model)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(process, rows))
    output = {q: item for q, item, _ in outcomes if item is not None}
    audits = [stat for _, _, stat in outcomes]
    save_json(output, args.output)
    save_json({
        "selection": "seeded_random_rejection_sampling",
        "check_mode": args.check_mode,
        "seed": args.seed,
        "model": args.model if args.check_mode == "llm" else None,
        "requested": len(rows), "retained": len(output),
        "rejected": len(rows) - len(output),
        "item_audit": audits,
    }, args.audit)
    print(f"Retained {len(output)} of {len(rows)} questions; wrote {args.output} and {args.audit}")
    if len(output) != len(rows):
        raise SystemExit("Some questions had insufficient safe candidates; inspect audit before continuing")
    if args.check_mode == "span-only":
        print("WARNING: lexical checks only. No LLM answerability screening was performed.")


if __name__ == "__main__":
    main()
