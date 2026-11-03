import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_utils.gemini import generate_text
from rag_utils.io import atomic_json
import os
import json
import asyncio
import requests
import re
from itertools import cycle
from tqdm.asyncio import tqdm

# ---------------- CONFIG ----------------
# Load credentials from environment only. Never place keys in code or logs.
API_KEYS = [s.strip() for s in (os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").split(",") if s.strip()]

MODEL_NAME = "gemini-2.0-flash"
CONCURRENT_LIMIT = 30  # overall in-flight cap (parallel across questions)

INPUT_JSON_FILE = "rag_benchmark_clean_new.json"  # after distractor filtering
OUTPUT_JSON_FILE = "rag_benchmark_gold_validated_new.json"
OUTPUT_ERRORS_JSON_FILE = "rag_benchmark_gold_validation_errors_new.json"
KEY_CYCLE = cycle(API_KEYS)
key_lock = asyncio.Lock()
sem = asyncio.Semaphore(CONCURRENT_LIMIT)


# ---------------- REST CALL ----------------
def call_gemini_rest_api(prompt: str, api_key: str) -> str:
    try:
        return generate_text(MODEL_NAME, prompt, key=api_key, max_tokens=2048, json_output=True)
    except Exception:
        return "__ERROR__:request_failed"


_key_index = 0

async def get_next_key() -> str:
    global _key_index
    keys = API_KEYS or [os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")]
    if not keys or not all(keys): raise RuntimeError("Set GEMINI_API_KEY or GOOGLE_API_KEY")
    key = keys[_key_index % len(keys)]
    _key_index += 1
    return key


# ---------------- TEXT NORMALIZATION ----------------
def norm(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n\"'.,;:!?()[]{}")
    return s


def word_count_1_to_3(span: str) -> bool:
    w = norm(span).split()
    return 1 <= len(w) <= 3


def golden_variants_from_field(golden_answer_field: str) -> list[str]:
    parts = [p.strip() for p in golden_answer_field.split(",")]
    variants = []
    seen = set()
    for p in parts:
        pn = norm(p)
        if pn and pn not in seen:
            seen.add(pn)
            variants.append(p.strip())
    return variants


def is_verbatim_in_doc(span: str, doc_text: str) -> bool:
    if not span:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(norm(span)) + r"(?!\w)", norm(doc_text)))


# ---------------- JSON EXTRACTION (FIX) ----------------
def extract_json_object(text: str) -> dict:
    """
    Extracts a JSON object from a model response that may include:
      - ```json ... ``` fences
      - leading/trailing commentary
    Returns parsed dict or raises ValueError.
    """
    if not text:
        raise ValueError("Empty response")

    t = text.strip()

    # Remove markdown fences if present
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    # Extract outermost JSON object (first '{' to last '}')
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object braces found")

    candidate = t[start:end + 1].strip()
    return json.loads(candidate)


# ---------------- GOLD VALIDATION ----------------
async def validate_gold_with_answer_list(question: str, gold_doc: str, golden_answer_field: str) -> dict:
    gold_variants = golden_variants_from_field(golden_answer_field)
    gold_norm_set = {norm(v) for v in gold_variants}

    prompt = f"""
You are a strict gold-document QA validator.

You will be given:
(1) TEXT (gold document)
(2) QUESTION
(3) ALLOWED_ANSWERS: a list of acceptable gold answers (variants)

Your job:
A) Decide if the QUESTION can be answered from the TEXT by COPYING a 1–3 word span verbatim.
B) If yes, choose the best matching answer from ALLOWED_ANSWERS that appears verbatim in TEXT.
C) Ensure there is NO OTHER distinct 1–3 word span in TEXT that could also answer the question.
D) Provide a short justification based ONLY on TEXT (no outside knowledge, no reasoning/guessing).

STRICT RULES:
- Do NOT infer, guess, paraphrase, or combine sentences.
- Only accept answers that appear verbatim in TEXT.
- The matched answer MUST be exactly 1–3 words.
- The "other_answers" MUST be verbatim 1–3 word spans from TEXT (if any exist).
- If answer is not explicitly present OR ambiguous, verdict must be FAIL.

Output rules:
- Output MUST start with "{{" and end with "}}".
- Do NOT use markdown fences.
- Do NOT include any extra text before or after the JSON.

Return ONLY valid JSON in this exact schema:
{{
  "verdict": "PASS" or "FAIL",
  "matched_answer": "<one of ALLOWED_ANSWERS verbatim from TEXT, or empty string>",
  "other_answers": ["<verbatim 1-3 word span>", ...],
  "justification": "<brief quote-based justification>"
}}

ALLOWED_ANSWERS:
{json.dumps(gold_variants, ensure_ascii=False)}

TEXT:
\"\"\"{gold_doc}\"\"\"

QUESTION:
{question}
""".strip()

    async with sem:
        api_key = await get_next_key()
        resp = await asyncio.to_thread(call_gemini_rest_api, prompt, api_key)

    if resp.startswith("__ERROR__"):
        return {"ok": False, "reason": "api_error", "model_raw": resp}

    # Parse JSON robustly
    try:
        data = extract_json_object(resp)
    except Exception as e:
        return {
            "ok": False,
            "reason": "json_parse_error",
            "parse_error": str(e),
            "model_raw": resp
        }

    verdict = str(data.get("verdict", "")).strip().upper()
    matched = data.get("matched_answer", "")
    other = data.get("other_answers", [])
    justification = data.get("justification", "")

    if verdict not in {"PASS", "FAIL"}:
        return {"ok": False, "reason": "bad_schema_verdict", "model_raw": resp}

    if not isinstance(other, list) or not all(isinstance(x, str) for x in other) or not isinstance(matched, str) or not isinstance(justification, str):
        return {"ok": False, "reason": "invalid_schema", "model_raw": resp}

    # Hard enforcement: matched must be verbatim & 1-3 words
    matched_ok = bool(matched) and is_verbatim_in_doc(matched, gold_doc) and word_count_1_to_3(matched)
    matched_in_gold_list = norm(matched) in gold_norm_set

    # Hard enforcement: other answers must be verbatim & 1-3 words, and distinct from matched
    filtered_other = []
    seen_other = set()
    for x in other:
        if not isinstance(x, str):
            continue
        if not is_verbatim_in_doc(x, gold_doc):
            continue
        if not word_count_1_to_3(x):
            continue
        xn = norm(x)
        if not xn or xn in gold_norm_set:
            continue
        if xn in seen_other:
            continue
        seen_other.add(xn)
        filtered_other.append(x)

    if verdict == "PASS":
        if not matched_ok:
            return {
                "ok": False,
                "reason": "pass_but_matched_not_verbatim_or_not_1to3",
                "matched_answer": matched,
                "other_answers": filtered_other,
                "justification": justification,
                "gold_variants": gold_variants,
                "model_raw": resp
            }
        if not matched_in_gold_list:
            return {
                "ok": False,
                "reason": "pass_but_matched_not_in_golden_list",
                "matched_answer": matched,
                "other_answers": filtered_other,
                "justification": justification,
                "gold_variants": gold_variants,
                "model_raw": resp
            }
        if len(filtered_other) > 0:
            return {
                "ok": False,
                "reason": "gold_doc_ambiguous_other_answers_exist",
                "matched_answer": matched,
                "other_answers": filtered_other,
                "justification": justification,
                "gold_variants": gold_variants,
                "model_raw": resp
            }

        return {
            "ok": True,
            "reason": "pass",
            "matched_answer": matched,
            "justification": justification
        }

    # Model returned FAIL
    return {
        "ok": False,
        "reason": "model_fail",
        "matched_answer": matched,
        "other_answers": filtered_other,
        "justification": justification,
        "gold_variants": gold_variants,
        "model_raw": resp
    }


# ---------------- DRIVER ----------------
async def process_dataset(json_data: dict):
    global sem, key_lock, KEY_CYCLE
    sem, key_lock, KEY_CYCLE = asyncio.Semaphore(CONCURRENT_LIMIT), asyncio.Lock(), cycle(API_KEYS)
    cleaned = {}
    errors = []

    questions = list(json_data.keys())
    total = len(questions)

    print(f"Gold validation for {total} questions")
    print(f"Keys in rotation: {len(API_KEYS)} | Concurrency: {CONCURRENT_LIMIT}")
    print("-" * 70)

    async def worker(q: str):
        item = json_data[q]
        gold_doc = item.get("golden_document_text", "")
        gold_ans_field = item.get("golden_answer", "")

        if not gold_doc or not gold_ans_field:
            return (q, False, {
                "question": q,
                "reason": "missing_gold_doc_or_golden_answer",
                "gold_doc_present": bool(gold_doc),
                "golden_answer_present": bool(gold_ans_field)
            })

        res = await validate_gold_with_answer_list(q, gold_doc, gold_ans_field)

        if res.get("ok"):
            item["gold_validation_matched_answer"] = res.get("matched_answer", "")
            item["gold_validation_justification"] = res.get("justification", "")
            return (q, True, item)

        err = {
            "question": q,
            "reason": res.get("reason"),
            "parse_error": res.get("parse_error", ""),
            "golden_answer_field": gold_ans_field,
            "matched_answer": res.get("matched_answer", ""),
            "other_answers": res.get("other_answers", []),
            "justification": res.get("justification", ""),
            "model_raw": res.get("model_raw", ""),
            "gold_doc_preview": gold_doc[:500],
        }
        return (q, False, err)

    tasks = [worker(q) for q in questions]

    done = 0
    for coro in asyncio.as_completed(tasks):
        q, ok, payload = await coro
        done += 1

        if ok:
            cleaned[q] = payload
            status = "PASS"
        else:
            errors.append(payload)
            status = f"FAIL ({payload.get('reason')})"

        short_q = (q[:70] + "..") if len(q) > 70 else q
        tqdm.write(f"[{done}/{total}] {status} | Q: {short_q}")

    return {q: cleaned[q] for q in questions if q in cleaned}, errors


def main():
    if not os.path.exists(INPUT_JSON_FILE):
        raise FileNotFoundError(INPUT_JSON_FILE)

    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    cleaned, errors = asyncio.run(process_dataset(json_data))

    atomic_json(OUTPUT_JSON_FILE, cleaned)
    atomic_json(OUTPUT_ERRORS_JSON_FILE, errors)
    if any(e.get("reason") in {"api_error", "json_parse_error", "bad_schema_verdict", "invalid_schema"} for e in errors):
        raise RuntimeError("Gold validation had API/schema errors; do not treat the partial file as a finished dataset")

    print("\n" + "=" * 40)
    print("GOLD VALIDATION COMPLETE")
    print("=" * 40)
    print(f"Total input:   {len(json_data)}")
    print(f"Passed kept:   {len(cleaned)}")
    print(f"Failed drop:   {len(errors)}")
    print(f"Saved: {OUTPUT_JSON_FILE}")
    print(f"Errors: {OUTPUT_ERRORS_JSON_FILE}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate answer uniqueness in each gold document")
    parser.add_argument("--input", default=INPUT_JSON_FILE)
    parser.add_argument("--output", default=OUTPUT_JSON_FILE)
    parser.add_argument("--errors", default=OUTPUT_ERRORS_JSON_FILE)
    parser.add_argument("--model", default=os.getenv("GEMINI_GOLD_MODEL", MODEL_NAME))
    args = parser.parse_args()
    INPUT_JSON_FILE, OUTPUT_JSON_FILE, OUTPUT_ERRORS_JSON_FILE = args.input, args.output, args.errors
    MODEL_NAME = args.model
    if not API_KEYS:
        parser.error("Set GEMINI_API_KEY or GEMINI_API_KEYS in your environment")
    main()