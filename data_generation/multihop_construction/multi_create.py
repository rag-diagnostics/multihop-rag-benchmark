import os
import json
import re
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_utils.gemini import Model
from rag_utils.io import atomic_json, read_json
from rag_utils.tables import read_records
import logging
import random
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
from tqdm.asyncio import tqdm
from dotenv import load_dotenv


# -----------------------------------------------------
# 1) Configuration & Setup
# -----------------------------------------------------
load_dotenv()

# --- API KEYS ---
# Credentials are injected through environment variables, never source code.
API_KEYS = [s.strip() for s in (os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").split(",") if s.strip()]

# --- FILES ---
INPUT_JSON = "rag_benchmark_gold_validated_new.json"          # Input dictionary: {question: item_data}
GLOBAL_DOCS_FILE = "updated_modified_final_dataset_6_02.xlsx"         # Excel with document texts
OUTPUT_FILE = "000_new_multihop_dataset_final.json"
ERROR_FILE = "000_new_multihop_errors.json"

# --- TUNING ---
SAVE_INTERVAL = 5                      # Save progress every 5 items
WORKERS_PER_KEY = 5                    # 3 workers * 2 keys = 6 concurrent requests
MODEL_NAME = "gemini-2.5-flash"        # Preserved model identifier; verify access before running
HOP_MAX = 4                            # Generate chains up to n=4

# --- LOGIC SETTINGS ---
ENTITY_CANDIDATES_K = 6                # How many entities to extract as potential bridges
DOC_TRIES_PER_HOP = 8                  # How many global docs to scan for a match
GEN_TRIES_PER_DOC = 2                  # How many times to retry generation if it fails

# --- CONTEXT WINDOW ---
# Preserve full documents. Provider errors are rejected, not hidden by clipping.
DOC_SNIPPET_CHARS = None              
VERIFY_DOC_SNIPPET_CHARS = None

# --- LOGGING ---
# Using utf-8 encoding for file handler to prevent emoji crashes on Windows
logging.getLogger(__name__).addHandler(logging.NullHandler())
SAFETY_SETTINGS = {}

# -----------------------------------------------------
# 2) Utilities
# -----------------------------------------------------
def save_json(data: Any, filename: str) -> None:
    atomic_json(filename, data)

def strip_json_fence(text: str) -> str:
    return re.sub(r"```json|```", "", text or "").strip()

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def contains_entity(doc: str, ent: str) -> bool:
    return bool(ent and re.search(r"(?<!\w)" + re.escape(ent.strip()) + r"(?!\w)", doc, re.I))

def safe_json_loads(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None

# -----------------------------------------------------
# 3) Data Loading
# -----------------------------------------------------
def load_global_docs(filepath: str, column: str = "Document") -> List[str]:
    # Explicit Document priority is independent of workbook column order.
    from rag_utils.tables import read_records
    rows = read_records(filepath)
    if not rows or any(not isinstance(r.get(column), str) or not r[column].strip() for r in rows):
        raise ValueError("Global documents require a nonempty " + column + " column; never substitute raw text")
    return list(dict.fromkeys(r[column].strip() for r in rows))

# -----------------------------------------------------
# 4) Gemini Wrapper (Robust Retry Logic)
# -----------------------------------------------------
async def safe_generate(model, prompt: str, max_retries: int = 3) -> Optional[str]:
    """Wraps generation with exponential backoff for 429 Rate Limit errors."""
    for attempt in range(max_retries):
        try:
            resp = await asyncio.to_thread(
                model.generate_content,
                prompt,
                safety_settings=SAFETY_SETTINGS,
            )
            if not getattr(resp, "parts", None):
                return None # Content blocked by safety filter
            return (resp.text or "").strip()

        except Exception as e:
            msg = str(e)
            if "429" in msg or "Resource has been exhausted" in msg:
                # Jittered backoff: 2s, 4s, 8s...
                wait = (2 ** attempt) + random.uniform(0, 1)
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    logging.warning(" Max retries hit (Rate Limit).")
                    return None
            else:
                logging.error("API request failed; error details suppressed")
                return None
    return None

# -----------------------------------------------------
# 5) Core Logic: Entity Extraction & Selection
# -----------------------------------------------------
async def extract_entities_topk(question: str, model, k: int = ENTITY_CANDIDATES_K) -> List[str]:
    """Extracts potential 'Bridge Entities' (Subjects) from the question."""
    prompt = f"""
    Extract up to {k} distinct candidate bridge entities from the question.
    Entities should be Proper Nouns or unique terms (Country, City, Person, Treaty, Organization).
    Prefer entities that are the grammatical SUBJECT or OBJECT of the sentence.

    Return ONLY JSON list of strings.
    Question: "{question}"
    """
    out = await safe_generate(model, prompt)
    
    # API-based extraction
    if out:
        data = safe_json_loads(strip_json_fence(out))
        if isinstance(data, list):
            # Clean up extraction
            return [x.strip() for x in data if isinstance(x, str) and len(x.strip()) > 2 and contains_entity(question, x.strip())][:k]

    return []  # Failed extraction is not treated as a valid generated hop.

def select_linking_docs(
    entities: List[str],
    global_docs: List[str],
    used_docs: set,
    max_pairs: int = DOC_TRIES_PER_HOP
) -> List[Tuple[str, str]]:
    """
    Scans the GLOBAL pool for documents that mention the extracted entities.
    Returns list of (document_text, matched_entity).
    """
    pairs = []
    if not entities: return pairs

    for ent in entities:
        count = 0
        for doc in global_docs:
            if doc in used_docs: continue
            
            if contains_entity(doc, ent):
                pairs.append((doc, ent))
                count += 1
                if len(pairs) >= max_pairs: return pairs
                if count >= 2: break # Don't get too many docs for the same entity
    return pairs

# -----------------------------------------------------
# 6) Core Logic: "Russian Doll" Generation
# -----------------------------------------------------
async def generate_hop_question(
    current_question: str,
    new_doc_text: str,
    answer: str,
    bridge_entity: str,
    hop_level: int,
    model
) -> Optional[Dict[str, str]]:
    """
    Uses One-Shot Prompting and JSON output to create a clean dependency chain.
    """
    doc_snip = new_doc_text[:DOC_SNIPPET_CHARS]
    
    prompt = f"""
You are an expert linguistics puzzle creator.
Your goal is to rewrite a question by replacing a specific name (Bridge Entity) with a descriptive phrase based on a new document.

**Input Data:**
- Current Question: "{current_question}"
- Bridge Entity to Replace: "{bridge_entity}"
- New Document: \"\"\"{doc_snip}\"\"\"

**Instructions:**
1. Read the New Document and find facts about "{bridge_entity}".
2. Create a **natural, specific description** of "{bridge_entity}".
   - BAD: "the entity that..." / "the thing which..."
   - GOOD: "the nation that..." / "the commander who..." / "the ally of France who..."
3. Rewrite the Current Question by swapping "{bridge_entity}" with your new description.
4. Ensure the grammar flows smoothly.

**Output JSON Format (Strict):**
{{
  "description": "the specific phrase you created",
  "reasoning_step": "Identify {bridge_entity} as [description]",
  "new_question": "The full new question text..."
}}
"""
    out = await safe_generate(model, prompt)
    if not out: return None
    
    try:
        data = safe_json_loads(strip_json_fence(out))
        if isinstance(data, dict) and all(isinstance(data.get(k), str) and data[k].strip() for k in ("new_question", "description", "reasoning_step")):
            return data
    except:
        return None
    return None

# -----------------------------------------------------
# 7) Verification: Ablation Testing
# -----------------------------------------------------
async def can_answer_with_docs(question: str, docs: List[str], answer: str, model) -> Tuple[bool, Optional[str]]:
    """Ask model if answer is determinable from the given docs."""
    doc_block = "\n\n".join([f"DOC{i+1}: {d[:VERIFY_DOC_SNIPPET_CHARS]}" for i, d in enumerate(docs)])
    prompt = f"""
    Determine if the exact answer to the question can be found using ONLY these documents.
    
    Question: "{question}"
    Documents:
    {doc_block}

    If yes, return JSON: {{"can_answer": true, "answer": "<exact answer found>"}}
    If no, return JSON: {{"can_answer": false}}
    """
    out = await safe_generate(model, prompt)
    if not out: raise RuntimeError("Answerability request failed")

    data = safe_json_loads(strip_json_fence(out))
    if not isinstance(data, dict) or type(data.get("can_answer")) is not bool:
        raise ValueError("Invalid answerability response")
    if data["can_answer"] and (not isinstance(data.get("answer"), str) or not data["answer"].strip()):
        raise ValueError("Answerability=true requires an answer")
    return (data["can_answer"], data.get("answer"))

async def verify_russian_doll_hop(prev_question: str, new_question: str, prev_docs: List[str],
                                 new_doc: str, bridge_entity: str, answer: str, model) -> Tuple[bool, str]:
    """Structural replacement, gold-answer agreement, and leave-one-document-out checks."""
    import re
    import unicodedata
    def norm(text):
        s = unicodedata.normalize("NFKC", str(text or "")).casefold()
        return " ".join(re.sub(r"[^\w\s]", " ", s).split())
    def present(entity, text):
        return bool(entity and re.search(r"(?<!\w)" + re.escape(entity.strip()) + r"(?!\w)", text, re.I))
    if not isinstance(new_question, str) or not new_question.strip(): return False, "empty"
    if not present(bridge_entity, prev_question): return False, "bridge_absent_from_previous_question"
    if present(bridge_entity, new_question): return False, "entity_not_removed"
    variants = [x.strip() for x in str(answer).split(",") if x.strip()]
    if not variants: return False, "missing_gold_answer"
    if any(present(a, new_question) for a in variants): return False, "answer_leaked"
    if new_doc in prev_docs or not new_doc: return False, "invalid_new_document"
    all_docs = prev_docs + [new_doc]
    try:
        can_all, ans_all = await can_answer_with_docs(new_question, all_docs, answer, model)
        if not can_all: return False, "not_answerable_with_all_docs"
        if norm(ans_all) not in {norm(a) for a in variants}: return False, "returned_answer_disagrees_with_gold"
        can_prev, _ = await can_answer_with_docs(new_question, prev_docs, answer, model)
        if can_prev: return False, "answerable_without_new_doc"
        # Also reject shortcuts that do not need one of the earlier gold documents.
        for i in range(len(prev_docs)):
            reduced = all_docs[:i] + all_docs[i+1:]
            can_reduced, _ = await can_answer_with_docs(new_question, reduced, answer, model)
            if can_reduced: return False, "answerable_without_previous_gold_" + str(i)
    except Exception:
        return False, "answerability_check_failed"
    return True, "ok"

# -----------------------------------------------------
# 8) Chain Processing (The main logic loop)
# -----------------------------------------------------
async def process_chain(item_key: str, item_data: Dict[str, Any], global_docs: List[str], model) -> Dict[str, Any]:
    original_q = item_key
    answer = item_data.get("golden_answer")
    base_doc = item_data.get("golden_document_text")
    if not all(isinstance(v, str) and v.strip() for v in (original_q, answer, base_doc)):
        raise ValueError("Missing base question, answer or gold document")

    # Randomly sampled, answer-screened distractors (no ranked/noise-based pools).
    distractor_pool = item_data.get("distractors", [])[:]
    if not isinstance(distractor_pool, list) or len(distractor_pool) != 10:
        raise ValueError("Expected exactly 10 answer-screened random distractors for n1")

    if len(set(distractor_pool + [base_doc])) != 11:
        raise ValueError("Base context must contain 11 distinct documents")
    used_docs = set([base_doc])
    current_q = original_q
    golden_docs = [base_doc]
    
    # Initialize decomposition chain
    decomposition = []

    result = {
        "source_id": item_data.get("Serial", item_data.get("source_id")),
        "original_question": original_q,
        "answer": answer,
        "decomposition": decomposition,
        "hops": {
            "n1": {
                "question": original_q,
                "golden_docs": [base_doc],
                "context_distractors": distractor_pool[:]
            }
        }
    }

    # Loop for n=2, n=3, n=4
    for n in range(2, HOP_MAX + 1):
        # A) Extract Candidates
        entities = await extract_entities_topk(current_q, model, k=ENTITY_CANDIDATES_K)

        # B) Find Linking Docs in Global Pool
        pairs = select_linking_docs(entities, global_docs, used_docs, max_pairs=DOC_TRIES_PER_HOP)
        if not pairs:
            result["hops"][f"n{n}"] = {"error": f"No linking global doc found for entities={entities}"}
            break

        hop_success = False
        last_reason = "unknown"

        # C) Try Generating Valid Question
        for (new_doc, bridge_ent) in pairs:
            for _ in range(GEN_TRIES_PER_DOC):
                # Generate
                gen_data = await generate_hop_question(current_q, new_doc, str(answer), bridge_ent, n, model)
                if not gen_data:
                    last_reason = "generation_failed"
                    continue

                new_q = normalize_space(gen_data["new_question"])
                desc = gen_data["description"]
                step = gen_data["reasoning_step"]
                if normalize_space(desc).casefold() not in new_q.casefold():
                    last_reason = "description_not_used_in_question"
                    continue

                # Verify
                ok, reason = await verify_russian_doll_hop(
                    current_q, new_q, golden_docs[:], new_doc, bridge_ent, str(answer), model
                )

                if ok:
                    # Success! Update pools and state
                    used_docs.add(new_doc)
                    golden_docs.append(new_doc)
                    current_q = new_q
                    
                    # Keep exactly 11 documents per hop. Each added gold
                    # document replaces one randomly chosen base distractor.
                    if new_doc in distractor_pool:
                        distractor_pool.remove(new_doc)
                    elif distractor_pool:
                        distractor_pool.pop(0)

                    # Update Decomposition
                    decomposition.append({
                        "hop_level": n,
                        "entity_to_resolve": bridge_ent,
                        "clue_description": desc,
                        "reasoning": step
                    })

                    result["hops"][f"n{n}"] = {
                        "question": new_q,
                        "bridge_entity_replaced": bridge_ent,
                        "description_used": desc,
                        "added_golden_doc": new_doc,
                        "all_golden_docs": golden_docs[:],
                        "context_distractors_remaining": distractor_pool[:]
                    }
                    hop_success = True
                    break
                else:
                    last_reason = reason
            
            if hop_success: break
        
        if not hop_success:
            result["hops"][f"n{n}"] = {"error": f"Failed verification", "reason": last_reason}
            break

        # Rate Limit Breather
        await asyncio.sleep(0.5)

    return result

# -----------------------------------------------------
# 9) Orchestrator & Workers
# -----------------------------------------------------
async def worker(worker_id: str, api_key: str, queue: asyncio.Queue, global_docs: List[str],
                 results: List[Dict[str, Any]], errors: List[Dict[str, Any]],
                 save_lock: asyncio.Lock, pbar):
    """Persistent worker to process queue items."""
    model = Model(MODEL_NAME, key=api_key)

    logging.info(f"[Worker] {worker_id} ready.")

    while True:
        try:
            item = await queue.get()
        except asyncio.CancelledError:
            break

        q_text, q_data = item
        try:
            chain = await process_chain(q_text, q_data, global_docs, model)
            results.append(chain)
        except Exception as e:
            logging.error(f"[Error] {worker_id}: {type(e).__name__}")
            errors.append({"question": q_text, "error_type": type(e).__name__})
        finally:
            pbar.update(1)
            queue.task_done()
            
            # Periodic Save
            if len(results) % SAVE_INTERVAL == 0:
                async with save_lock:
                    save_json(results, OUTPUT_FILE)
                    save_json(errors, ERROR_FILE)

async def main():
    # 1. Load Data
    if not API_KEYS or WORKERS_PER_KEY < 1: raise ValueError("At least one API key and worker are required")
    global_docs = load_global_docs(GLOBAL_DOCS_FILE)
    if not global_docs: raise ValueError("Empty document corpus")

    try:
        with open(INPUT_JSON, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("Input base-QA JSON not found") from None

    # 2. Setup Queue
    queue = asyncio.Queue()
    for q, d in raw_data.items():
        queue.put_nowait((q, d))

    results = []
    errors = []
    save_lock = asyncio.Lock()
    
    print(f" Starting Multi-Hop Generator")
    print(f" Items: {len(raw_data)} | Workers: {WORKERS_PER_KEY * len(API_KEYS)} | Model: {MODEL_NAME}")
    
    pbar = tqdm(total=len(raw_data), unit="chains")
    workers = []

    # 3. Launch Workers
    for key_index, api_key in enumerate(API_KEYS, start=1):
        for i in range(WORKERS_PER_KEY):
            workers.append(asyncio.create_task(worker(f"Key{key_index}-W{i}", api_key, queue, global_docs, results, errors, save_lock, pbar)))

    # 4. Wait & Cleanup
    await queue.join()
    for w in workers: w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    pbar.close()

    # Stable source order, independent of worker completion order.
    order = {q: i for i, q in enumerate(raw_data)}
    results.sort(key=lambda r: order[r["original_question"]])
    # Final Save
    save_json(results, OUTPUT_FILE)
    save_json(errors, ERROR_FILE)
    if errors: raise RuntimeError("Some construction requests failed; inspect the error output")
    print(f"\n Processing Complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate 2-4 hop QA chains from random-screened base contexts")
    parser.add_argument("--input", default=INPUT_JSON)
    parser.add_argument("--global-docs", default=GLOBAL_DOCS_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--errors", default=ERROR_FILE)
    parser.add_argument("--model", default=os.getenv("GEMINI_MULTIHOP_MODEL", MODEL_NAME))
    parser.add_argument("--workers-per-key", type=int, default=WORKERS_PER_KEY)
    args = parser.parse_args()
    INPUT_JSON, GLOBAL_DOCS_FILE = args.input, args.global_docs
    OUTPUT_FILE, ERROR_FILE = args.output, args.errors
    MODEL_NAME, WORKERS_PER_KEY = args.model, args.workers_per_key
    if not API_KEYS:
        parser.error("Set GEMINI_API_KEY or GEMINI_API_KEYS in your environment")
    asyncio.run(main())
