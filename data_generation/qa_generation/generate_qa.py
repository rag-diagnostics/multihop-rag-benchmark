"""Original QA prompt, stable question-type assignment and safe JSON/XLSX handoffs."""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_utils.tables import read_records, ensure_serials
from rag_utils.stages import run_rows
from rag_utils.gemini import generate_text
from data_generation.qa_generation.prompt import build_prompt

MODEL_NAME = "gemini-2.5-pro"
QUESTION_TYPES = ["What", "Where", "Which year", "Which season", "Which town"]
SYSTEM_INSTRUCTION = "You are an expert in creating factual question-answer pairs."


def parse_qa(text):
    q = a = None
    for line in str(text).strip().splitlines():
        if line.startswith("QUESTION:"): q = line[len("QUESTION:"):].strip()
        elif line.startswith("ANSWER:"): a = line[len("ANSWER:"):].strip()
    if not q or not a: raise ValueError("Response needs QUESTION: and ANSWER: lines")
    if not 1 <= len(a.split()) <= 2: raise ValueError("Answer exceeds the preserved prompt's 1-2 word limit")
    return q, a


def generate_qa(summary_text, question_type, *, model=MODEL_NAME, call=generate_text, max_tokens=2048):
    prompt = build_prompt(summary_text, question_type)
    response = call(model, prompt, max_tokens=max_tokens, system=SYSTEM_INSTRUCTION)
    q, a = parse_qa(response)
    # Preserve a literal extractive answer; do not replace a failed answer with another model output.
    if not re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", summary_text, re.I):
        raise ValueError("Generated answer is not a verbatim span of its narrative")
    return q, a


def run(input_path, output_path, model=MODEL_NAME, call=generate_text, max_tokens=2048):
    if Path(input_path).resolve() == Path(output_path).resolve(): raise ValueError("QA output must be separate from input")
    rows = ensure_serials(read_records(input_path))
    for i, r in enumerate(rows):
        if not isinstance(r.get("Summary"), str) or not r["Summary"].strip(): raise ValueError("Every row needs a Summary")
        # Based on original source-row order, not on how many cached rows remain.
        r["question_type"] = QUESTION_TYPES[i % len(QUESTION_TYPES)]
    protocol = {"stage": "qa", "model": model, "max_tokens": max_tokens, "prompt": build_prompt("{text}", "{type}"), "system": SYSTEM_INSTRUCTION}
    def process(row):
        question, answer = generate_qa(row["Summary"], row["question_type"], model=model, call=call, max_tokens=max_tokens)
        return {"question": question, "answer": answer, "Document": row["Summary"], "qa_model": model}
    return run_rows(rows, output_path, protocol, process, ["question", "answer", "Document"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=MODEL_NAME); p.add_argument("--max-tokens", type=int, default=2048)
    a = p.parse_args(); run(a.input, a.output, a.model, max_tokens=a.max_tokens)

if __name__ == "__main__": main()
