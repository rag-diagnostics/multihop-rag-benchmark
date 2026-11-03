"""Convert source messages to narratives with stable IDs and lossless JSON storage."""
from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_utils.tables import read_records, write_records, ensure_serials
from rag_utils.stages import run_rows
from rag_utils.gemini import generate_text
from rag_utils.io import file_sha256
from data_generation.narrative_generation.prompt import PROMPT_TEMPLATE


def collect_messages(directory, pattern="game*_messages.txt", source="unspecified"):
    rows = []
    for p in sorted(Path(directory).glob(pattern)):
        if not p.is_file(): continue
        text = p.read_text(encoding="utf-8")
        if not text.strip(): raise ValueError(f"Empty source transcript: {p.name}")
        rows.append({"game_id": p.stem, "source": source, "text": text,
                     "source_filename": p.name, "source_sha256": file_sha256(p)})
    if not rows: raise ValueError("No matching message files; check --pattern and --messages-dir")
    return ensure_serials(rows)


def run(rows, output, model="gemini-2.5-flash", max_tokens=32768, call=generate_text):
    for row in rows:
        if not isinstance(row.get("text"), str) or not row["text"].strip(): raise ValueError("Every input needs text")
    protocol = {"stage": "narrative", "model": model, "max_tokens": max_tokens,
                "prompt": PROMPT_TEMPLATE, "temperature": 0.0}
    def process(row):
        prompt = PROMPT_TEMPLATE.format(dialogue_text=row["text"], word_count=len(row["text"].split()))
        text = call(model, prompt, max_tokens=max_tokens)
        return {"Summary": text, "narrative_model": model}
    return run_rows(rows, output, protocol, process, ["Summary"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=Path, help="Selected source JSON with Serial, text and optional source labels")
    g.add_argument("--messages-dir", type=Path)
    p.add_argument("--pattern", default="game*_messages.txt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="gemini-2.5-flash", help="Original notebook setting; set explicitly for a different experiment")
    p.add_argument("--max-tokens", type=int, default=32768)
    a = p.parse_args()
    if a.input and a.input.resolve() == a.output.resolve(): p.error("Do not overwrite source records")
    rows = read_records(a.input) if a.input else collect_messages(a.messages_dir, a.pattern)
    run(rows, a.output, a.model, a.max_tokens)

if __name__ == "__main__": main()
