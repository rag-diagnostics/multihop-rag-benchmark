"""Separate incomplete/malformed chains; never change question or document contents."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_utils.io import read_json, atomic_json
from evaluation.dataset import check_chain


def filter_entries(entries, family="random", strict=True):
    if not isinstance(entries, list): raise ValueError("Expected a JSON array")
    kept = []; rejected = []
    for i, row in enumerate(entries):
        try:
            issues = check_chain(row, family)
            if strict and issues: raise ValueError("Chain quality checks: " + ", ".join(x["issue"] for x in issues))
        except (ValueError, TypeError) as e:
            rejected.append({"source_entry_index": i, "original_question": row.get("original_question") if isinstance(row, dict) else None, "reason": str(e)})
            continue
        kept.append(row)
    return kept, rejected


def filter_json_remove_entries_with_hops_error(input_path, output_path, family="random", strict=True):
    if Path(input_path).resolve() == Path(output_path).resolve(): raise ValueError("Preserve unfiltered source; use a separate output")
    rows = read_json(input_path)
    kept, rejected = filter_entries(rows, family, strict)
    atomic_json(output_path, kept)
    atomic_json(str(output_path) + ".rejected.json", rejected)
    print({"input": len(rows), "kept": len(kept), "rejected": len(rejected)})
    return kept, rejected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--distractor-family", choices=("random", "relevant", "irrelevant"), default="random")
    p.add_argument("--schema-only", action="store_true", help="Retain semantically flagged records, recording only structural checks")
    a = p.parse_args()
    kept, rejected = filter_json_remove_entries_with_hops_error(a.input, a.output, a.distractor_family, not a.schema_only)
    if not kept: raise SystemExit(2)

if __name__ == "__main__": main()
