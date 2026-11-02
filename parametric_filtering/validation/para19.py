#!/usr/bin/env python3
"""Independent Qwen PKS re-scoring and decision verification.

Re-evaluates each distinct original_question from the benchmark; produces the
REPRO_qwen_* files used for filtering consistency and count verification.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pks_core import (read_json, write_json, load_calibration,
                      load_model_and_tokenizer, nll_scorer, score_qa,
                      classify_pks, normalized_questions)


def independently_filter(rows, stats, scorer, *, fail_on_error: bool = True):
    rows = normalized_questions(rows)
    scores = {}
    for item in rows:
        q, a = item["original_question"].strip(), item["answer"].strip()
        if q in scores:
            continue
        try:
            pmi = float(scorer(q, a))
            if not math.isfinite(pmi):
                raise ValueError("Non-finite PKS")
            winner, z = classify_pks(pmi, stats)
            failure = None
        except Exception as e:
            if fail_on_error:
                raise RuntimeError(f"Could not score {q[:100]}") from e
            pmi, winner, z = None, "nonsense", None
            failure = f"{type(e).__name__}: {e}"
        scores[q] = {"question": q, "answer": a, "pmi_classic": pmi,
                     "nearest_class": winner, "z_distances": z,
                     "keep": winner != "common", "error": failure}
    kept = [i for i in rows if scores[i["original_question"].strip()]["keep"]]
    removed = [i for i in rows if not scores[i["original_question"].strip()]["keep"]]
    return kept, removed, scores


def read_category_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return {r["question"].strip(): r["category"].strip() for r in csv.DictReader(f)}


def verify_categories(recomputed, csv_path: Path):
    expected = read_category_csv(csv_path)
    if set(expected) != set(recomputed):
        missing = set(recomputed) - set(expected)
        extra = set(expected) - set(recomputed)
        raise ValueError(f"Classification questions differ: missing={len(missing)}, extra={len(extra)}")
    mismatches = [q for q, result in recomputed.items() if expected[q] != result["nearest_class"]]
    if mismatches:
        raise ValueError(f"{len(mismatches)} classification disagreements; first: {mismatches[0][:120]}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--model", required=True)
    p.add_argument("--calibration-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--compare-categories", type=Path,
                   help="Optional category CSV from 00_para14.py to check for disagreement")
    p.add_argument("--expected-total-questions", type=int, default=None)
    p.add_argument("--expected-removed-questions", type=int, default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--allow-failed-scores", action="store_true", help="Match original fail-safe KEEP on scoring failure")
    args = p.parse_args()
    stats = load_calibration(args.calibration_dir)
    rows = read_json(args.input)
    tokenizer, model = load_model_and_tokenizer(args.model, args.device)
    nll = nll_scorer(tokenizer, model)
    kept, removed, records = independently_filter(
        rows, stats, lambda q, a: score_qa(nll, q, a), fail_on_error=not args.allow_failed_scores)
    if args.compare_categories:
        verify_categories(records, args.compare_categories)
    n_removed = sum(not record["keep"] for record in records.values())
    summary = {"unique_questions": len(records), "removed_questions": n_removed,
               "retained_questions": len(records) - n_removed,
               "kept_rows": len(kept), "removed_rows": len(removed),
               "model": args.model,
               "failed_scores": sum(record["error"] is not None for record in records.values())}
    if args.expected_total_questions is not None and len(records) != args.expected_total_questions:
        raise AssertionError(f"Expected {args.expected_total_questions} questions; found {len(records)}")
    if args.expected_removed_questions is not None and n_removed != args.expected_removed_questions:
        raise AssertionError(f"Expected {args.expected_removed_questions} removed questions; found {n_removed}")
    output_dir = args.output_dir
    write_json(output_dir / "REPRO_qwen_kept.json", kept)
    write_json(output_dir / "REPRO_qwen_removed.json", removed)
    write_json(output_dir / "REPRO_qwen_scores_and_decisions.json", list(records.values()))
    write_json(output_dir / "REPRO_qwen_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
