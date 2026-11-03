#!/usr/bin/env python3
"""Qwen PKS classification and question-level benchmark filtering.

Uses the original no-context prompt, summed-token NLL, absolute Z-distance,
original_question grouping, and removal of the common/known class.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pks_core import (read_json, write_json, load_calibration, load_model_and_tokenizer,
                      nll_scorer, score_qa, classify_pks, normalized_questions)


def filter_items(rows, stats, scorer, on_error="error"):
    """Classify each unique base question once and apply its decision to all rows."""
    if on_error not in ("keep", "remove", "error"):
        raise ValueError("on_error must be keep, remove or error")
    rows = normalized_questions(rows)
    decisions = {}
    for item in rows:
        q, a = item["original_question"].strip(), item["answer"].strip()
        if q in decisions:
            continue
        try:
            score = float(scorer(q, a))
            if not math.isfinite(score):
                raise ValueError("Non-finite PKS")
            category, z_scores = classify_pks(score, stats)
            failure = None
        except Exception as exc:
            if on_error == "error":
                raise
            # Explicit keep reproduces the original fail-open policy;
            # the release default raises instead of assigning a class.
            category, score, z_scores = ("nonsense" if on_error == "keep" else "common"), None, None
            failure = type(exc).__name__
        decisions[q] = {
            "question": q, "answer": a, "pmi_classic": score,
            "category": category, "action": "REMOVE" if category == "common" else "KEEP",
            "z_distances": z_scores, "error": failure,
        }
    kept = [item for item in rows if decisions[item["original_question"].strip()]["action"] == "KEEP"]
    removed = [item for item in rows if decisions[item["original_question"].strip()]["action"] == "REMOVE"]
    return kept, removed, decisions


def run(input_path: Path, model_name: str, calibration_dir: Path, output_dir: Path, device: str, on_error: str, allow_legacy_calibration: bool = False):
    summary_path = calibration_dir / "calibration_summary.json"
    if not summary_path.is_file() and not allow_legacy_calibration:
        raise ValueError("Missing calibration_summary.json; calibrate first or explicitly allow legacy calibration files")
    if summary_path.is_file():
        calibration_model = read_json(summary_path).get("model")
        if calibration_model != model_name:
            raise ValueError("Calibration and filtering reference models differ; regenerate calibration with the same model identifier")
    stats = load_calibration(calibration_dir)
    rows = read_json(input_path)
    tokenizer, model = load_model_and_tokenizer(model_name, device)
    nll = nll_scorer(tokenizer, model)
    kept, removed, decisions = filter_items(rows, stats, lambda q, a: score_qa(nll, q, a), on_error=on_error)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "000_abl_filtered_unknowns.json", kept)
    write_json(output_dir / "000_abl_removed_knowns.json", removed)
    with (output_dir / "000_abl_question_categories.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "category"])
        writer.writeheader()
        for record in decisions.values():
            writer.writerow({"question": record["question"], "category": record["category"]})
    write_json(output_dir / "pks_question_decisions.json", list(decisions.values()))
    counts = {cat: sum(rec["category"] == cat for rec in decisions.values()) for cat in ("common", "plausible", "nonsense")}
    summary = {"model": model_name, "input_rows": len(rows), "kept_rows": len(kept),
               "removed_rows": len(removed), "unique_questions": len(decisions),
               "categories": counts, "failed_scores": sum(rec["error"] is not None for rec in decisions.values()),
               "on_error": on_error}
    write_json(output_dir / "pks_filter_summary.json", summary)
    print(summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path, help="Multi-hop JSON array with original_question and answer")
    p.add_argument("--model", required=True, help="Qwen model directory or model ID")
    p.add_argument("--calibration-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--on-error", choices=["keep", "remove", "error"], default="error",
                   help="Default error stops on scoring failures; keep/remove reproduce explicit alternative policies")
    p.add_argument("--allow-legacy-calibration", action="store_true", help="Accept old calibration files without a model/checksum manifest; provenance then remains unverified")
    args = p.parse_args()
    run(args.input, args.model, args.calibration_dir, args.output_dir, args.device, args.on_error, args.allow_legacy_calibration)


if __name__ == "__main__":
    main()
