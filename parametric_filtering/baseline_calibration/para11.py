#!/usr/bin/env python3
"""Compute the three no-context Qwen PKS calibration distributions.

Narrow PKS-only version of the 19-metric calibration experiment in para11.py.
Outputs retain the original pmi_classic JSON key and filenames.
"""
from __future__ import annotations
import argparse
import json
import tempfile
import shutil
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pks_core import (CATEGORIES, DEFAULT_QA_FILENAMES, DEFAULT_CALIBRATION_FILENAMES,
                      load_qa_pairs, load_model_and_tokenizer, nll_scorer, score_qa, write_json,
                      calibration_stats)


def run(model_name: str, data_dir: Path, output_dir: Path, device: str = "auto") -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Calibration output is not empty; use a new directory to avoid mixing score distributions")
    pairs = {cat: load_qa_pairs(data_dir / DEFAULT_QA_FILENAMES[cat]) for cat in CATEGORIES}
    source_hashes = {cat: hashlib.sha256((data_dir / DEFAULT_QA_FILENAMES[cat]).read_bytes()).hexdigest() for cat in CATEGORIES}
    tokenizer, model = load_model_and_tokenizer(model_name, device)
    nll = nll_scorer(tokenizer, model)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pks-calibration-", dir=output_dir.parent))
    output = {}
    try:
        for cat in CATEGORIES:
            scores = []
            for idx, item in enumerate(pairs[cat], start=1):
                scores.append(float(score_qa(nll, item["question"], item["answer"])))
                print(f"{cat}: {idx}/{len(pairs[cat])}", flush=True)
            info = calibration_stats(scores)
            filename = DEFAULT_CALIBRATION_FILENAMES[cat]
            write_json(staging / filename, {"pmi_classic": scores})
            output[cat] = {**info, "output_file": filename}
        hashes = {cat: hashlib.sha256((staging / DEFAULT_CALIBRATION_FILENAMES[cat]).read_bytes()).hexdigest() for cat in CATEGORIES}
        write_json(staging / "calibration_summary.json", {"model": model_name, "categories": output,
                   "qa_sha256": source_hashes, "score_sha256": hashes, "status": "complete"})
        if output_dir.exists(): output_dir.rmdir()  # Only the already-checked empty directory.
        staging.replace(output_dir)
        return output
    finally:
        if staging.exists(): shutil.rmtree(staging)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Qwen model directory or Hugging Face model ID")
    p.add_argument("--data-dir", required=True, type=Path, help="Folder containing the three calibration QA JSON files")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args()
    run(args.model, args.data_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
