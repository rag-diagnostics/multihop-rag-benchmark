#!/usr/bin/env python3
"""Five-reference-model PKS baseline distribution comparison.

Compute no-context PMI scores for the same three calibration QA groups with
several causal LMs. This script compares baseline distributions; instance-level
filtering is performed by 00_para14.py with the primary reference model.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pks_core import (CATEGORIES, DEFAULT_QA_FILENAMES, load_qa_pairs, read_json,
                      write_json, load_model_and_tokenizer, nll_scorer, score_qa,
                      DEFAULT_CALIBRATION_FILENAMES, calibration_stats)


def score_reference_models(models, datasets, model_loader, nll_factory, cached_primary=None):
    """Return per-model scores and baseline statistics using matching models."""
    all_scores = {}
    summary = {}
    for name, model_id in models.items():
        print(f"Scoring calibration categories with {name}", flush=True)
        if cached_primary and name == cached_primary[0]:
            cache_dir = Path(cached_primary[1])
            groups = {}
            for category in CATEGORIES:
                obj = read_json(cache_dir / DEFAULT_CALIBRATION_FILENAMES[category])
                groups[category] = [float(x) for x in obj["pmi_classic"] if x is not None and math.isfinite(float(x))]
                if len(groups[category]) != len(datasets[category]):
                    raise ValueError(f"Cached {category} scores do not match number of calibration QA pairs")
        else:
            tokenizer, model = model_loader(model_id)
            nll = nll_factory(tokenizer, model)
            groups = {
                cat: [float(score_qa(nll, item["question"], item["answer"])) for item in datasets[cat]]
                for cat in CATEGORIES
            }
            del nll, tokenizer, model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        all_scores[name] = groups
        summary[name] = {cat: calibration_stats(groups[cat]) for cat in CATEGORIES}
    return all_scores, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-json", required=True, type=Path,
                   help='JSON object: {"display name": "model directory or HF ID", ...}')
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--primary-cache-dir", type=Path,
                   help="Optional PKS score cache from para11.py for the primary model")
    p.add_argument("--primary-name", default="Qwen/Qwen2.5-7B-Instruct",
                   help="Key in models JSON whose scores may be reused from the primary cache")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = p.parse_args()
    models = read_json(args.models_json)
    if not isinstance(models, dict) or not models or not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in models.items()):
        raise ValueError("models-json must be a non-empty string-to-string JSON object")
    if args.primary_cache_dir and args.primary_name not in models:
        raise ValueError("--primary-name is not present in models-json")
    data = {cat: load_qa_pairs(args.data_dir / DEFAULT_QA_FILENAMES[cat]) for cat in CATEGORIES}
    all_scores, summary = score_reference_models(
        models, data, lambda model_id: load_model_and_tokenizer(model_id, args.device), nll_scorer,
        (args.primary_name, args.primary_cache_dir) if args.primary_cache_dir else None,
    )
    write_json(args.output_dir / "experiment_pmi_scores_detailed_ALL.json", all_scores)
    write_json(args.output_dir / "experiment_model_means_summary_ALL.json", summary)
    print("Baseline comparison complete:", ", ".join(models))


if __name__ == "__main__":
    main()
