#!/usr/bin/env python3
"""Summarize tiered accuracy, CMD, CONFAB, EAR, CCS, and error origin."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    from ..run_inference import load_jsonl, latest_results
    from .matching import token_phrase_in_text
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from run_inference import load_jsonl, latest_results
    from scoring.matching import token_phrase_in_text


def classify_error(row: Dict[str, Any]) -> str | None:
    """Paper's lexical diagnostic: wrong answer in a distractor vs absent from distractors."""
    if row.get("is_correct") is not False or row.get("judge_verdict") == "NO_ANSWER":
        return None
    pred = str(row.get("normalized_ans") or row.get("generated_answer") or "")
    if not pred or pred.upper() == "HALLUCINATED":
        return None
    docs = row.get("documents") or []
    gold = set(row.get("gold_positions_1_based") or [])
    distractors = [d for i, d in enumerate(docs, 1) if i not in gold]
    return "DISTRACTED" if any(token_phrase_in_text(pred, d) for d in distractors) else "FABRICATED"


def summarize_group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(items)
    # Exclude generation failures from answered coverage, but keep them in the denominator.
    scored = [r for r in items if type(r.get("is_correct")) is bool]
    complete = len(scored) == n
    primary = all(r.get("mode", "main") == "main" for r in items)
    cited = all(r.get("mode") == "citation" for r in items)
    confab_count = sum(r.get("judge_verdict") == "NO_ANSWER" for r in scored)
    errors = [classify_error(r) for r in scored]
    row = {"n_attempted": n, "n_scored": len(scored),
           "n_generation_errors": sum(r.get("status") != "ok" for r in items),
           "n_judge_errors": sum(r.get("judgment_status") == "judge_error" for r in items),
           "judgment_complete": complete,
           "accuracy": sum(r["is_correct"] for r in scored) / n if complete else None,
           "cmd": sum(r.get("command_following") is True for r in items) / n if primary and not any(r.get("free_answer") for r in items) else None,
           "confab": confab_count / n if complete else None,
           "ear": sum(r.get("is_citation_correct") is True for r in items) / n if cited and all(r.get("status") == "ok" for r in items) else None,
           "n_distracted": errors.count("DISTRACTED"),
           "n_fabricated": errors.count("FABRICATED"),
           "n_unclassified_incorrect": sum(r.get("is_correct") is False for r in scored) - sum(x is not None for x in errors)}
    return row


def aggregate(rows: List[Dict[str, Any]], *, single_condition: str = "A2") -> Dict[str, Any]:
    if single_condition not in {"A1", "A2", "A3"}:
        raise ValueError("single_condition must be A1, A2 or A3")
    by_group = defaultdict(list)
    for row in rows:
        key = (row["model_key"], row.get("run_tag", "primary"), row.get("mode", "main"), row["hop"], row["configuration"])
        by_group[key].append(row)
    by_condition = []
    for (model, tag, mode, hop, cfg), group_rows in sorted(by_group.items()):
        by_condition.append({"model": model, "run_tag": tag, "mode": mode,
                             "hop": hop, "configuration": cfg,
                             **summarize_group(group_rows)})
    # A2->B1/B2/B3 continuity: a single-hop central placement is the baseline,
    # and each B configuration is tracked through n2, n3, n4 for the same chain.
    chain_map = defaultdict(dict)
    for r in rows:
        if r.get("mode", "main") != "main":
            continue
        key = (r["model_key"], r.get("run_tag", "primary"), r["source_entry_index"])
        chain_map[key][(r["hop"], r["configuration"])] = r
    ccs = []
    for model, tag in sorted({(m, t) for m, t, ix in chain_map}):
        for multi_cfg in ("B1", "B2", "B3"):
            expected = [("n1", single_condition)] + [(f"n{i}", multi_cfg) for i in range(2, 5)]
            valid = []
            incomplete = 0
            for (m, t, chain_id), lookup in chain_map.items():
                if m != model or t != tag:
                    continue
                one = [lookup.get(e) for e in expected]
                if not all(one) or any(type(r.get("is_correct")) is not bool for r in one):
                    incomplete += 1
                    continue
                valid.append([r["is_correct"] for r in one])
            part = {"model": model, "run_tag": tag, "single_condition": single_condition,
                    "multi_condition": multi_cfg, "n_complete_chains": len(valid),
                    "n_incomplete_chains": incomplete}
            for depth in range(1, 5):
                part[f"ccs_n{depth}"] = (sum(all(q[:depth]) for q in valid) / len(valid)) if valid and incomplete == 0 else None
            ccs.append(part)
    return {"by_condition": by_condition, "chain_completion": ccs}


def save_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Judged inference JSONL")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ccs-single", choices=("A1", "A2", "A3"), default="A2")
    args = p.parse_args()
    rows = list(latest_results(load_jsonl(args.input)).values())
    if not rows:
        raise ValueError("No scored records")
    report = aggregate(rows, single_condition=args.ccs_single)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    save_csv(args.output_dir / "by_condition.csv", report["by_condition"])
    save_csv(args.output_dir / "chain_completion.csv", report["chain_completion"])
    print(f"Wrote {args.output_dir}/evaluation_metrics.json, by_condition.csv, chain_completion.csv")


if __name__ == "__main__":
    main()
