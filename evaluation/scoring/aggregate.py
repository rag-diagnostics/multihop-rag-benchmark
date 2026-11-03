"""Coverage-aware diagnostic metrics; never silently count provider errors as answers."""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from rag_utils.io import JsonlIndex, read_json, atomic_json
from .matching import token_phrase_in_text


def classify_error(row):
    if row.get("is_correct") is not False or row.get("judge_verdict") == "NO_ANSWER": return None
    pred = str(row.get("normalized_ans") or row.get("generated_answer") or "")
    if not pred or pred.upper() in {"HALLUCINATED", "CONFABULATED"}: return None
    docs = row.get("documents") or []; gold = set(row.get("gold_positions_1_based") or [])
    return "DISTRACTED" if any(token_phrase_in_text(pred, d) for i, d in enumerate(docs, 1) if i not in gold) else "FABRICATED"


def summarize_group(items):
    n = len(items); scored = [r for r in items if type(r.get("is_correct")) is bool and r.get("status") == "ok"]
    complete = n > 0 and len(scored) == n
    primary = all(r.get("mode", "main") == "main" for r in items)
    citation = all(r.get("mode") == "citation" for r in items)
    errors = [r.get("error_origin", classify_error(r)) for r in scored]
    return {"dataset_quality_status": "requires_review" if any(r.get("dataset_quality_status") == "requires_review" for r in items) else "see_dataset_audit", "n_attempted": n, "n_scored": len(scored),
            "n_generation_errors": sum(r.get("status") != "ok" for r in items),
            "n_judge_errors": sum(r.get("judgment_status") == "judge_error" for r in items),
            "judgment_complete": complete,
            "accuracy": sum(r["is_correct"] for r in scored) / n if complete else None,
            "observed_scored_accuracy": sum(r["is_correct"] for r in scored) / len(scored) if scored else None,
            "cmd": sum(r.get("command_following") is True for r in items) / n if n and primary and all(r.get("status") == "ok" and type(r.get("command_following")) is bool for r in items) else None,
            "confab": sum(r.get("judge_verdict") == "NO_ANSWER" for r in scored) / n if complete else None,
            "ear": sum(r.get("is_citation_correct") is True for r in items) / n if n and citation and all(r.get("status") == "ok" for r in items) else None,
            "n_distracted": errors.count("DISTRACTED"), "n_fabricated": errors.count("FABRICATED"),
            "n_unclassified_incorrect": sum(r.get("is_correct") is False for r in scored) - sum(x is not None for x in errors)}


def protocol(row):
    return (row["model_key"], row.get("run_tag", "primary"), row.get("distractor_family", "unspecified"),
            row.get("run_identity_sha256", "legacy"), row.get("judging_protocol", "unspecified"))


def aggregate(rows, *, single_condition="A2"):
    if single_condition not in {"A1", "A2", "A3"}: raise ValueError("Invalid CCS single-hop reference")
    groups = defaultdict(list); chains = defaultdict(dict); seen = set()
    for row in rows:
        p = protocol(row); mode = row.get("mode", "main"); hop = row["hop"]; config = row["configuration"]
        unique = (p, mode, row["source_entry_index"], hop, config)
        if unique in seen: raise ValueError("Duplicate result for one chain/condition; use the latest JSONL records")
        seen.add(unique)
        compact = {k: v for k, v in row.items() if k not in {"documents", "decomposition", "length_matching"}}
        compact["error_origin"] = row["error_origin"] if "error_origin" in row else classify_error(row)
        groups[(p, mode, hop, config)].append(compact)
        if mode == "main": chains[(p, row["source_entry_index"])][(hop, config)] = compact
    by_condition = []
    def meta(p):
        return dict(zip(("model", "run_tag", "distractor_family", "run_identity", "judging_protocol"), p))
    for (p, mode, hop, cfg), group in sorted(groups.items()):
        by_condition.append({**meta(p), "mode": mode, "hop": hop, "configuration": cfg, **summarize_group(group)})
    ccs = []
    for p in sorted({p for p, ix in chains}):
        for cfg in ("B1", "B2", "B3"):
            expected = [("n1", single_condition)] + [(f"n{i}", cfg) for i in range(2, 5)]
            valid = []; incomplete = 0
            for (rp, ix), lookup in chains.items():
                if rp != p: continue
                one = [lookup.get(k) for k in expected]
                if not all(one) or any(r.get("status") != "ok" or type(r.get("is_correct")) is not bool for r in one):
                    incomplete += 1; continue
                valid.append([r["is_correct"] for r in one])
            record = {**meta(p), "single_condition": single_condition, "multi_condition": cfg,
                      "n_complete_chains": len(valid), "n_incomplete_chains": incomplete}
            for d in range(1, 5): record[f"ccs_n{d}"] = sum(all(v[:d]) for v in valid) / len(valid) if valid and not incomplete else None
            ccs.append(record)
    return {"by_condition": by_condition, "chain_completion": ccs,
            "error_diagnostic": "normalized_token_phrase_in_distractors; lexical association, not causal attribution",
            "ccs_definition": f"Cumulative correctness, {single_condition} baseline followed by the same B condition"}


def save_csv(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def aggregate_file(input_path, output_dir, single_condition="A2", manifest_path=None):
    src = Path(input_path); dest = Path(output_dir)
    index = JsonlIndex(src)
    if not index.entries: raise ValueError("No judged records")
    # Stream long records; retain scalar metrics, not duplicated document text.
    def compact_rows():
        for row in index.rows():
            row["error_origin"] = classify_error(row)
            row.pop("documents", None)
            yield row
    # Preserve error-origin value already computed before documents were removed.
    report = aggregate(compact_rows(), single_condition=single_condition)
    manifest_path = Path(manifest_path) if manifest_path else src.parent / "run_manifest.json"
    if manifest_path.is_file():
        ident = read_json(manifest_path)["identity"]
        wanted = set()
        for model in ident["models"]:
            for ix in ident["source_indices"]:
                for depth in range(1, 5):
                    for conf in ("A1", "A2", "A3") if depth == 1 else ("B1", "B2", "B3"):
                        if conf in ident["configurations"]:
                            wanted.add(f"{model}|{ident['distractor_family']}|{ix}|n{depth}|{conf}|{ident['run_tag']}|{ident['mode']}")
        missing = wanted - set(index.entries); extra = set(index.entries) - wanted
        report["coverage"] = {"expected": len(wanted), "present": len(index.entries), "missing": len(missing), "unexpected": len(extra)}
        if extra: raise ValueError("Judged file contains records outside the run manifest")
        if missing:
            for g in report["by_condition"]:
                g.update(accuracy=None, confab=None, ear=None, cmd=None, judgment_complete=False)
            for g in report["chain_completion"]:
                for d in range(1, 5): g[f"ccs_n{d}"] = None
    else:
        report["coverage"] = {"expected": None, "present": len(index.entries), "note": "No run manifest; completeness of scheduled requests is not established"}
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest / "evaluation_metrics.json", report)
    save_csv(dest / "by_condition.csv", report["by_condition"])
    save_csv(dest / "chain_completion.csv", report["chain_completion"])
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ccs-single", choices=("A1", "A2", "A3"), default="A2")
    p.add_argument("--manifest", type=Path)
    a = p.parse_args(); report = aggregate_file(a.input, a.output_dir, a.ccs_single, a.manifest)
    print(f"Wrote {a.output_dir}/evaluation_metrics.json")
    if any(not g["judgment_complete"] for g in report["by_condition"]): raise SystemExit(2)

if __name__ == "__main__": main()
