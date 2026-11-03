"""Audit released chains and construct tasks directly, without rewriting the dataset."""
from __future__ import annotations
import argparse
import copy
import json
import random
import re
from pathlib import Path
from rag_utils.io import iter_json_array, atomic_json, file_sha256
from data_generation.context_formation.construct_contexts import build_one, validate_docs
from .scoring.matching import token_phrase_in_text, aliases

FAMILIES = ("relevant", "irrelevant", "random")


def pool(hop, depth, family):
    base = "context_distractors" if family == "random" else "context_" + family
    return hop.get(base + ("_remaining" if depth > 1 else ""))


def entity_present(text, entity):
    return bool(isinstance(entity, str) and entity.strip() and
                re.search(r"(?<!\w)" + re.escape(entity.strip()) + r"(?!\w)", text, re.I))


def check_chain(entry, family):
    if family not in FAMILIES: raise ValueError("Select relevant, irrelevant or random explicitly")
    if not isinstance(entry, dict) or not isinstance(entry.get("hops"), dict): raise ValueError("Missing hops object")
    for field in ("original_question", "answer"):
        if not isinstance(entry.get(field), str) or not entry[field].strip(): raise ValueError("Missing " + field)
    warnings = []; previous_docs = []; previous_question = ""
    trace = entry.get("decomposition", [])
    if not isinstance(trace, list): raise ValueError("decomposition must be a list")
    levels = {}
    for step in trace:
        if not isinstance(step, dict): raise ValueError("Invalid decomposition step")
        if step.get("hop_level") in levels: raise ValueError("Repeated decomposition hop level")
        levels[step.get("hop_level")] = step
    for depth in range(1, 5):
        hop = entry["hops"].get(f"n{depth}")
        if not isinstance(hop, dict) or "error" in hop: raise ValueError(f"Missing/failed n{depth}")
        q = hop.get("question")
        if not isinstance(q, str) or not q.strip(): raise ValueError(f"n{depth}: missing question")
        gold = hop.get("golden_docs" if depth == 1 else "all_golden_docs")
        distractors = pool(hop, depth, family)
        validate_docs(gold, distractors, depth)
        if depth == 1:
            if q.strip() != entry["original_question"].strip(): warnings.append({"hop": depth, "issue": "n1_original_question_mismatch"})
        else:
            if not set(previous_docs).issubset(gold): warnings.append({"hop": depth, "issue": "non_cumulative_gold_evidence"})
            step = levels.get(depth, {})
            bridge = hop.get("bridge_entity_replaced") or step.get("entity_to_resolve")
            if not bridge: warnings.append({"hop": depth, "issue": "missing_bridge_trace"})
            elif not entity_present(previous_question, bridge):
                warnings.append({"hop": depth, "issue": "bridge_absent_from_previous_question", "bridge": bridge})
            if bridge and entity_present(q, bridge): warnings.append({"hop": depth, "issue": "bridge_still_in_new_question", "bridge": bridge})
            if step.get("entity_to_resolve") and hop.get("bridge_entity_replaced") and step["entity_to_resolve"] != hop["bridge_entity_replaced"]:
                warnings.append({"hop": depth, "issue": "bridge_trace_disagrees_with_hop"})
        spans = [i for i, d in enumerate(distractors) if any(token_phrase_in_text(a, d) for a in aliases(entry["answer"]))]
        if spans: warnings.append({"hop": depth, "issue": "gold_answer_span_in_distractor", "indices_zero_based": spans})
        previous_docs, previous_question = gold, q
    return warnings


def audit_dataset(path, family, expected_chains=None):
    count = 0; issues = []; invalid = []; duplicates = []; seen = set(); max_chars = 0
    for index, entry in enumerate(iter_json_array(path)):
        count += 1
        q = entry.get("original_question")
        if isinstance(q, str):
            if q in seen: duplicates.append(index)
            seen.add(q)
        try:
            flags = check_chain(entry, family)
            for d in range(1, 5):
                h = entry["hops"][f"n{d}"]
                docs = h["golden_docs" if d == 1 else "all_golden_docs"] + pool(h, d, family)
                max_chars = max(max_chars, sum(map(len, docs)))
            if flags: issues.append({"source_entry_index": index, "issues": flags})
        except (ValueError, TypeError) as e:
            invalid.append({"source_entry_index": index, "reason": str(e)})
    if count == 0: invalid.append({"reason": "empty dataset"})
    if expected_chains is not None and count != expected_chains: invalid.append({"reason": f"Expected {expected_chains} chains; found {count}"})
    if duplicates: invalid.append({"reason": "duplicate original_question", "indices": duplicates})
    return {"sha256": file_sha256(path), "chains": count, "distractor_family": family,
            "schema_valid": not invalid, "invalid_records": invalid,
            "chains_needing_review": len(issues), "quality_issues": issues,
            "max_context_characters": max_chars,
            "note": "Lexical and structural checks do not establish semantic validity. A gold span in a distractor is not proof it answers the question."}


def cap_distractors(entry, family, policy="none", tokenizer=None):
    """Use stored membership/order. Only explicitly selected length policy alters text."""
    if policy not in {"none", "legacy_word_min", "tokenizer_min"}: raise ValueError("Unknown length policy")
    out = dict(entry); out["hops"] = {}; report = {}
    for depth in range(1, 5):
        hop = dict(entry["hops"][f"n{depth}"])
        docs = list(pool(hop, depth, family))
        caps = []
        if policy != "none":
            if family == "random": raise ValueError("Random single-pool datasets use --length-policy none")
            rel, irr = pool(hop, depth, "relevant"), pool(hop, depth, "irrelevant")
            if not isinstance(rel, list) or not isinstance(irr, list) or len(rel) != len(irr) or len(rel) != len(docs):
                raise ValueError("Length matching needs both stored pools with equal slot counts")
            if policy == "legacy_word_min":
                caps = [min(int(len(a.split()) * 1.3), int(len(b.split()) * 1.3)) for a, b in zip(rel, irr)]
                def cut(text, n):
                    if n <= 0: return ""
                    words = text.split(); count = max(1, int(n / 1.3))
                    return " ".join(words[:count]) if len(words) > count else text
            else:
                if tokenizer is None: raise ValueError("tokenizer_min needs a tokenizer")
                caps = [min(len(tokenizer.encode(a, add_special_tokens=False)), len(tokenizer.encode(b, add_special_tokens=False))) for a, b in zip(rel, irr)]
                def cut(text, n):
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    return tokenizer.decode(ids[:n], skip_special_tokens=True) if len(ids) > n else text
            docs = [cut(d, c) for d, c in zip(docs, caps)]
        hop["context_distractors" if depth == 1 else "context_distractors_remaining"] = docs
        out["hops"][f"n{depth}"] = hop
        report[f"n{depth}"] = {"policy": policy, "slot_caps": caps,
            "original_chars": [len(d) for d in pool(entry["hops"][f"n{depth}"], depth, family)],
            "used_chars": list(map(len, docs)), "gold_truncated": False}
    return out, report


def tasks_for_chain(entry, index, family, seed=42, length_policy="none", tokenizer=None):
    mapped, caps = cap_distractors(entry, family, length_policy, tokenizer)
    tasks = build_one(mapped, index, seed)
    for row in tasks:
        row["distractor_family"] = family
        row["length_matching"] = caps[row["hop"]]
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--distractor-family", choices=FAMILIES, required=True)
    p.add_argument("--expected-chains", type=int)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    a = p.parse_args(); report = audit_dataset(a.input, a.distractor_family, a.expected_chains)
    atomic_json(a.report, report)
    print(json.dumps({k: report[k] for k in ("chains", "schema_valid", "chains_needing_review", "max_context_characters")}, indent=2))
    if not report["schema_valid"] or (a.strict and report["chains_needing_review"]): raise SystemExit(2)

if __name__ == "__main__": main()
