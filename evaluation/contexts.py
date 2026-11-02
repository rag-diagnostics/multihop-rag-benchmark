"""Schema validation and repeatable selection of the canonical 11-document tasks."""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

CONFIG_BY_HOP = {"n1": {"A1", "A2", "A3"},
                 "n2": {"B1", "B2", "B3"},
                 "n3": {"B1", "B2", "B3"},
                 "n4": {"B1", "B2", "B3"}}


def task_key(row: Dict[str, Any]) -> str:
    return f"{row['source_entry_index']}|{row['hop']}|{row['configuration']}"


def validate_task(row: Dict[str, Any]) -> None:
    if not isinstance(row, dict):
        raise ValueError("Context row must be an object")
    depth = row.get("hop")
    config = row.get("configuration")
    if depth not in CONFIG_BY_HOP or config not in CONFIG_BY_HOP[depth]:
        raise ValueError(f"Incompatible hop/configuration: {depth}/{config}")
    ix = row.get("source_entry_index")
    if not isinstance(ix, int) or ix < 0:
        raise ValueError("source_entry_index must be a nonnegative integer")
    if not isinstance(row.get("question"), str) or not row["question"].strip():
        raise ValueError("Missing question")
    if not isinstance(row.get("answer"), str) or not row["answer"].strip():
        raise ValueError("Missing gold answer")
    docs = row.get("documents")
    if not isinstance(docs, list) or len(docs) != 11:
        raise ValueError("Every context must contain exactly 11 documents")
    if not all(isinstance(d, str) and d.strip() for d in docs) or len(set(docs)) != 11:
        raise ValueError("Context must contain 11 unique nonempty documents")
    positions = row.get("gold_positions_1_based")
    n = int(depth[1:])
    if not isinstance(positions, list) or len(positions) != n:
        raise ValueError(f"{depth} requires {n} gold positions")
    if any(not isinstance(p, int) or p < 1 or p > 11 for p in positions) or len(set(positions)) != n:
        raise ValueError("Invalid or repeated gold position")
    if depth == "n1" and positions != [{"A1": 1, "A2": 6, "A3": 11}[config]]:
        raise ValueError("Single-hop gold position disagrees with A1/A2/A3")
    if depth != "n1" and config in {"B1", "B2"}:
        first = (11 - n) // 2 + 1
        expected = list(range(first, first + n))
        if positions != expected:
            raise ValueError(f"{depth}/{config}: expected central gold positions {expected}")


def load_contexts(path: str | Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Context JSON must be a nonempty list")
    seen = set()
    for row in data:
        validate_task(row)
        key = task_key(row)
        if key in seen:
            raise ValueError(f"Duplicate context {key}")
        seen.add(key)
    return data


def select_tasks(data: Sequence[Dict[str, Any]], configurations: Optional[Sequence[str]],
                 sample_chains: Optional[int], seed: int) -> List[Dict[str, Any]]:
    if configurations:
        names = set(configurations)
        if not names.issubset({"A1", "A2", "A3", "B1", "B2", "B3"}):
            raise ValueError("Unknown configuration")
        data = [r for r in data if r["configuration"] in names]
    if sample_chains is not None:
        if sample_chains < 1:
            raise ValueError("sample_chains must be >=1")
        chains = sorted({r["source_entry_index"] for r in data})
        if sample_chains > len(chains):
            raise ValueError(f"Requested {sample_chains} chains but only {len(chains)} available")
        included = set(random.Random(seed).sample(chains, sample_chains))
        data = [r for r in data if r["source_entry_index"] in included]
    return list(data)
