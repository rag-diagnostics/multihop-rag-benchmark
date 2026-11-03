#!/usr/bin/env python3
"""Create the canonical A1-A3/B1-B3 contexts from the 528-instance release JSON.

The released benchmark already contains gold evidence and two distractor families.
This adapter selects one family, preserves the dataset, and delegates every
placement to the existing context_formation.construct_contexts.build_one().
It does not resample distractors, filter questions, or call any LLM.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator

try:
    from .construct_contexts import build_one
except ImportError:
    from construct_contexts import build_one

HOPS = ("n1", "n2", "n3", "n4")
FAMILIES = ("irrelevant", "relevant")


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Read a top-level JSON array, one item at a time, with bounded memory.

    The research JSON can be large; parsing and writing an entire 528-chain
    context collection in memory is unnecessary.
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        buf = ""
        pos = 0
        eof = False

        def get_char() -> str:
            nonlocal buf, pos, eof
            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1
                if pos < len(buf):
                    return buf[pos]
                if eof:
                    return ""
                buf = ""
                pos = 0
                part = stream.read(chunk_size)
                if part:
                    buf = part
                else:
                    eof = True

        def consume(expected: str):
            nonlocal pos
            actual = get_char()
            if actual != expected:
                raise ValueError(f"Expected {expected!r} in JSON array, got {actual!r}")
            pos += 1

        consume("[")
        if get_char() == "]":
            consume("]")
        else:
            while True:
                while True:
                    try:
                        item, stop = decoder.raw_decode(buf, pos)
                        pos = stop
                        break
                    except json.JSONDecodeError as exc:
                        if eof:
                            raise ValueError("Truncated or invalid JSON array item") from exc
                        more = stream.read(chunk_size)
                        if not more:
                            eof = True
                        buf = buf[pos:] + more
                        pos = 0
                yield item
                sep = get_char()
                if sep == "]":
                    consume("]")
                    break
                consume(",")
                if get_char() in ("]", ""):
                    raise ValueError("Trailing comma or incomplete JSON array")
        if get_char():
            raise ValueError("Unexpected trailing content after JSON array")


def map_released_entry(entry: Dict[str, Any], family: str) -> Dict[str, Any]:
    """Map the released `context_relevant/irrelevant` fields to canonical keys."""
    if family not in FAMILIES:
        raise ValueError(f"Distractor family must be one of: {FAMILIES}")
    if not isinstance(entry, dict):
        raise ValueError("Each released entry must be a JSON object")
    hops = entry.get("hops")
    if not isinstance(hops, dict):
        raise ValueError("Missing `hops` object")
    normalized_hops = {}
    for depth, hop_name in enumerate(HOPS, 1):
        hop = hops.get(hop_name)
        if not isinstance(hop, dict):
            raise ValueError(f"Missing {hop_name} hop object")
        source_key = f"context_{family}" + ("_remaining" if depth > 1 else "")
        destination_key = "context_distractors_remaining" if depth > 1 else "context_distractors"
        docs = hop.get(source_key)
        if not isinstance(docs, list):
            raise ValueError(f"{hop_name}: missing array {source_key}")
        normalized_hops[hop_name] = dict(hop)
        normalized_hops[hop_name][destination_key] = docs
    transformed = dict(entry)
    transformed["hops"] = normalized_hops
    return transformed


def convert_file(input_path: Path, output_path: Path | None = None, *,
                 family: str = "irrelevant", seed: int = 42,
                 expected_chains: int | None = None, limit: int | None = None,
                 check_only: bool = False,
                 shard_dir: Path | None = None,
                 chains_per_shard: int = 4) -> Dict[str, Any]:
    """Validate the released JSON; write one full file or independent small shards.

    Sharding controls file size and inference memory without altering any task's
    documents, question, gold positions, or B3 random seed. Each shard contains
    all 12 configurations for its source chains. Source indices stay global.
    """
    if family not in FAMILIES:
        raise ValueError("Select --distractor-family irrelevant or relevant")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if expected_chains is not None and expected_chains < 1:
        raise ValueError("--expected-chains must be positive")
    if limit is not None and expected_chains is not None:
        raise ValueError("Do not combine --limit with --expected-chains")
    if chains_per_shard < 1:
        raise ValueError("--chains-per-shard must be positive")
    if shard_dir is not None and (output_path is not None or check_only):
        raise ValueError("--shard-dir cannot be combined with --output or --check-only")
    if not check_only and output_path is None and shard_dir is None:
        raise ValueError("Supply --output or --shard-dir unless --check-only is used")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if shard_dir is not None and shard_dir.exists():
        raise FileExistsError(f"Refusing to mix shard datasets in existing directory: {shard_dir}")

    temp_path = None
    staged_dir = None
    out = None
    emitted = 0
    count = 0
    shard_count = 0
    records_in_file = 0
    max_context_chars = 0

    def close_array():
        nonlocal out
        if out is not None:
            out.write("\n]\n")
            out.close()
            out = None

    def start_array(file_path: Path):
        nonlocal out, records_in_file
        out = file_path.open("w", encoding="utf-8")
        out.write("[\n")
        records_in_file = 0

    try:
        if shard_dir is not None:
            shard_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_dir = Path(tempfile.mkdtemp(prefix=".shards_tmp_", dir=shard_dir.parent))
        elif not check_only:
            assert output_path is not None
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = output_path.with_name(output_path.name + ".tmp")
            start_array(temp_path)

        for index, source in enumerate(iter_json_array(input_path)):
            if limit is not None and index >= limit:
                break
            standardized = map_released_entry(source, family)
            contexts = build_one(standardized, index, seed)
            if len(contexts) != 12:
                raise RuntimeError(f"Entry {index}: expected 12 contexts, got {len(contexts)}")
            if staged_dir is not None and count % chains_per_shard == 0:
                close_array()
                start_array(staged_dir / f"contexts_{shard_count:04d}.json")
                shard_count += 1
            count += 1
            for context in contexts:
                max_context_chars = max(max_context_chars, sum(map(len, context["documents"])))
                if out is not None:
                    context["distractor_family"] = family
                    if records_in_file:
                        out.write(",\n")
                    json.dump(context, out, ensure_ascii=False, separators=(",", ":"))
                    records_in_file += 1
                emitted += 1

        if expected_chains is not None and count != expected_chains:
            raise ValueError(f"Expected {expected_chains} chains; found {count}")
        if count == 0:
            raise ValueError("Input contains no complete chains")
        close_array()
        if staged_dir is not None:
            assert shard_dir is not None
            os.replace(staged_dir, shard_dir)
            staged_dir = None
        elif temp_path is not None:
            assert output_path is not None
            os.replace(temp_path, output_path)

        target = shard_dir if shard_dir is not None else output_path
        return {"validated_chains": count, "contexts": emitted,
                "distractor_family": family,
                "output": str(target) if not check_only else None,
                "shard_count": shard_count,
                "max_context_characters": max_context_chars,
                "check_only": check_only}
    finally:
        if out is not None:
            out.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        if staged_dir is not None and staged_dir.exists():
            shutil.rmtree(staged_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Downloaded 528-instance JSON")
    parser.add_argument("--output", type=Path, help="Canonical contexts JSON (not used with --shard-dir)")
    parser.add_argument("--shard-dir", type=Path, help="Write bounded-memory context shards to a NEW directory")
    parser.add_argument("--chains-per-shard", type=int, default=4, help="Chains per shard (default 4)")
    parser.add_argument("--distractor-family", choices=FAMILIES, default="irrelevant")
    parser.add_argument("--seed", type=int, default=42, help="Fixed B3 shuffle seed")
    parser.add_argument("--expected-chains", type=int, help="Fail unless input count matches")
    parser.add_argument("--limit", type=int, help="Process the first N chains for a small trial")
    parser.add_argument("--check-only", action="store_true", help="Validate input without writing contexts")
    args = parser.parse_args()
    report = convert_file(args.input, args.output, family=args.distractor_family,
                          seed=args.seed, expected_chains=args.expected_chains,
                          limit=args.limit, check_only=args.check_only,
                          shard_dir=args.shard_dir, chains_per_shard=args.chains_per_shard)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
