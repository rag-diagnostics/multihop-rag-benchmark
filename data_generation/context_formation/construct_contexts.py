#!/usr/bin/env python3
"""Build ECIR A1-A3 and B1-B3 11-document contexts from multihop JSON.

This only arranges already selected gold documents and screened RANDOM distractors.
It does not rank distractors, call any model, or modify the archived positioning
functions. See README.md and PROVENANCE.md for the source and limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from .original_singlehop import construct_context as _singlehop
    from .original_multihop import construct_context as _multihop
except ImportError:
    from original_singlehop import construct_context as _singlehop
    from original_multihop import construct_context as _multihop


CONTEXT_SIZE = 11
SINGLE_CONFIGS = {'A1': 'first', 'A2': 'middle', 'A3': 'last'}
MULTI_CONFIGS = ('B1', 'B2', 'B3')


def stable_seed(base_seed: int, entry_index: int, depth: int, configuration: str) -> int:
    """Stable cross-process seed; distinct from historical Python hash() seeding."""
    identity = f'{base_seed}:{entry_index}:n{depth}:{configuration}'
    return int.from_bytes(hashlib.sha256(identity.encode('utf-8')).digest()[:4], 'big')


def validate_docs(gold_docs: Any, distractors: Any, depth: int) -> None:
    """Fail rather than accidentally creating malformed or biased contexts."""
    if not isinstance(gold_docs, list) or not isinstance(distractors, list):
        raise ValueError(f'n{depth}: expected lists of gold and distractor documents')
    if len(gold_docs) != depth:
        raise ValueError(f'n{depth}: expected {depth} gold documents, found {len(gold_docs)}')
    if len(distractors) != CONTEXT_SIZE - depth:
        raise ValueError(f'n{depth}: expected {CONTEXT_SIZE - depth} distractors, found {len(distractors)}')
    docs = gold_docs + distractors
    if not all(isinstance(x, str) and x.strip() for x in docs):
        raise ValueError(f'n{depth}: all documents must be non-empty strings')
    if len(set(docs)) != CONTEXT_SIZE:
        raise ValueError(f'n{depth}: duplicate text among gold and distractor documents')


def build_one(entry: Dict[str, Any], entry_index: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Emit exactly 12 context configurations from a complete 4-depth chain.

    Input shape is produced by the prepared multihop_construction/multi_create.py:
    n1 has `golden_docs` and `context_distractors`; n2-n4 each have
    `all_golden_docs` and `context_distractors_remaining`.
    """
    if not isinstance(entry, dict):
        raise ValueError('Each input entry must be a JSON object')
    hops = entry.get('hops')
    if not isinstance(hops, dict):
        raise ValueError('Missing object: hops')
    results: List[Dict[str, Any]] = []
    base_q = entry.get('original_question')
    if not isinstance(base_q, str) or not base_q.strip():
        raise ValueError('Missing original_question')
    answer = entry.get('answer')
    if answer is None or not str(answer).strip():
        raise ValueError('Missing answer')

    for depth in range(1, 5):
        hop = hops.get(f'n{depth}')
        if not isinstance(hop, dict) or 'error' in hop:
            raise ValueError(f'Missing or failed n{depth} hop; filter failed chains first')
        gold_docs = hop.get('golden_docs') if depth == 1 else hop.get('all_golden_docs')
        distractors = (hop.get('context_distractors') if depth == 1
                       else hop.get('context_distractors_remaining'))
        validate_docs(gold_docs, distractors, depth)
        question = hop.get('question')
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'n{depth}: missing question')

        config_names = SINGLE_CONFIGS if depth == 1 else MULTI_CONFIGS
        for configuration in config_names:
            shuffle_seed = None
            if depth == 1:
                ordered = _singlehop(gold_docs[0], distractors, SINGLE_CONFIGS[configuration])
            else:
                shuffle_seed = stable_seed(seed, entry_index, depth, configuration)
                ordered = _multihop(gold_docs, distractors, configuration, shuffle_seed)
            if len(ordered) != CONTEXT_SIZE or set(ordered) != set(gold_docs + distractors):
                raise AssertionError(f'n{depth}/{configuration}: placement changed document membership')
            positions = [i + 1 for i, doc in enumerate(ordered) if doc in gold_docs]
            if len(positions) != depth:
                raise AssertionError(f'n{depth}/{configuration}: gold document count changed')
            results.append({
                'source_entry_index': entry_index,
                'original_question': base_q,
                'question': question,
                'answer': answer,
                'hop': f'n{depth}',
                'configuration': configuration,
                'documents': ordered,
                'gold_positions_1_based': positions,
                'shuffle_seed': shuffle_seed if configuration == 'B3' else None,
                'decomposition': entry.get('decomposition', []),
            })
    return results


def build_all(entries: List[Dict[str, Any]], seed: int = 42, skip_incomplete: bool = False) -> List[Dict[str, Any]]:
    if not isinstance(entries, list):
        raise ValueError('Input JSON must be an array of multihop entries')
    result: List[Dict[str, Any]] = []
    skipped = 0
    for i, entry in enumerate(entries):
        try:
            result.extend(build_one(entry, i, seed))
        except (ValueError, AssertionError) as exc:
            if not skip_incomplete:
                raise ValueError(f'Entry {i}: {exc}') from exc
            skipped += 1
            print(f'Skipping entry #{i}: {exc}')
    print(f'Constructed {len(result)} contexts from {len(entries) - skipped} chains '
          f'({skipped} skipped)')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path,
                        help='Filtered output JSON from multihop_construction/')
    parser.add_argument('--output', required=True, type=Path, help='Where to write ordered contexts JSON')
    parser.add_argument('--seed', type=int, default=42,
                        help='Reproducibility seed for B3 ordering; default: 42')
    parser.add_argument('--skip-incomplete', action='store_true',
                        help='Skip incomplete/corrupt chains; default is to fail explicitly')
    args = parser.parse_args()
    entries = json.loads(args.input.read_text(encoding='utf-8'))
    output = build_all(entries, seed=args.seed, skip_incomplete=args.skip_incomplete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
