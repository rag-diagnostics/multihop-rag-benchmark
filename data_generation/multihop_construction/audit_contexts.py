#!/usr/bin/env python3
"""Offline integrity audit of n=1..4 multihop contexts (no API or dataset editing)."""
import argparse, json
from pathlib import Path
import sys


def audit_one(example):
    errors=[]
    hops=example.get('hops',{})
    n1=hops.get('n1',{})
    base_doc=n1.get('golden_docs',[])
    pool=n1.get('context_distractors',[])
    for depth in range(1,5):
        hop=hops.get(f'n{depth}',{})
        if not hop or 'error' in hop:
            continue
        docs = base_doc if depth == 1 else hop.get('all_golden_docs', [])
        distractors = pool if depth == 1 else hop.get('context_distractors_remaining',[])
        if len(docs) != depth:
            errors.append(f'n{depth}: expected {depth} gold docs, got {len(docs)}')
        if len(docs) + len(distractors) != 11:
            errors.append(f'n{depth}: context length {len(docs)+len(distractors)} != 11')
        context = docs + distractors
        if len(context) != len(set(context)):
            errors.append(f'n{depth}: duplicate context documents')
        if any(not isinstance(x,str) or not x.strip() for x in context):
            errors.append(f'n{depth}: missing/invalid document')
    return errors


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('file', help='multihop JSON list')
    args=p.parse_args()
    entries=json.loads(Path(args.file).read_text(encoding='utf-8'))
    if not isinstance(entries, list):
        p.error('Expected top-level JSON list')
    failed=0
    for i, entry in enumerate(entries):
        problems=audit_one(entry)
        if problems:
            failed+=1
            print(f'entry #{i}: '+ '; '.join(problems))
    print(f'Integrity audit: {len(entries)} entries, {failed} with context issues')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
