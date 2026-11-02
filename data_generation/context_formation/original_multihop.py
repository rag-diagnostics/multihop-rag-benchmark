"""Original multi-hop context placement from gem/gem1.ipynb, cell 1.

The function below is copied without edits (including its fallback behavior).
Use the validated CLI wrapper for public releases.
"""
import random
from typing import List


def construct_context(gold_docs: List[str], distractors: List[str], test_type: str, seed: int) -> List[str]:
    working_gold = list(gold_docs)
    if "1" in test_type:
        mid = len(distractors) // 2
        return distractors[:mid] + working_gold + distractors[mid:]
    elif "2" in test_type:
        working_gold.reverse()
        mid = len(distractors) // 2
        return distractors[:mid] + working_gold + distractors[mid:]
    elif "3" in test_type:
        combined = working_gold + list(distractors)
        rng = random.Random(seed)   # deterministic per task
        rng.shuffle(combined)
        return combined
    return working_gold + list(distractors)
