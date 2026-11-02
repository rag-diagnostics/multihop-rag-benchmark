"""Original single-hop context placement from gemini.ipynb, cell 0.

The function below is copied without edits (including its fallback behavior).
Use the validated CLI wrapper for public releases.
"""
from typing import List


def construct_context(golden_doc_text: str, distractor_docs: List[str], position: str) -> List[str]:
    if position == "first":
        return [golden_doc_text] + distractor_docs
    elif position == "last":
        return distractor_docs + [golden_doc_text]
    elif position == "middle":
        mid = len(distractor_docs) // 2
        return distractor_docs[:mid] + [golden_doc_text] + distractor_docs[mid:]
    return [golden_doc_text]
