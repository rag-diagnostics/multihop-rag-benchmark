"""The generator-only prompt and a separate evidence-citation probe."""
from __future__ import annotations
import re
from typing import List, Optional, Tuple


def format_documents(documents: List[str]) -> str:
    return "\n\n".join(f"Document [{i+1}]:\n{doc}" for i, doc in enumerate(documents))


def main_prompt(question: str, documents: List[str]) -> str:
    """Exact wording of build_user_prompt() from the archived gemini.ipynb."""
    context_string = format_documents(documents)
    user_content = (
        "You must respond based *only* on the information from the provided documents below. "
        "Do not incorporate any external knowledge or infer any details. "
        "Answer the following question in just One to Three words.\n\n"
        f"Question: {question}\n\n"
        f"--- Documents ---\n{context_string}"
    )
    return user_content


def free_answer_prompt(question: str, documents: List[str]) -> str:
    """Separate generation-budget ablation prompt from deepinfra5.ipynb."""
    return (
        "Answer the question using only the information in the provided documents. "
        "Do not use external knowledge. "
        "You may explain your reasoning if useful. "
        "At the end of your response, write your answer on a separate line in the form:\n"
        "Final answer: <answer>\n\n"
        f"Question: {question}\n\n"
        f"--- Documents ---\n{format_documents(documents)}"
    )


def citation_prompt(question: str, documents: List[str]) -> str:
    """A separate supplementary probe, never used for primary answer accuracy."""
    return (
        "You must respond based *only* on the information from the provided documents below. "
        "Do not incorporate any external knowledge or infer any details. "
        "Answer the following question in just One to Three words, and additionally state the "
        "number of the document containing evidence for your answer.\n\n"
        "Respond STRICTLY in the following two-line format:\n"
        "Answer: <your one-to-three word answer>\n"
        "Document: <the document number>\n\n"
        f"Question: {question}\n\n"
        f"--- Documents ---\n{format_documents(documents)}"
    )


def parse_citation(response: str) -> Tuple[str, Optional[int]]:
    answer = response.strip()
    citation = None
    for line in response.splitlines():
        line = line.strip()
        m = re.fullmatch(r"Document\s*:\s*\[?(\d+)\]?\.?", line, flags=re.I)
        if m:
            citation = int(m.group(1))
        elif re.match(r"^Answer\s*:", line, flags=re.I):
            answer = line.split(":", 1)[1].strip()
    if citation is not None and answer == response.strip():
        answer = re.split(r"\bDocument\s*:", response, flags=re.I)[0].strip()
    return answer, citation


def extract_final_answer(response: str, *, free_answer: bool) -> str:
    text = str(response or "").strip()
    if not text:
        return ""
    if free_answer:
        answers = re.findall(r"^\s*Final\s+answer\s*:\s*(.+)$", text, re.I | re.M)
        if answers:
            return answers[-1].strip()
    line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", line, flags=re.I).strip()
