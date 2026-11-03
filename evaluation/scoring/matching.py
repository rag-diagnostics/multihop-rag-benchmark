"""Normalised exact-match tier and non-oracle answer extraction."""
from __future__ import annotations
import re
import unicodedata


def normalize(text: str) -> str:
    """Unicode/case/punctuation normalization; do not turn substring into EM."""
    s = unicodedata.normalize("NFKC", str(text or "")).casefold()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return " ".join(s.split())


def aliases(golden_answer: str) -> list[str]:
    return [x.strip() for x in str(golden_answer).split(",") if x.strip()]


def exact_match(pred: str, gold: str, policy: str = "normalized") -> bool:
    if policy == "legacy":
        normalizer = lambda x: str(x or "").strip().lower().rstrip(" \t\n\r.,!?;:")
    elif policy == "normalized":
        normalizer = normalize
    else:
        raise ValueError("Unknown exact-match policy")
    p = normalizer(pred)
    return bool(p) and any(p == normalizer(x) for x in aliases(gold))


def command_following(raw_response: str) -> bool:
    words = str(raw_response or "").split()
    return 1 <= len(words) <= 3


def token_phrase_in_text(answer: str, document: str) -> bool:
    needle = normalize(answer)
    haystack = normalize(document)
    return bool(needle and f" {needle} " in f" {haystack} ")
