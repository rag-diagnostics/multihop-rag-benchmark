"""Shared no-context PKS mathematics and inference utilities.

The prompt and summed-token NLL/PMI formula match the original Qwen
parametric-filtering scripts. This module has no model imports at import time,
so classification can be tested without GPU or model downloads.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

CATEGORIES = ("common", "plausible", "nonsense")

# Preserve whitespace and instruction wording from 00_para14.py / para11.py.
TEMPLATE_NO_CONTEXT = """Instruction: You must respond based *only* on the information that you are trained with and the knowledge that you know. Answer the following question in just One to Three words.

Question: {q}

Answer: {a}"""

DEFAULT_CALIBRATION_FILENAMES = {
    "common": "7common_knowledge_scores_v2_no_ctx.json",
    "plausible": "7plausible_but_false_scores_v2_no_ctx.json",
    "nonsense": "7nonsense_scores_v2_no_ctx.json",
}
DEFAULT_QA_FILENAMES = {
    "common": "7common_knowledge.json",
    "plausible": "7plausible_but_false.json",
    "nonsense": "7nonsense_knowledge.json",
}


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
    tmp.replace(p)


def load_qa_pairs(path: str | Path) -> List[Dict[str, str]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array of QA objects: {path}")
    pairs: List[Dict[str, str]] = []
    for i, row in enumerate(data):
        if not isinstance(row, dict) or not isinstance(row.get("question"), str) or not isinstance(row.get("answer"), str):
            raise ValueError(f"Invalid question/answer at {path} row {i}")
        q, a = row["question"].strip(), row["answer"].strip()
        if not q or not a:
            raise ValueError(f"Empty question/answer at {path} row {i}")
        pairs.append({"question": q, "answer": a})
    if len(pairs) < 2:
        raise ValueError(f"Calibration category must contain at least two QA pairs: {path}")
    return pairs


def classic_pmi_from_nll(l_q: float, l_a: float, l_aq: float) -> float:
    """PKS(q,a) = L(q) + L(a) - L(q,a); L values are summed token NLLs."""
    values = [float(l_q), float(l_a), float(l_aq)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("PKS requires finite NLL values")
    return values[0] + values[1] - values[2]


def score_qa(nll_fn: Callable[[str], float], question: str, answer: str) -> float:
    """Return source-equivalent PKS with empty question/answer prompt slots."""
    l_q = nll_fn(TEMPLATE_NO_CONTEXT.format(q=question, a=""))
    l_a = nll_fn(TEMPLATE_NO_CONTEXT.format(q="", a=answer))
    l_aq = nll_fn(TEMPLATE_NO_CONTEXT.format(q=question, a=answer))
    return classic_pmi_from_nll(l_q, l_a, l_aq)


def calibration_stats(values: Iterable[float]) -> Dict[str, float]:
    scores = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    if len(scores) < 2:
        raise ValueError("Each baseline must have at least two finite PKS scores")
    mean = sum(scores) / len(scores)
    # Original np.std(scores) uses population standard deviation (ddof=0).
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    std = math.sqrt(variance)
    if not math.isfinite(std) or std <= 0:
        raise ValueError("Calibration standard deviation must be positive")
    return {"mean": mean, "std": std, "n": len(scores)}


def load_calibration(directory: str | Path) -> Dict[str, Dict[str, float]]:
    directory = Path(directory)
    provenance_path = directory / "calibration_summary.json"
    provenance = read_json(provenance_path) if provenance_path.is_file() else {}
    expected_hashes = provenance.get("score_sha256", {})
    stats: Dict[str, Dict[str, float]] = {}
    for cat in CATEGORIES:
        path = directory / DEFAULT_CALIBRATION_FILENAMES[cat]
        if expected_hashes and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes.get(cat):
            raise ValueError("Calibration score checksum mismatch: " + cat)
        data = read_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("pmi_classic"), list):
            raise ValueError(f"Expected a pmi_classic score list in {path}")
        stats[cat] = calibration_stats(data["pmi_classic"])
    return stats


def classify_pks(score: float, stats: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, float]]:
    """Nearest class by absolute Z-distance; tie order common/plausible/nonsense."""
    if not math.isfinite(float(score)):
        raise ValueError("Score must be finite")
    distances = {
        cat: abs((score - stats[cat]["mean"]) / stats[cat]["std"])
        for cat in CATEGORIES
    }
    return min(CATEGORIES, key=distances.__getitem__), distances


def normalized_questions(data: Any) -> List[Dict[str, Any]]:
    """Validate multi-hop rows; identify one answer per base question.

    The historical script re-used the previous classification for rows with an
    empty original_question. Such rows now raise a clear input error instead.
    """
    if not isinstance(data, list):
        raise ValueError("Benchmark input must be a JSON array")
    seen: Dict[str, str] = {}
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Row {i} is not a JSON object")
        q = item.get("original_question", "")
        a = item.get("answer", "")
        if not isinstance(q, str) or not q.strip():
            raise ValueError(f"Row {i} is missing original_question")
        if not isinstance(a, str) or not a.strip():
            raise ValueError(f"Row {i} is missing answer")
        q, a = q.strip(), a.strip()
        if q in seen and seen[q] != a:
            raise ValueError(f"Conflicting answers for original_question in row {i}: {q[:100]}")
        seen[q] = a
    return data


def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """Lazy Hugging Face loading, with no hard-coded HPC path or token."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    cuda = torch.cuda.is_available() if device == "auto" else device == "cuda"
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"use_cache": False}
    if cuda:
        kwargs.update({"device_map": "auto", "torch_dtype": torch.bfloat16})
    else:
        kwargs.update({"torch_dtype": torch.float32})
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not cuda:
        model.to("cpu")
    model.eval()
    return tokenizer, model


def nll_scorer(tokenizer: Any, model: Any) -> Callable[[str], float]:
    """Sum negative next-token log probabilities, as in the source scripts."""
    import torch

    def nll(text: str) -> float:
        inputs = tokenizer(text, return_tensors="pt")
        # The embedding device works for both single GPU and device_map="auto".
        device = model.get_input_embeddings().weight.device
        inputs = inputs.to(device)
        if inputs.input_ids.shape[1] < 2:
            return 0.0
        with torch.no_grad():
            logits = model(**inputs).logits
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs.input_ids[:, 1:]
            logp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            tok_logp = torch.gather(logp, 2, shift_labels.unsqueeze(-1)).squeeze(-1)
            score = -tok_logp.sum().item()
        if not math.isfinite(score):
            raise ValueError("Non-finite negative log-likelihood")
        return float(score)

    return nll
