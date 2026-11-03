# Current release note

[FIXES.md](../docs/FIXES.md) documents the current I/O, validation, resume, and fail-closed changes. The following earlier provenance record describes the preceding release and must not override the current commands.

---

# What changed versus the archived research code

Source: user's `16-12-2025_final paper(1).zip`. No original generated datasets
are redistributed here. This package is **not** claimed to reproduce the
historical run byte-for-byte.

| Release file | Original file | What was changed |
|---|---|---|
| `distractor_selection/select_random_distractors.py` | `data_creation.ipynb` + `data_check.py` | **New implementation**: discarded Qwen-ranking, top/bottom, and 10+10 pools; implemented seeded random sampling, exclusion of gold document and answer spans, and optional full Gemini answerability screening (default on). Rejection of API errors is stricter than the original script's 'safe on ERROR' fallback. |
| `dataset_validation/validate_gold.py` | `data_check2.py` | Preserved prompt/acceptance logic; removed embedded key fallbacks, moved REST authentication to header, added CLI arguments. |
| `multihop_construction/multi_create.py` | `multi_create.py` | Preserved the original bridge construction and verification prompts; removed embedded keys, changed input to gold-validated output, replaced ranked-pool bookkeeping with one neutral random distractor pool, added path CLI arguments. |
| `multihop_construction/filter_multihop.py` | First code cell in `multi_check.ipynb` | Original code cell preserved. |
| `multihop_construction/audit_contexts.py` | None | New *offline* integrity checker for the 11-document budget. |

**Why this matters:** The manuscript describes randomly sampled distractors,
whereas the recovered old code includes noise-ranking and top/bottom selection.
This release now matches the **intended random selection policy**, but the
fact that it differs from the archived implementation must be acknowledged
when describing the provenance of historical experimental results.

Potential limitations: the span exclusion handles comma-separated gold-answer
variants and token boundaries, not every possible semantic alias. Model-based
checks may still make mistakes. If the paper's original numbers used different
selection, they must not be described as results of this newly adapted code
without rerunning the benchmark.

Still outside this package: raw game-to-narrative processing for LLM games,
300+300 source selection, PKS, human annotation records, model evaluation,
A1–A3/B1–B3 evidence ordering, and final benchmark data distribution.
