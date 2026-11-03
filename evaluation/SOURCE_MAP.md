# Evaluation source map

| Release component | Basis |
|---|---|
| `prompting.py` primary prompt | Primary generation prompt recovered from the supplied notebooks. |
| `dataset.py` context positions | Existing `data_generation/context_formation/construct_contexts.py` and original positioning functions. |
| `legacy_word_min` policy | Supplied API scripts' word estimate and slot-wise minimum cap. |
| `scoring/legacy_prompt.py` | `build_llm_prompt` from the supplied `evaluation(1).py`; credentials excluded. |
| `legacy_short` policy | That uploaded script's exact-match / short-nonexact / long-nonexact routing. |
| `semantic_all` policy | Explicit alternative corresponding to the manuscript's all-nonexact semantic checking. |
| `run_dataset.py`, downloader, manifests and streaming I/O | New release integration, not historical scripts. |
| Metrics aggregator | Coverage-aware implementation with explicit normalized-phrase diagnostics and CCS condition pairing; not blanket numeric parity with original analysis notebooks. |

Read [the change record](../docs/FIXES.md) before comparing a new run with historical paper numbers. Raw model outputs and the exact experiment settings are preserved in each run.
