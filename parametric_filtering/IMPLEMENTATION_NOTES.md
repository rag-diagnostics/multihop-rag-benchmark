# Current release note

[FIXES.md](../docs/FIXES.md) documents the current I/O, validation, resume, and fail-closed changes. The following earlier provenance record describes the preceding release and must not override the current commands.

---

# Implementation notes

## Source correspondence

The PKS implementation is based on the following research scripts:

| Research script | Public module | Retained operation |
|---|---|---|
| `para11.py` | `baseline_calibration/para11.py` | No-context NLL and classic PMI scoring of three calibration groups |
| `00_para14.py` | `pks_filtering/00_para14.py` | Qwen base-question scoring, nearest absolute Z-score, removal of the common class |
| `para19.py` | `validation/para19.py` | Independent Qwen PKS scoring, category and expected-count checks |
| `para13.py` | `validation/para13.py` | Reference-model calibration distribution comparison |

The shared no-context question-answer prompt preserves the text and whitespace of `00_para14.py`. The PMI calculation uses three forward passes on the formatted question-only, answer-only and question-answer strings; the original summed-token negative log-likelihood and `ddof=0` Z-score calibration are retained.

## Repository-focused interfaces

- Model locations and input/output paths are command-line arguments instead of local HPC paths.
- `para11.py` calculates the `pmi_classic` metric required by the PKS filter. The 18 additional exploratory metrics in the research calibration script are excluded from this focused release.
- Baseline caches are saved with the original three filenames and the `pmi_classic` key.
- Missing base questions, empty answers and conflicting answers for the same base question raise explicit input errors. The original row-loop inherited the previous decision when `original_question` was empty.
- The primary filter retains the original default behaviour for failed scoring: an unscorable question is placed in `nonsense` and therefore kept. An explicit `--on-error error` mode is offered for strict validation.
- `para19.py` re-scores and checks decisions independently. Unlike the primary filter, its default is to fail on scoring errors.
- `para13.py` accepts model configuration through JSON rather than requiring the local cluster's model directories. The reference-model comparison uses matching calibration samples for every model.

## Data and model dependency

To reproduce the reported numerical selection, use the calibration QA sets, checkpoint versions, prompt format, tokenizer and candidate question set from the corresponding research run. This code package contains no model weights, private cluster paths, calibration datasets or generated benchmark records.
