# Evaluation of the released benchmark

**Input:** the existing JSON array of benchmark chains. **Output:** raw model responses, explicitly judged labels, diagnostics, and coverage reports. No regeneration, PKS rerun, separate context file, or sharding is required.

Use the [root quick start](../README.md#quick-start). `evaluate_dataset.py` runs the stages in order. `run_dataset.py` runs generation only; `scoring/judge_answers.py` and `scoring/aggregate.py` can be run independently.

## Required choices

| Option | Meaning |
|---|---|
| `--distractor-family relevant` / `irrelevant` | Select exactly one stored pool in the released JSON. These are field labels, not proof of how that pool was historically selected. |
| `--distractor-family random` | Use the single `context_distractors` pool in newly constructed data. |
| `--length-policy none` | Preserve stored documents without length matching. |
| `--length-policy legacy_word_min` | Use the original API scripts' `int(word_count * 1.3)` estimate, matching corresponding relevant/irrelevant distractor slots to their minimum. Gold documents are never truncated. |
| `--length-policy tokenizer_min` | Match the two stored pools using an actual tokenizer; supply `tokenizer_id` for API models. This needs the local-model requirements. |
| `--judge-policy legacy_short` | Exact match is correct; a non-exact output with at most three Unicode word tokens is incorrect without judging; a longer non-exact output goes to the original type-aware prompt. |
| `--judge-policy semantic_all` | Every non-exact output goes to that judge prompt. This differs from the uploaded short-answer script and may change accuracy. |
| `--match-policy legacy` | Original strip/lowercase/trailing-punctuation EM, plus Unicode-regex word counting for CMD. Direct runner default. |
| `--match-policy normalized` | Unicode/case/punctuation normalization and whitespace CMD counting, compatible with the earlier general interface. |
| `--max-tokens` | Explicit generator cap; otherwise the model config value, then 50. The supplementary 15-token protocol must be selected deliberately. |
| `--judge-model`, `--judge-max-tokens` | Explicit judge identifier and output budget (default 2,048 to allow structured output). The preserved prompt is not evidence of historical model/budget identity. |

Do not combine scores across these policies, model revisions, source datasets, samples, or distractor families without explicitly defining the comparison. Each run records these choices. B3 uses the existing seeded context builder, not a claim to recover historical unrecorded random orders.

## Dataset-quality handling

The parser scans the whole JSON before a paid generation run, even when evaluating two chains. It checks record count, complete n1–n4 schema, 11 distinct documents, cumulative gold evidence, decomposition links, and whether gold-answer phrases appear in distractors. Missing or duplicated records and malformed JSON are hard failures.

`--audit-policy strict` also stops on lexical/chain warnings. `--audit-policy report` permits measuring the unchanged released file but records `dataset_quality_status: requires_review` in generated records and metric summaries. It does not drop, rename, repair, or resample data. A string match in a distractor does not prove answerability, and lexical checks cannot certify a question's semantic correctness. Resolve semantic inconsistencies using the source records and validation evidence, not automated entity substitutions.

New dataset construction uses strict validation. Historical evaluation can retain the original data with an explicit report policy. These are different tasks.

## Separate generation and judging commands

```bash
python -m evaluation.run_dataset \
  --dataset data/benchmark_528.json --expected-chains 528 \
  --models evaluation/model_config.original_api.example.json \
  --only-model gemini-flash-lite --output-dir runs/example \
  --distractor-family irrelevant --length-policy legacy_word_min \
  --max-tokens 50 --sample-chains 2 --seed 42 --dry-run
```

Inspect the audit and remove `--dry-run` for a live run. Use `--audit-policy report` only to run the original release with its quality warnings recorded. Then:

```bash
python -m evaluation.scoring.judge_answers \
  --input runs/example/generations.jsonl --output runs/example/judged.jsonl \
  --judge-model gemini-2.5-pro --judge-policy legacy_short
python -m evaluation.scoring.aggregate \
  --input runs/example/judged.jsonl --output-dir runs/example/summary --ccs-single A2
```

`--only-model` takes keys from the configuration, not remote model identifiers. The example configurations are editable connection examples, not a complete or live-verified paper roster. API availability, access, context limits, and costs are checked in the user's environment. No embedded credentials are accepted. The direct evaluator reads process environment variables; merely creating a `.env` file does not load it automatically.

For a local model install `requirements-local-models.txt`, use `model_config.local.example.json`, and run one local model at a time. The input is not silently truncated to fit a checkpoint's window. A context-window failure is an operational error requiring an explicit experimental decision. A fixed number of documents does not mean a fixed number of tokens.

## Outputs, resume, and failures

- `dataset_audit.json`: full-input structure, counts, checksum, and quality warnings.
- `run_manifest.json`: dataset hash, sample indices, code identity, models, token budgets, seed, family, and length/match settings.
- `generations.jsonl`: original question/context identifiers, the actual supplied documents, untouched raw responses, extracted citation fields when requested, status, usage, finish reason, and generation fingerprint.
- `judged.jsonl`: predictions plus judge model/policy, prompt/code fingerprint, strict typed verdict, normalized answer, and judgment status.
- `summary/`: per-condition CSVs, explicit CCS reference condition, and JSON coverage metrics.

The records retain the actual supplied documents, so output files can be large even though processing is incremental. JSONL is an append-only record log. On retries, readers use the latest entry for each request ID. Successful identical work is reused; operational failures are retried. Changing generation inputs/settings/code requires a new output directory. Changing judge model, policy, budget, prompt/code, or prediction invalidates old judgments. Keep raw generations: rejudging should not require paying for generation again.

An active `.lock` prevents concurrent writers. After an interrupted process, remove a stale lock only after checking no writer is active. A partially written JSONL tail causes a clear parse error rather than silently losing data; preserve a backup before removing only the truncated final line.

Generation and judge transport/schema failures are **not wrong answers and not confabulations**. Final metrics are withheld for incomplete groups or missing scheduled records. The separate `observed_scored_accuracy` describes only the scored subset and must not be presented as a complete-run result.

## Metric definitions

Accuracy is the mean of final Boolean correctness labels when the scheduled group is complete. CMD uses raw generator output, not judge-normalized text. CONFAB uses the `NO_ANSWER` verdict and therefore depends on the chosen judging policy. A short non-exact answer in `legacy_short` is not semantically diagnosed.

DISTRACTED/FABRICATED use normalized, token-boundary phrase presence in distractors. This is an explicit lexical diagnostic, not the fuzzy token-matching procedure in every historical analysis notebook, and not causal proof. The decomposition is retained for inspection; intermediate-hop failure percentages are not automatically reconstructed.

CCS uses the explicitly selected single-hop baseline (default A2), then B1, B2, or B3 consistently at deeper hops of the same chain. It requires all four depths; incomplete chains are reported and final CCS is withheld. This convention must match the experiment being compared.

Citation probing is separate: use `python -m evaluation.run_dataset ... --mode citation --run-tag citation --max-tokens 60` in a new directory. A cited document must equal an integer gold position. Position 1 does not match position 11. EAR is a document-citation check, not verification that every reasoning step was followed. The two-line citation prompt and output budget differ from primary generation.

## Compatibility interfaces

`run_inference.py` and `prepare_released_dataset.py` remain for previously exported context files. They are not needed for the direct workflow. The old sharded export is optional and does not shorten a model's input. The direct workflow is the recommended entry point for the 528-chain JSON.

See [changes](../docs/FIXES.md), [source map](SOURCE_MAP.md), and [verification](../docs/TEST_REPORT.md).
