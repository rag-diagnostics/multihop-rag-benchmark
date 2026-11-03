# Changes and scientific boundaries

This update is based on the release packages supplied in this conversation, not a verified checkout of the current GitHub branch. It fixes observed code defects and makes experimental choices explicit. It does not retroactively prove the historical model choices, calibration data, sampling IDs, or reported results.

## Existing JSON first

| Issue | Correction |
|---|---|
| Released JSON field names differ from exported-context schema | Direct runner reads n1 gold plus n2-n4 cumulative gold and the explicitly chosen stored pool. Contexts are built in memory. |
| Mandatory preprocessing/sharding complicated a previously direct workflow | Neither is required; generation reads the JSON array one chain at a time. |
| Source scripts capped distractors; former adapter did not | `none`, `legacy_word_min`, and `tokenizer_min` are explicit choices. No gold truncation or hidden pool mixing. |
| Judging rule changed from the uploaded script | `legacy_short` retains the original exact/short/long routing and original type-aware prompt; `semantic_all` is separately named. |
| Judge model/budget changed silently | CLI model/policy choices are explicit; the 50-vs-15 generator budget is documented and configurable. Judge output budget is independently recorded. |
| Old judgments reused after judge changes | Model, prompt, code, policy, budget, gold, and actual prediction enter the judgment fingerprint. |
| Generator resume could mix altered experiments | Dataset hash, sample indices, model settings, code, caps, seed and family enter a checked manifest. |
| API/parse errors looked like incorrect answers or unsafe empty success | Distinct operational statuses, retries, coverage checks, and withheld final metrics. |
| Citation number substring comparison treated 1 as 11 | Exact integer membership in gold positions. |
| Whole-response handling changed to first-line extraction | Main generation stores and scores the original response; citation mode parses only its explicitly requested format. |
| Ambiguous diagnostic denominator | Per-condition coverage, no-answer counts, non-causal distractor string diagnostic, explicit CCS reference condition. |

The shared generation prompt matches the archived primary prompt, not necessarily every model-specific experimental variant. Local models use their checkpoint's chat template and greedy decoding. Exact equivalence to all old notebooks is not claimed. The normalized distractor phrase diagnostic is not a reproduction of every original fuzzy-token analysis routine. No analysis or ablation scripts are added.

## New dataset construction

| Observed defect | Correction |
|---|---|
| Narrative workbook omitted `Serial` | Stable, unique, safe IDs are assigned and preserved across stages. |
| Excel truncated long transcripts | JSON is canonical; XLSX export refuses oversized cells before writing. |
| Raw `text` selected before transformed `Document` | Multihop loader requires `Document` explicitly. Schema adapter aligns it with transformed `Summary`. |
| Wrong returned answer still passed a hop | Compare the returned full-evidence answer with accepted gold variants. |
| Bridge absent from preceding question was accepted | Require actual bridge presence before replacement and absence afterward. |
| Shortcut chains could pass minimal checks | Require new evidence and every earlier gold document under separate leave-one-out checks. This is an intentional stricter validation rule. |
| API failure interpreted as inability to answer | Failure rejects/blocks the stage, rather than satisfying an abstention check. |
| Global API configuration raced across workers | Per-worker model wrapper with key in a request header. |
| Transformed documents changed but stale QA reused | Per-record and per-stage source/prompt/model identity. |
| Random question types changed after resume | Stable assignment by original source-row order. |
| Missing game phase invented a date / winter mislabeled | Recorded phase parsing with explicit failure; Winter retained correctly. |
| Failed PKS score silently kept | Default stop-on-error; fail-open behavior only through explicit `--on-error keep`. |

Narrative and QA prompts, the original token-wise entity/season/year/synonym transformation, and core PKS mathematics are retained where indicated. The I/O, exception handling, identifiers, checkpoints and validation requirements changed. Therefore new construction can produce different numbers of valid records.

The shared construction REST adapter uses explicit temperature 0 and top-p 1; this is not a claim that every recovered notebook used those same provider defaults. Calibration now publishes all three distributions atomically and verifies their hashes, avoiding mixed old/new baseline files after an interrupted run.

## Data problems are not code patches

An absent bridge or inconsistent question/document transformation in the released JSON cannot be reliably repaired by renaming fields or entities. The auditor reports such issues. Strict evaluation stops; report mode evaluates the original release without mutating or omitting its records and marks the output for review. The release does not include an automatically “repaired” 528-record dataset.

## Security and external requirements

No API tokens, credential JSON files, model weights, real datasets, or raw historical outputs are included. Revoke keys previously embedded in research scripts. Required data, upstream game engine, calibration QA sets, NLP resources, model access and sufficient hardware remain external inputs. Endpoint access and inference costs are not established by offline unit tests.
