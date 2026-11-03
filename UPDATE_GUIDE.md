# Apply the corrected code

## Which archive to use

**`repository_fixes_update.zip` is the recommended update.** It contains all added/changed files relative to the last release packages prepared in this conversation, with paths relative to the repository root. Merge it into a backup branch of the existing repository. It does not upload, edit, or delete anything remotely.

`multihop_rag_corrected_code.zip` contains the recovered code collection plus these corrections. It is not a fresh checkout of the current GitHub branch. Preserve your existing `assets/readme/` SVGs, `Supplementary Material.pdf`, licences, citation customizations, raw data and any newer local changes. No third-party engine, checkpoint weights, calibration research datasets, API credentials or benchmark JSON is bundled.

Do not delete the existing repository and replace it with the archive. Review the relative paths in `UPDATE_MANIFEST.json`; back up any locally changed file before overwriting it. Your 528-instance JSON is never modified by the update.

## Prioritised workflow

1. Merge the update, install `requirements.txt` in a fresh environment, and run `python -m pytest -q` after installing `requirements-dev.txt`.
2. Download the JSON with `python -m tools.download_dataset --output data/benchmark_528.json --expected-chains 528`, or download in a browser and run the same command with `--verify-only`.
3. Audit the file and run a two-chain dry run using the root README. Use the explicit family, length policy, token budget, judge policy and judge model that define the experiment.
4. Remove `--dry-run` for real generation, judging and metrics. Provider credentials, model access and adequate resources are required. Use a new directory for a full run.
5. Only for a new benchmark, follow `docs/BUILD_DATASET.md`. The released JSON does not need reconstruction or another PKS pass.

No sharding is needed. The single direct command is `python -m evaluation.evaluate_dataset`.

## Important migration choices

- Keep historical raw generations and judgments unchanged. Do not merge them into new JSONL outputs by hand.
- The new direct result interface is JSONL, not the old per-question nested JSON. This package does not silently reinterpret historical result files as new runs.
- `legacy_short` preserves the uploaded short-answer judging rule; `semantic_all` is an explicit different protocol. Choose, record, and keep scores separate.
- `legacy_word_min` reproduces the supplied API scripts' heuristic length caps; `tokenizer_min` uses a declared tokenizer; `none` does not truncate documents. Do not assume they produce the same inputs.
- Generator limits of 15 and 50 appear in different source descriptions. Select the one used by the experiment; do not change it under the same output directory.
- Strict audit warnings are not repaired by changing JSON field names. Report mode permits evaluation of the unchanged release but marks its results for data-quality review. The sample inconsistencies described in the earlier audit require source-level reconciliation.
- Retain original code/provenance in version control. The stronger construction validators deliberately reject cases the old validator accepted.
- Narrative and QA notebooks now call the Python implementation instead of carrying a second divergent copy. Run notebooks from the repository root.
- Rotate credentials previously embedded in old uploaded scripts. Do not copy them into the new files or share them here.

## Verification scope

Read `docs/TEST_REPORT.md`. Offline tests and simulated providers verify code flow, safety checks and interfaces. They do not establish that the entire remote Drive dataset is semantically valid, that every endpoint is currently accessible, or that historical paper scores will be reproduced.
