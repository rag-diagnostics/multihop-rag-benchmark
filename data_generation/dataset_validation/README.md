# Dataset Validation

Checks gold-answer variants, verbatim 1-3-word support and ambiguity. Failures of API or schema are operational failures, not valid judgments.

Run from the repository root:

```bash
python -m data_generation.dataset_validation.validate_gold --input work/base.json --output work/gold.json --errors work/gold_errors.json
```

Use the [complete construction guide](../../docs/BUILD_DATASET.md) for dependencies, model options and stage ordering. The orchestrator `python -m data_generation.run_pipeline` checks the full sequence. JSON is canonical for long text; exporting oversized Excel cells fails explicitly. Existing notebook entry points import the Python implementation. See [the changes](../../docs/FIXES.md) for intentional validation changes and historical-parity boundaries.
