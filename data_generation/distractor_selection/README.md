# Distractor Selection

Samples ten distinct random candidates; excludes gold and answer-variant spans and performs extraction-only answerability screening. It does not rank documents. Do not use span-only mode as a claim of model-validated screening.

Run from the repository root:

```bash
python -m data_generation.distractor_selection.select_random_distractors --input work/documents_qa.json --output work/base.json --audit work/base_audit.json --seed 42
```

Use the [complete construction guide](../../docs/BUILD_DATASET.md) for dependencies, model options and stage ordering. The orchestrator `python -m data_generation.run_pipeline` checks the full sequence. JSON is canonical for long text; exporting oversized Excel cells fails explicitly. Existing notebook entry points import the Python implementation. See [the changes](../../docs/FIXES.md) for intentional validation changes and historical-parity boundaries.
