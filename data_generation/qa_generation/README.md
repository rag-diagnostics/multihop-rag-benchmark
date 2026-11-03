# Qa Generation

Uses the recovered 1-2-word QA prompt with explicit Gemini 2.5 Pro default. Creates/preserves Serial, checks answer support, and checkpoints against source and prompt identity.

Run from the repository root:

```bash
python -m data_generation.qa_generation.generate_qa --input work/transformed.json --output work/qa.json
```

Use the [complete construction guide](../../docs/BUILD_DATASET.md) for dependencies, model options and stage ordering. The orchestrator `python -m data_generation.run_pipeline` checks the full sequence. JSON is canonical for long text; exporting oversized Excel cells fails explicitly. Existing notebook entry points import the Python implementation. See [the changes](../../docs/FIXES.md) for intentional validation changes and historical-parity boundaries.
