# Narrative Generation

Full transcripts with stable IDs become full narrative JSON. The recovered prompt is in prompt.py. Default model: Gemini 2.5 Flash; override explicitly for another experiment.

Run from the repository root:

```bash
python -m data_generation.narrative_generation.generate_narratives --input work/selected_sources.json --output work/narratives.json
```

Use the [complete construction guide](../../docs/BUILD_DATASET.md) for dependencies, model options and stage ordering. The orchestrator `python -m data_generation.run_pipeline` checks the full sequence. JSON is canonical for long text; exporting oversized Excel cells fails explicitly. Existing notebook entry points import the Python implementation. See [the changes](../../docs/FIXES.md) for intentional validation changes and historical-parity boundaries.
