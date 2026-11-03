# Entity Remapping

Retains the source token-wise remapping, season/year changes and WordNet synonym probability. Source rows must not already contain QA or chains. Summary and Document are updated together; maps and hashes are recorded.

Run from the repository root:

```bash
python -m data_generation.entity_remapping.remap_entities --input work/narratives.json --output work/transformed.json --seed 42
```

Use the [complete construction guide](../../docs/BUILD_DATASET.md) for dependencies, model options and stage ordering. The orchestrator `python -m data_generation.run_pipeline` checks the full sequence. JSON is canonical for long text; exporting oversized Excel cells fails explicitly. Existing notebook entry points import the Python implementation. See [the changes](../../docs/FIXES.md) for intentional validation changes and historical-parity boundaries.
