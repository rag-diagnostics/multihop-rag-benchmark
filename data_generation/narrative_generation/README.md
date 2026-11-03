# Narrative Generation

Full transcripts with stable IDs become full narrative JSON. 

Run from the repository root:

```bash
python -m data_generation.narrative_generation.generate_narratives --input work/selected_sources.json --output work/narratives.json
```

Use the [complete construction guide](../../docs/BUILD_DATASET.md) for dependencies, model options and stage ordering. 
