# Schema Adapters

Aligns the transformed Summary and Document without selecting the raw text column. Adds stable identifiers where missing and checks conflicts. Supports JSON and size-checked XLSX.

Run from the repository root:

```bash
python -m data_generation.schema_adapters.prepare_distractor_input --input work/qa.json --output work/documents_qa.json
```

Use the [complete construction guide](../../docs/BUILD_DATASET.md) for dependencies, model options and stage ordering. The orchestrator `python -m data_generation.run_pipeline` checks the full sequence.
