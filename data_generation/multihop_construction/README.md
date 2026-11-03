# Multihop Construction

Reads Document explicitly, requires valid bridge substitution and gold-answer agreement, and performs full-evidence and leave-one-document-out checks. Run filter_multihop on the result before PKS; incomplete chains are rejected.

Run from the repository root:

```bash
python -m data_generation.multihop_construction.multi_create --input work/gold.json --global-docs work/documents_qa.json --output work/chains.json --errors work/chain_errors.json
```

Use the [complete construction guide](../../docs/BUILD_DATASET.md) for dependencies, model options and stage ordering. The orchestrator `python -m data_generation.run_pipeline` checks the full sequence.
