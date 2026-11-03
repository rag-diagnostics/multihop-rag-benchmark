# Controlled eleven-document contexts

A1/A2/A3 place the single gold document first/middle/last. At depths 2-4, B1 places the gold block centrally in canonical order, B2 reverses the block, and B3 shuffles gold and distractors with the recorded seed. Each chain yields 12 contexts and every context has 11 distinct documents.

For the released 528-chain JSON, use the [direct evaluator](../../evaluation/README.md). It calls these positioning functions in memory. **You do not need to export context files or create shards.** Stored relevant/irrelevant pools and their length-matching policies are selected explicitly in the evaluator.

The standalone interfaces remain for compatibility:

```bash
python data_generation/context_formation/construct_contexts.py --input work/multihop_validated.json --output work/contexts.json --seed 42
```

This expects the newly generated single-pool schema (`context_distractors` and higher-hop `context_distractors_remaining`). `prepare_released_dataset.py` is the earlier export adapter for the two-pool release; its shard feature is optional. It does not perform the full quality audit or the explicit length-matching policies in the new direct runner, so its output is not the default execution path.

The original positioning function bodies are retained in `original_singlehop.py` and `original_multihop.py`. The wrapper uses stable seed derivation; this is not a recovery of unrecorded historical Python `hash()` seeds. The paper's A/B labels refer to hop placement/formation, while some older notebook TestA/TestB labels referred to different distractor pools. They are not interchangeable naming systems.

Placement checks do not establish semantic chain validity. The direct evaluator records audit warnings and selected source identities. See [FIXES.md](../../docs/FIXES.md).
