# Generation-budget ablation

`generation_budget.py` runs three paired generation conditions on the **same 25 chains** (seed 42), using the prepared 11-document contexts:

| Condition | Output tokens | Output instruction |
|---|---:|---|
| `short15` | 15 | Answer in 1–3 words |
| `short128` | 128 | Answer in 1–3 words |
| `free128` | 128 | Explanation permitted; final answer requested |

To keep evidence formation fixed, this experiment uses n1/A2 and n2–n4/B1 (central gold evidence). It does **not** replace the primary six-configuration evaluation. The script shares the model interfaces and resume behaviour of `evaluation/run_inference.py`.

From the repository root:

```bash
python analysis/ablations/generation_budget.py \
  --input data_generation/work/contexts_11doc.json \
  --models evaluation/model_config.example.json \
  --output-dir analysis/ablations/work \
  --sample-chains 25 --seed 42 --dry-run
```

Remove `--dry-run` to execute provider calls. Use the judge and aggregator from `evaluation/scoring/` on each output JSONL before comparing correctness. Output files are not part of the source release.
