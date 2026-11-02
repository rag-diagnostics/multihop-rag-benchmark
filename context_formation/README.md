# ECIR Section 3.7: context formation (11 documents)


## Conditions

| Condition | Hop(s) | Positioning |
|---|---|---|
| A1 | n1 | Single gold document first (position 1) |
| A2 | n1 | Single gold document in middle (position 6) |
| A3 | n1 | Single gold document last (position 11) |
| B1 | n2-n4 | Gold documents grouped centrally, canonical chain order |
| B2 | n2-n4 | Gold documents grouped centrally, reverse chain order |
| B3 | n2-n4 | All gold and distractor documents shuffled together |

Every condition has exactly **11 documents**. Gold/distractor counts are:
1+10, 2+9, 3+8, and 4+7 for hop depths 1, 2, 3, and 4.

## Evaluate the released 528-instance dataset

The downloadable [528-instance benchmark JSON](https://drive.google.com/file/d/1GmHj2K5SCaV0-a8xFjaj9UprHv6hbmXa/view?usp=sharing) uses `context_relevant`, `context_irrelevant`, and the corresponding `_remaining` arrays instead of `context_distractors` arrays. Use **`prepare_released_dataset.py`**, which maps these fields without modifying the source JSON and delegates evidence placement to the same `construct_contexts.build_one()` implementation.

```bash
# Save the downloaded file as data_generation/work/benchmark_528.json.
python data_generation/context_formation/prepare_released_dataset.py \
  --input data_generation/work/benchmark_528.json \
  --expected-chains 528 --check-only

python data_generation/context_formation/prepare_released_dataset.py \
  --input data_generation/work/benchmark_528.json \
  --output data_generation/work/contexts_11doc_irrelevant.json \
  --distractor-family irrelevant --expected-chains 528 --seed 42
```

Each of the 528 chains contributes 12 conditions, yielding **6,336 context rows**. The CLI also accepts `--distractor-family relevant` (write a separate output file) and `--limit 2` for a short trial. It validates all 11 documents per context, including exact gold/distractor counts and duplicate checks, and writes output atomically. Processing is incremental to avoid holding all contexts in memory simultaneously.

For large benchmarks, generate independently evaluable context shards instead of one multi-gigabyte output:

```bash
python data_generation/context_formation/prepare_released_dataset.py \
  --input data_generation/work/benchmark_528.json \
  --shard-dir data_generation/work/context_shards_irrelevant \
  --chains-per-shard 4 --distractor-family irrelevant \
  --expected-chains 528 --seed 42
```

The shard directory must not exist beforehand. Each shard contains complete sets of 12 contexts for its chains, with the original global chain index. Run `evaluation.run_inference` against each shard and combine its result JSONL files for scoring.




## Usage

Run from the GitHub repository root after the multi-hop dataset has been
constructed and failed chains removed:

```bash
python data_generation/context_formation/construct_contexts.py \
  --input data_generation/work/filtered_multihop.json \
  --output data_generation/work/contexts_11doc.json \
  --seed 42
```

The code reads a JSON list of chains with fields:

- Each entry: `original_question`, `answer`, `decomposition`, `hops`
- `hops.n1`: `question`, `golden_docs` (1 document), `context_distractors` (10)
- `hops.n2`, `.n3`, `.n4`: `question`, `all_golden_docs` (2/3/4),
  `context_distractors_remaining` (9/8/7)

It writes a JSON list containing one record per entry x hop x condition:
`source_entry_index`, `question`, `answer`, `hop`, `configuration`,
`documents` (ordered), `gold_positions_1_based`, `shuffle_seed` (B3 only),
`decomposition`, and `original_question`.
For a fully valid 4-hop chain, there are 12 output records: 3 single-hop and
9 multi-hop configurations.

You can run a source-only offline test:

```bash
python -m unittest discover -s data_generation/context_formation/tests -v
```
