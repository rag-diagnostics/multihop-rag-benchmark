# ECIR Section 3.7: context formation 
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


## Usage


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
