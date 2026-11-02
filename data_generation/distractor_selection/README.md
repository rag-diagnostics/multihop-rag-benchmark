# Random distractor selection
## Input

An existing QA workbook with columns `question`, `answer`, `Document` (one gold
question/answer/document per row). The workbook is **not** included in the
repository; it can be supplied with `--input-xlsx`.

## Filtering

For each base QA instance, sample without replacement from the other documents
in the corpus. Exclude the gold document and any candidate containing any
comma-separated gold-answer variant, using case-insensitive word-token matching.
In the default `llm` mode, additionally ask Gemini, using the archived
extraction-only checker prompt, whether the candidate alone can answer the
question. Accept only an explicit `UNKNOWN`; reject ambiguous/model-error cases
(fail closed). Continue down the shuffled candidate list until exactly **10**
unique distractors pass. The resulting n=1 context therefore has 1 gold + 10
distractors, for a total of 11 documents.


## Install

```bash
python -m pip install -r data_generation/requirements.remaining.txt
```

## Full check (API access required)

```bash
export GEMINI_API_KEY="<set-this-locally-not-in-GitHub>"
python data_generation/distractor_selection/select_random_distractors.py \
  --input-xlsx /path/to/updated_modified_final_dataset_6_02.xlsx \
  --output data_generation/work/rag_benchmark_clean_new.json \
  --audit data_generation/work/random_distractor_audit.json \
  --seed 42 --count 10 --check-mode llm
```
