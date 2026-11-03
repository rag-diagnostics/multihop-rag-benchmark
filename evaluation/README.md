# Evaluation of the released benchmark

Evaluate the published **528-chain JSON directly**, without regenerating data, rerunning PKS, preparing separate context files, or sharding. The pipeline generates answers, judges them, and reports accuracy and diagnostics. See the [root quick start](../README.md#quick-start).

## Quick start 

Download the [benchmark JSON](https://drive.google.com/file/d/1GmHj2K5SCaV0-a8xFjaj9UprHv6hbmXa/view?usp=sharing) to `data/benchmark_528.json`, configure the model IDs and credentials for your environment, and test two chains:

```bash
python -m evaluation.evaluate_dataset \
  --dataset data/benchmark_528.json --expected-chains 528 \
  --models evaluation/model_config.original_api.example.json \
  --only-model gemini-flash-lite --output-dir runs/test \
  --distractor-family irrelevant --length-policy legacy_word_min \
  --judge-model gemini-2.5-pro --judge-policy semantic_all \
  --max-tokens 50 --sample-chains 2 --seed 42 \
  --audit-policy report --dry-run
```

Remove `--dry-run` for real model calls. For the full benchmark, also remove `--sample-chains 2` and use a **new output directory**. `--only-model` is a key in the model configuration.

## Reproducibility settings

| Setting | Choice |
| --- | --- |
| `--distractor-family` | `relevant` or `irrelevant` selects one pool from the released JSON, `random` is for newly constructed data. |
| `--length-policy` | `none` preserves text, `legacy_word_min` matches distractor lengths , `tokenizer_min` uses a tokenizer. Gold documents are not truncated. |
| `--judge-policy` | `legacy_short` marks short non-exact answers incorrect without judging, `semantic_all` judges every non-exact answer. |
| `--match-policy` | `legacy` uses the original exact-match rules (default). |
| `--max-tokens` | Generation cap. |
| `--audit-policy` | `strict` stops on dataset-quality warnings. |



## Separate stages (optional)

`evaluation.evaluate_dataset` runs the stages in order. To run them separately, use `evaluation.run_dataset` for generation, then:

```bash
python -m evaluation.scoring.judge_answers \
  --input runs/test/generations.jsonl --output runs/test/judged.jsonl \
  --judge-model gemini-2.5-pro --judge-policy legacy_short

python -m evaluation.scoring.aggregate \
  --input runs/test/judged.jsonl --output-dir runs/test/summary \
  --ccs-single A2
```
