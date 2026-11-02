# Evaluation: controlled multi-hop RAG generation (Section 4)

This directory evaluates generator models against the benchmark's fixed 11-document contexts, using the six evidence-placement/ordering configurations. It implements inference, exact-match screening, Gemini-based semantic equivalence checking, and diagnostic metrics. The released evaluation module focuses on the fixed-context benchmark; supplemental experiments are documented in the supplementary PDF.

## Experimental design

| Depth | Configuration | Placement |
|---|---|---|
| n1 | A1 | Gold document first |
| n1 | A2 | Gold document at position 6 |
| n1 | A3 | Gold document last |
| n2, n3, n4 | B1 | Gold block centred, chain order |
| n2, n3, n4 | B2 | Gold block centred, reversed order |
| n2, n3, n4 | B3 | Gold documents shuffled with distractors |

The input is the output of `data_generation/context_formation/construct_contexts.py` or `prepare_released_dataset.py`. For each complete four-hop chain it contains 12 context records, each with 11 documents, a hop question, gold answer, and gold positions. The same contexts are used for every model.

The primary prompt instructs each model to answer in 1–3 words using **only** the provided documents. The primary prompt uses the fixed-context generation instruction from the study. Decoding is greedy (`temperature=0`, `top_p=1`). The runner default is 50 output tokens; the primary configuration documented in Supplementary Material Section K.6 specifies 15, so the released-benchmark command explicitly uses `--max-tokens 15`. For the local Transformers engine, `do_sample=False` is used. Provider-specific model availability and parameter support should be confirmed before a run.

## Installation

Run from the repository root:

```bash
python -m pip install -r evaluation/requirements.txt
```

For local model inference through Transformers, install compatible versions of `torch`, `transformers`, and `accelerate` for the target GPU. Set provider tokens in your shell environment (`DEEPINFRA_TOKEN`, `GOOGLE_API_KEY`, or other names referenced in the model configuration). Do not commit these values.

Copy `evaluation/model_config.example.json` to a private configuration file and select the models and provider endpoints for your experiment. The example contains representative model identifiers used in the research scripts, **not** the full 12-model experimental manifest. To run local models, add entries with `"provider": "transformers"` and `"model_id": "/path/to/model"`. For an OpenAI-compatible local vLLM server, use `"provider": "openai_compatible"`, its server URL, and the configured key environment variable.

## Use the downloadable 528-instance benchmark

The [released PKS-filtered JSON](https://drive.google.com/file/d/1GmHj2K5SCaV0-a8xFjaj9UprHv6hbmXa/view?usp=sharing) can be evaluated without rerunning dataset generation or PKS filtering. Download it as `data_generation/work/benchmark_528.json`, then create the evaluator's expected input:

```bash
python data_generation/context_formation/prepare_released_dataset.py \
  --input data_generation/work/benchmark_528.json \
  --output data_generation/work/contexts_11doc_irrelevant.json \
  --distractor-family irrelevant --expected-chains 528 --seed 42

python -m evaluation.run_inference \
  --input data_generation/work/contexts_11doc_irrelevant.json \
  --models evaluation/model_config.example.json \
  --output evaluation/work/generations_irrelevant.jsonl \
  --run-tag released528_irrelevant --max-tokens 15 --dry-run
```

For the full 528-chain corpus, the context builder also supports `--shard-dir` and `--chains-per-shard` so the existing inference runner can operate on small files without loading all contexts simultaneously. The root README includes a sharded command loop. The judging/aggregation commands currently load their JSONL input into memory, so reserve adequate RAM or process smaller batches first. Model context-window capacity must be checked independently from Python memory capacity. Some experimental scripts also use per-slot distractor length matching; the adapter does not apply those model-specific caps.

The adapter selects one of the JSON's two distractor families. To compare the alternative `relevant` pool, prepare it separately and use a different run tag/output path. See the [root README](../README.md#run-the-released-benchmark) for the full download → evaluation → judging → aggregation sequence. For a new dataset with `context_distractors` fields, continue with the general context preparation procedure below.

## 1. Prepare the evaluation contexts

Complete dataset generation, PKS filtering, and multi-hop construction first. When the filtered multi-hop JSON is ready:

```bash
python data_generation/context_formation/construct_contexts.py \
  --input data_generation/work/filtered_multihop.json \
  --output data_generation/work/contexts_11doc.json \
  --seed 42
```

This step creates and freezes the A1–A3 and B1–B3 evidence orderings. Do not resample distractors or reposition evidence independently for each model.

## 2. Run model generation

Validate the planned experiment offline:

```bash
python -m evaluation.run_inference \
  --input data_generation/work/contexts_11doc.json \
  --models evaluation/model_config.example.json \
  --output evaluation/work/generations.jsonl \
  --dry-run
```

Then run the same command without `--dry-run` to generate responses. To select a single model use `--only-model qwen3-next-80b`. To conduct a small trial, add `--sample-chains 2 --seed 42`. Local Transformers inference requires `--workers 1`.

Each JSONL record stores the model ID, chain ID, hop, configuration, the 11 documents, evidence positions, raw generation, extracted answer, decoding metadata, and status. Completed requests are resumed by stable request ID; a different context/model/prompt fingerprint cannot be silently reused. Provider errors are recorded separately and are not treated as incorrect answers.

## 3. Tiered answer scoring

```bash
python -m evaluation.scoring.judge_answers \
  --input evaluation/work/generations.jsonl \
  --output evaluation/work/judged.jsonl \
  --judge-model gemini-2.5-pro
```

Stage 1 checks normalised exact match against comma-separated gold aliases. **Every non-exact output**, including short aliases and synonyms, is sent to the judge for strict factual equivalence. The judge receives only the question, the gold variants, and the generated response. It returns `CORRECT`, `INCORRECT`, or `NO_ANSWER` with a short normalised answer. Failed judge calls remain unscored and must be resolved before complete accuracy is reported. The original `evaluation.py` and `evaluation2.py` provide the basis for this judging approach; the packaged runner uses the supported `google-genai` SDK and the model specified above.

## 4. Aggregate the metrics

```bash
python -m evaluation.scoring.aggregate \
  --input evaluation/work/judged.jsonl \
  --output-dir evaluation/work/summary \
  --ccs-single A2
```

The aggregator exports `by_condition.csv`, `chain_completion.csv`, and `evaluation_metrics.json`:

- **Accuracy:** Correct answer after exact match plus judge, grouped by model/hop/configuration. Accuracy remains empty while any judgment is incomplete.
- **CMD:** Fraction of primary outputs containing 1–3 whitespace-delimited words.
- **CONFAB:** Fraction judged `NO_ANSWER`, including off-task, gibberish, or refusal. Missing judgments are not classified.
- **DISTRACTED / FABRICATED:** For judged incorrect answers, check whether the normalised predicted answer appears in a distractor (not a gold document). Otherwise report FABRICATED. Refusals/non-answers remain separate.
- **CCS:** Same-chain cumulative correctness at each hop depth. For a B1/B2/B3 series the default n1 reference is A2; use `--ccs-single` to choose a different n1 placement. A CCS is omitted unless all relevant chains and judgments are complete.

## 5. Optional evidence-attribution (EAR) probe

Evidence citation requires a **different prompt** and is evaluated independently of the primary accuracy experiment:

```bash
python -m evaluation.run_inference \
  --input data_generation/work/contexts_11doc.json \
  --models evaluation/model_config.example.json \
  --output evaluation/work/citations.jsonl \
  --mode citation --run-tag citation

python -m evaluation.scoring.judge_answers \
  --input evaluation/work/citations.jsonl \
  --output evaluation/work/citations_judged.jsonl

python -m evaluation.scoring.aggregate \
  --input evaluation/work/citations_judged.jsonl \
  --output-dir evaluation/work/citation_summary
```

The citation probe requests a two-line `Answer` and `Document` output. EAR here measures whether the cited integer document number is one of the gold document positions; no substring matching is applied to document numbers. For a multi-hop chain any required gold document is accepted as supporting evidence. The `00_new_metric3*.py` scripts are the source for this supplementary task.

## Source compatibility and run integrity

- `TestA1`–`TestB3` in some older inference scripts refer to two distractor **families** (relevant and irrelevant) and positional choices. They are **not identical** to the paper's A1–A3 single-hop / B1–B3 multi-hop conditions. This evaluator reads the canonical context-formation output instead.
- The live-retrieval benchmark uses BM25/BGE retrieval and **does not** measure the same generator-only oracle-context setting. Keep live-retrieval experiments separately under analysis.
- Provider model IDs, evaluation-set composition, output caps and the judge model must be recorded with each actual run. The example config is not the paper's exact model inventory.
- The CLI tests are offline and do not verify provider endpoints or reproduce historical numerical results; these require the released benchmark and actual model inference.

## Offline checks

```bash
python -m unittest discover -s evaluation/tests -v
```

All generated datasets, responses and reports belong in `evaluation/work/` (ignored by Git), not in the source-code release.
