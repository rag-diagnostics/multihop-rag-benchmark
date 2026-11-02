# Evaluation: controlled multi-hop RAG generation (Section 4)

This directory evaluates generator models against the benchmark's fixed 11-document contexts, using the six evidence-placement/ordering configurations. It implements inference, exact-match screening, Gemini-based semantic equivalence checking, and diagnostic metrics. Generation-budget experiments are located in `../analysis/ablations/`.

## Experimental design

| Depth | Configuration | Placement |
|---|---|---|
| n1 | A1 | Gold document first |
| n1 | A2 | Gold document at position 6 |
| n1 | A3 | Gold document last |
| n2, n3, n4 | B1 | Gold block centred, chain order |
| n2, n3, n4 | B2 | Gold block centred, reversed order |
| n2, n3, n4 | B3 | Gold documents shuffled with distractors |

The input is the output of `data_generation/context_formation/construct_contexts.py`. For each complete four-hop chain it contains 12 context records, each with 11 documents, a hop question, gold answer, and gold positions. The same contexts are used for every model.

The primary prompt instructs each model to answer in 1–3 words using **only** the provided documents. The prompt text is preserved from the archived `gemini.ipynb` generation script. Decoding is greedy (`temperature=0`, `top_p=1`) with a default allowance of 50 output tokens, matching the archived single-hop and multi-hop generator notebooks. For the local Transformers engine, `do_sample=False` is used. Provider-specific model availability and parameter support should be confirmed before a run.

## Installation

Run from the repository root:

```bash
python -m pip install -r evaluation/requirements.txt
```

For local model inference through Transformers, install compatible versions of `torch`, `transformers`, and `accelerate` for the target GPU. Set provider tokens in your shell environment (`DEEPINFRA_TOKEN`, `GOOGLE_API_KEY`, or other names referenced in the model configuration). Do not commit these values.

Copy `evaluation/model_config.example.json` to a private configuration file and select the models and provider endpoints for your experiment. The example contains representative model identifiers used in the research scripts, **not** the full 12-model experimental manifest. To run local models, add entries with `"provider": "transformers"` and `"model_id": "/path/to/model"`. For an OpenAI-compatible local vLLM server, use `"provider": "openai_compatible"`, its server URL, and the configured key environment variable.

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
