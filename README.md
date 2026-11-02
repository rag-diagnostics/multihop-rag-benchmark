# A Framework for Diagnosing Where Multi-Hop Reasoning Breaks in RAG Generators

**A controlled benchmark for evaluating evidence-grounded reasoning in retrieval-augmented generation (RAG).**

This repository provides the dataset-construction, parametric knowledge filtering, and generator-evaluation code associated with *A Framework for Diagnosing Where Multi-Hop Reasoning Breaks in RAG Generators*. The framework tests whether language models can combine evidence across multiple documents when the retrieved context is fixed. By controlling reasoning depth, document position, evidence order, and distractor content, it isolates failures in the **generator** rather than confounding them with retrieval quality.

## Benchmark at a glance

| Property | Design |
| --- | --- |
| Source material | 600 Diplomacy-derived source samples: 300 randomly selected LLM-vs-LLM games and 300 samples from the QANTA Diplomacy corpus |
| Benchmark after PKS filtering | 528 question instances |
| Reasoning complexity | 1, 2, 3, and 4 hops |
| Context budget | Exactly 11 documents per question |
| Evidence | 1-4 gold documents, complemented by 10-7 distractor documents |
| Single-hop configurations | A1 (first), A2 (middle), A3 (last) |
| Multi-hop configurations | B1 (ordered gold block), B2 (reversed gold block), B3 (shuffled evidence) |
| Expected answers | Short, document-grounded answers of 1-3 words |
| Evaluation | Deterministic model generation, exact match, semantic equivalence judging, and diagnostic metrics |

The primary experiments use **fixed document contexts**, not live retrieval. Every model receives the same documents in the same order for a given question and configuration. Distractors are randomly selected and screened to prevent them from independently revealing the gold answer.

## How the framework works

```text
300 LLM-vs-LLM Diplomacy games       300 QANTA Diplomacy samples
              |                                 |
              +-------------+-------------------+
                            |
                  Narrative generation
                            |
                   Entity remapping
                            |
                 Question-answer generation
                            |
              Distractor selection and validation
                            |
                 Multi-hop QA construction
                            |
              Parametric knowledge filtering
                            |
                Fixed 11-document contexts
               (A1-A3 for n1; B1-B3 for n2-n4)
                            |
                   Model inference
                            |
              Exact match + semantic judging
                            |
                 Diagnostic evaluation
```

The selected source samples are converted into narratives, transformed to reduce memorisation, and used to create evidence-grounded questions. Recursive bridge-entity expansion yields one- through four-hop questions. A calibrated Parametric Knowledge Score (PKS) then screens for questions that are likely to be answerable from model parameters alone. Finally, the retained questions are paired with controlled document contexts for generator evaluation.

## Repository structure

```text
multihop-rag-benchmark/
├── data_generation/
│   ├── llm_diplomacy/             # LLM game simulation and game-message extraction
│   ├── source_collection/         # Supporting source-collection utilities
│   ├── narrative_generation/      # Gameplay conversations to narratives
│   ├── entity_remapping/          # Entity and temporal transformations
│   ├── qa_generation/             # Document-grounded base QA generation
│   ├── schema_adapters/           # Workbook and downstream schema alignment
│   ├── distractor_selection/      # Random, answer-screened distractors
│   ├── dataset_validation/        # Gold evidence and answer checks
│   ├── multihop_construction/     # Recursive 1-4 hop QA chains
│   └── context_formation/         # Eleven-document A/B configurations
├── parametric_filtering/
│   ├── baseline_calibration/      # Three-class PKS reference scores
│   ├── pks_filtering/             # Question-level PKS screening
│   ├── validation/                # Reference-model and decision checks
│   └── pks_core.py                # Shared likelihood and classification logic
├── evaluation/
│   ├── run_inference.py           # Model inference over fixed contexts
│   ├── providers.py               # API and local-model backends
│   ├── prompting.py               # Grounded generation prompts
│   ├── contexts.py                # Context loading and validation
│   └── scoring/
│       ├── matching.py            # Exact-match and answer utilities
│       ├── judge_answers.py       # Semantic equivalence judging
│       └── aggregate.py           # Metrics and chain-level summaries
└── README.md
```

Detailed guides: **[Dataset generation](data_generation/README.md)** · **[Parametric filtering](parametric_filtering/README.md)** · **[Model evaluation](evaluation/README.md)**.

## Benchmark construction

### 1. Source selection and narrative generation

The source collection combines:

- **300 randomly selected LLM-vs-LLM Diplomacy games**, generated using the [GoodStartLabs AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy) environment.
- **300 gameplay samples from the [QANTA Diplomacy dataset](https://sites.google.com/view/qanta/projects/diplomacy)** (Peskov et al., 2020).

Gameplay conversations are transformed into historical-style narrative documents. The narrative texts undergo stochastic entity and temporal remapping, followed by the generation of short, span-supported question-answer pairs. The dataset-generation modules also perform random distractor selection, gold-answer validation, and recursive multi-hop question construction.

The `data_generation/source_collection/webdiplomacy/` utilities are supplementary collection tools; the benchmark's original-source partition is drawn from **QANTA Diplomacy**.

See [data_generation/README.md](data_generation/README.md) for input formats, dependencies, and the construction commands.

### 2. Parametric knowledge filtering

To reduce reliance on memorised question-answer associations, the framework computes the no-context Parametric Knowledge Score:

$$
\mathrm{PKS}(q,a)=L_q+L_a-L_{aq}
$$

where each \(L\) is a summed token-level negative log-likelihood under a fixed reference model. The score is calibrated using three question-answer distributions: **Known & Plausible**, **Unknown & Plausible**, and **Unknown & Implausible**. Each candidate is assigned to the distribution with the smallest absolute Z-score distance; instances in the Known & Plausible class are excluded.

The reported experiment filters **600 candidate instances to 528 retained instances**, using Qwen2.5-7B-Instruct as the primary PKS reference model. The scripts also provide reference-model checks.

The three calibration QA JSON files described in [parametric_filtering/README.md](parametric_filtering/README.md) are required to compute the baseline distributions.

### 3. Controlled evidence placement

Each evaluated context contains exactly **11 documents**. The number of distractors decreases as the number of gold documents increases.

| Hop depth | Gold documents | Distractors | Evaluated configurations |
| --- | ---: | ---: | --- |
| 1 | 1 | 10 | A1: first; A2: middle; A3: last |
| 2 | 2 | 9 | B1: canonical order; B2: reverse order; B3: shuffled |
| 3 | 3 | 8 | B1: canonical order; B2: reverse order; B3: shuffled |
| 4 | 4 | 7 | B1: canonical order; B2: reverse order; B3: shuffled |

The context-formation code records the ordered documents, gold evidence positions, hop depth, configuration, and decomposition metadata. A complete four-hop chain yields **12 controlled evaluation contexts**.

### 4. Generator evaluation

Models are instructed to answer using only the supplied documents, with a response of one to three words. Predictions are scored using a two-stage procedure:

1. **Normalised exact match** against the gold answer and its accepted variants.
2. **Semantic equivalence judging** for predictions that do not match exactly, using Gemini 2.5 Pro.

The evaluation package supports API-based models, OpenAI-compatible endpoints, and local Transformers inference. It produces model-level and condition-level results, including:

- **Answer accuracy** across hop depths and evidence configurations.
- **CMD (Command Following Score):** compliance with the one-to-three-word response requirement.
- **CONFAB (Confabulation/Non-Answer Rate):** off-task, unusable, and refusal-type responses.
- **CCS (Chain Completion Score):** cumulative correctness across the same multi-hop chain.
- **DISTRACTED / FABRICATED:** distinction between incorrect answers found in distractor documents and answers unsupported by those documents.
- **EAR (Evidence Attribution Rate):** document-citation correctness in a separately prompted citation evaluation.

See [evaluation/README.md](evaluation/README.md) for model configuration, judging, and scoring details.

## Installation

Use **Python 3.10+**. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -r parametric_filtering/requirements.txt
python -m pip install -r evaluation/requirements.txt
```

Dataset-generation modules have additional stage-specific dependencies; install them as described in [data_generation/README.md](data_generation/README.md) and its linked guides. For local open-weight model inference, install a PyTorch/Transformers build compatible with the available GPU.

API-backed steps read credentials from **environment variables** (for example `DEEPINFRA_TOKEN` and `GOOGLE_API_KEY`). Store credentials outside the repository and do not commit `.env` files or model-service tokens.

## Running the evaluation workflow

The commands below start from a completed multi-hop QA dataset (`data_generation/work/filtered_multihop.json`). Run them from the repository root. The input corpus, calibration QA files, model checkpoints, and API access must be provided separately.

**1. Compute the PKS calibration distributions.**

```bash
python parametric_filtering/baseline_calibration/para11.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-dir calibration_data \
  --output-dir parametric_filtering/outputs/calibration
```

**2. Filter the multi-hop question set.**

```bash
python parametric_filtering/pks_filtering/00_para14.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data_generation/work/filtered_multihop.json \
  --calibration-dir parametric_filtering/outputs/calibration \
  --output-dir parametric_filtering/outputs/filtering
```

The filter writes retained questions to `parametric_filtering/outputs/filtering/000_abl_filtered_unknowns.json`. Inspect `pks_filter_summary.json` for class counts and any failed scores.

**3. Freeze the 11-document contexts.**

```bash
python data_generation/context_formation/construct_contexts.py \
  --input parametric_filtering/outputs/filtering/000_abl_filtered_unknowns.json \
  --output data_generation/work/contexts_11doc.json \
  --seed 42
```

**4. Validate the model configuration and execute generation.**

```bash
python -m evaluation.run_inference \
  --input data_generation/work/contexts_11doc.json \
  --models evaluation/model_config.example.json \
  --output evaluation/work/generations.jsonl \
  --dry-run
```


**5. Judge non-exact answers and aggregate metrics.**

```bash
python -m evaluation.scoring.judge_answers \
  --input evaluation/work/generations.jsonl \
  --output evaluation/work/judged.jsonl \
  --judge-model gemini-2.5-pro

python -m evaluation.scoring.aggregate \
  --input evaluation/work/judged.jsonl \
  --output-dir evaluation/work/summary
```

## Supplementary Material

The [Supplementary Material (PDF)](./Supplementary%20Material.pdf) provides additional methodological details, experimental results, and validation evidence accompanying the paper.

It includes dataset statistics, worked examples of multi-hop question construction, human validation protocols and agreement measures, Parametric Knowledge Score (PKS) calibration and robustness checks, and the prompts and model configurations used throughout the framework.

Further experiments examine evidence positioning, context sensitivity, error attribution, chain completion, retrieval validation, statistical significance, and cross-domain generalisation. These materials complement the dataset-generation, parametric-filtering, and evaluation implementations provided in this repository.


## Citation and attribution

**Paper:** *A Framework for Diagnosing Where Multi-Hop Reasoning Breaks in RAG Generators*  
**Authors:** Syed Mahbubul Huq, Chris Child, Tillman Weyde, and Pranava Madhyastha  
**Affiliations:** City St George's, University of London; The Alan Turing Institute
