
<h1 align="center">Multi-Hop RAG Diagnostics</h1>

<p align="center">
  <strong>A framework for diagnosing where multi-hop reasoning breaks in RAG generators</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> &nbsp;·&nbsp;
  <a href="#dataset-download">Dataset</a> &nbsp;·&nbsp;
  <a href="#how-it-works">Method</a> &nbsp;·&nbsp;
  <a href="#evidence-controls">Evidence controls</a> &nbsp;·&nbsp;
  <a href="#what-the-paper-finds">Findings</a> &nbsp;·&nbsp;
  <a href="#quick-start">Quick start</a>
</p>

<p align="center">
  <strong>600</strong> source narratives &nbsp;|&nbsp;
  <strong>528</strong> retained instances &nbsp;|&nbsp;
  <strong>1–4</strong> hops &nbsp;|&nbsp;
  <strong>11</strong> documents per context &nbsp;|&nbsp;
  <strong>12</strong> evaluated models
</p>


---

## Overview

This repository provides the dataset-construction, parametric knowledge filtering, and generator-evaluation code associated with *A Framework for Diagnosing Where Multi-Hop Reasoning Breaks in RAG Generators*.

### Benchmark at a glance

| Property | Design |
| :-- | :-- |
| Source material | 600 Diplomacy-derived source samples: 300 LLM-vs-LLM game samples and 300 from the QANTA Diplomacy corpus |
| Retained after PKS filtering | **528 question chains**, each with 1- through 4-hop versions |
| Reasoning complexity | 1, 2, 3, and 4 hops |
| Context budget | Exactly **11 documents** per question |
| Evidence / distractors | 1-4 gold documents plus 10-7 screened distractors |
| Single-hop conditions | A1 (first), A2 (middle), A3 (last) |
| Multi-hop conditions | B1 (canonical), B2 (reversed), B3 (randomised) |
| Expected answers | Short, document-grounded answers of 1-3 words |
| Scoring | Deterministic generation, exact match, semantic equivalence, and diagnostic metrics |

## Dataset download

The **PKS-filtered benchmark (528 question chains)** is available as a downloadable JSON file:

**[Download the 528-instance benchmark from Google Drive](https://drive.google.com/file/d/1GmHj2K5SCaV0-a8xFjaj9UprHv6hbmXa/view?usp=sharing)**

The file contains each chain's `original_question`, gold `answer`, reasoning `decomposition`, and the `hops` (`n1`-`n4`) with gold documents. The benchmark is **already generated and PKS-filtered**. You can evaluate it directly without rerunning narrative generation, multi-hop construction, or PKS calibration.

Download the JSON and save it as `data_generation/work/benchmark_528.json`. The [`prepare_released_dataset.py`](data_generation/context_formation/prepare_released_dataset.py) adapter then converts the stored document pools into the evaluator's fixed 11-document contexts. See [Quick start](#quick-start) for the tested interface and commands. The JSON is hosted separately from the source-code repository.

## How it works

<p align="center">
  <img src="assets/readme/framework.svg" alt="Diagram showing source collection, narrative transformation, entity remapping, PKS filtering, controlled multi-hop composition, and generator evaluation." width="100%" />
</p>

The source samples are turned into historical-style narratives; named entities and temporal markers are remapped to make surface-level memorisation less useful. Short, span-supported questions are expanded into multi-hop chains by replacing bridge entities with descriptions that must be resolved from additional documents. A calibrated **Parametric Knowledge Score (PKS)** screens out instances with unusually strong question-answer associations under a reference model.

Finally, each retained question is placed inside a **fixed 11-document context**, with controlled gold-evidence placement and screened distractors, for generator-only evaluation.

<details>
<summary><strong>Learn more: source generation and PKS filtering</strong></summary>

### Source material and transformation

- **300 LLM-vs-LLM Diplomacy game samples**, generated using the [GoodStartLabs AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy) environment.
- **300 QANTA Diplomacy corpus samples**, based on the [Diplomacy dataset](https://sites.google.com/view/qanta/projects/diplomacy) (Peskov et al., 2020).

Gameplay conversations are transformed into historical-style narratives, followed by stochastic entity and temporal remapping, short question-answer generation, answer-span validation, distractor screening, and recursive multi-hop construction. The `data_generation/source_collection/webdiplomacy/` scripts are supplementary collection utilities; the benchmark's QANTA-source partition is drawn from QANTA Diplomacy, not from the WebDiplomacy scraper.

See [data generation](data_generation/README.md) for input formats, dependencies, and stage-specific commands.

### Parametric knowledge filtering

The framework computes a no-context, PMI-inspired association score:

$$
\mathrm{PKS}(q,a) = L_q + L_a - L_{aq}
$$

Here, each $L$ is a summed token-level negative log-likelihood under a fixed reference model. The score is calibrated using three distributions: **Known & Plausible**, **Unknown & Plausible**, and **Unknown & Implausible**. The nearest class is determined by standardised distance; the **Known & Plausible** category is excluded.

In the paper's main run, 600 candidates were filtered to **528 retained instances**, with Qwen2.5-7B-Instruct as the primary PKS reference model. This is a **screening heuristic for parametric answerability**, not a claim that training-data membership can be detected.

Three calibration QA JSON files are required to calculate the reference distributions. See [parametric filtering](parametric_filtering/README.md).

</details>

## Evidence controls

<p align="center">
  <img src="assets/readme/controls.svg" alt="Six rows show A1-A3 single-hop gold-document placements and B1-B3 four-hop evidence ordering under eleven documents." width="100%" />
</p>

Each evaluation context contains exactly **11 documents**. This holds the document budget constant while increasing hop depth and varying gold-evidence position or ordering. The diagram above is schematic, **B3 shows one example** of randomised placement.

| Hop depth | Gold documents | Distractors | Controlled conditions |
| :--: | :--: | :--: | :-- |
| 1 | 1 | 10 | A1: first, A2: middle, A3: last |
| 2 | 2 | 9 | B1: canonical, B2: reversed, B3: randomised |
| 3 | 3 | 8 | B1: canonical, B2: reversed, B3: randomised |
| 4 | 4 | 7 | B1: canonical, B2: reversed, B3: randomised |

The context-formation code records document order, gold evidence positions, hop depth, configuration, and intermediate bridge entities. A complete 4-hop chain therefore yields **12 controlled evaluation contexts** (4 depths x 3 configurations).

## What the paper finds

The experiments test **12 models** under a constant document budget. The headline result is that **answer accuracy declines as required reasoning hops increase**, even when all necessary gold evidence has been supplied.

| Diagnostic question | Finding in the paper |
| :-- | :-- |
| **Does deeper reasoning matter?** | Accuracy decreases from hop 1 to hop 4 across all 12 evaluated models. |
| **Does evidence placement matter?** | Many models exhibit single-hop sensitivity to middle-positioned evidence. |
| **Does multi-hop ordering matter?** | Randomised evidence placement is often numerically stronger, but no formation comparison survives Holm correction. |
| **Where do incorrect answers come from?** | Among single-hop failures, 97.7% are classified as *distracted* and 2.3% as *fabricated* (Supplementary Material, Table 2). |
| **Where does a chain stop?** | The retained intermediate-entity trace identifies cases in which models output a bridge entity rather than the requested final answer. |

See the paper's **Figure 2** for the model-by-hop accuracy heatmap and [Supplementary Material](Supplementary%20Material.pdf) for additional analyses and validation.

## Evaluation and diagnostic metrics

Models answer using only the supplied documents and are instructed to return **1-3 words**. Normalised exact match is checked first, non-exact predictions are then evaluated for factual semantic equivalence by a judge model (Gemini 2.5 Pro).

| Metric | What it measures |
| :-- | :-- |
| **Accuracy (EM + judge)** | Gold-answer correctness after deterministic and semantic checks |
| **CMD** | Compliance with the 1-3 word response constraint |
| **CONFAB** | Off-task, unusable, or refusal/non-answer outputs |
| **EAR** | Correct attribution of gold supporting evidence in a separately prompted citation evaluation |
| **CCS** | Correctness at every depth of the same reasoning chain, including shallower hops |
| **DISTRACTED** | Incorrect answer appearing in a distractor document |
| **FABRICATED** | Incorrect answer absent from the provided context under the paper's string-matching procedure |

Additional error analysis checks whether an incorrect final output matches an **intermediate bridge entity** from the decomposition trace.

## Quick start

**Requirements:** Python 3.10+, the downloadable benchmark JSON, access to your chosen model(s), and sufficient memory and model context-window capacity. Python dependencies for rebuilding the dataset are described in the [data-generation](data_generation/README.md) and [parametric-filtering](parametric_filtering/README.md) guides. GPU-compatible PyTorch/Transformers builds may be needed for local inference.

```bash
# Clone and set up an environment
git clone https://github.com/rag-diagnostics/multihop-rag-benchmark.git
cd multihop-rag-benchmark
python -m venv .venv
source .venv/bin/activate

# Install inference and scoring dependencies
python -m pip install -r evaluation/requirements.txt
mkdir -p data_generation/work evaluation/work
```

API-backed steps read credentials from environment variables (for example, `DEEPINFRA_TOKEN` and `GOOGLE_API_KEY`). Keep secrets outside the repository; do not commit `.env` files or service tokens.

### Run the released benchmark

**1. Download and validate the 528-chain JSON.** Download the [benchmark from Google Drive](https://drive.google.com/file/d/1GmHj2K5SCaV0-a8xFjaj9UprHv6hbmXa/view?usp=sharing) and save it to `data_generation/work/benchmark_528.json`. Then verify the full count and context schema:

```bash
python data_generation/context_formation/prepare_released_dataset.py \
  --input data_generation/work/benchmark_528.json \
  --distractor-family irrelevant \
  --expected-chains 528 \
  --check-only
```

The adapter does **not** edit the downloaded file, generate new questions, or rerun PKS filtering.

**2. Start with a two-chain trial.** Convert just two chains (24 fixed-context questions) and check the evaluation setup without calling any model:

```bash
python data_generation/context_formation/prepare_released_dataset.py \
  --input data_generation/work/benchmark_528.json \
  --output data_generation/work/contexts_trial.json \
  --distractor-family irrelevant --limit 2 --seed 42

python -m evaluation.run_inference \
  --input data_generation/work/contexts_trial.json \
  --models evaluation/model_config.example.json \
  --output evaluation/work/trial.jsonl \
  --run-tag trial2_irrelevant --max-tokens 15 --dry-run
```

Remove `--dry-run` to obtain actual model responses. The example model configuration is a **starting point**, not the exact 12-model roster from the paper. Set the model IDs and providers for the experiment being reproduced. The 15-token generation cap matches Supplementary Material, Section K.6.

**3. Prepare the full 528-chain evaluation.** Each chain yields 12 contexts (three configurations for each of four hop depths): **6,336 contexts per distractor family**. Because the documents can be long, generate separate context shards instead of one large JSON array:

```bash
python data_generation/context_formation/prepare_released_dataset.py \
  --input data_generation/work/benchmark_528.json \
  --shard-dir data_generation/work/context_shards_irrelevant \
  --chains-per-shard 4 \
  --distractor-family irrelevant --expected-chains 528 --seed 42
```

This creates 132 files named `contexts_0000.json`, `contexts_0001.json`, etc. The shard directory must not already exist. Sharding does not truncate documents or change their positions; verify that each prepared prompt fits the input limits of the model you run.

**4. Generate answers, then judge and aggregate.** On a Unix shell, run the existing inference module once per context shard:

```bash
mkdir -p evaluation/work/shards_irrelevant
for shard in data_generation/work/context_shards_irrelevant/contexts_*.json; do
  stem=$(basename "$shard" .json)
  python -m evaluation.run_inference \
    --input "$shard" \
    --models evaluation/model_config.example.json \
    --output "evaluation/work/shards_irrelevant/${stem}.jsonl" \
    --run-tag released528_irrelevant --max-tokens 15
done

cat evaluation/work/shards_irrelevant/contexts_*.jsonl \
  > evaluation/work/generations_irrelevant.jsonl

python -m evaluation.scoring.judge_answers \
  --input evaluation/work/generations_irrelevant.jsonl \
  --output evaluation/work/judged_irrelevant.jsonl \
  --judge-model gemini-2.5-pro

python -m evaluation.scoring.aggregate \
  --input evaluation/work/judged_irrelevant.jsonl \
  --output-dir evaluation/work/summary_irrelevant
```

Inference and semantic judging can incur API costs. The judging and aggregation steps currently read their input JSONL files into memory; ensure adequate RAM and disk space. Some archived experiments use model-specific distractor-length matching, which the schema adapter does not reproduce. To match a published run, record the distractor family, original document-length treatment, model revision, random seed, prompt and decoding settings. See the [evaluation guide](evaluation/README.md) for details.

<details>
<summary><strong>Rebuild from source material (separate from evaluating the release)</strong></summary>

The repository also provides the dataset-construction and PKS-filtering code. Those stages require the original source-game samples, the three calibration QA JSON datasets, and the relevant model access. Follow the [data-generation guide](data_generation/README.md) to construct new multi-hop QA data, then the [PKS guide](parametric_filtering/README.md) to calibrate and filter it. For filtered inputs containing `context_distractors` fields, use `data_generation/context_formation/construct_contexts.py` instead of the released-dataset adapter.

</details>

For provider support, command-line options, and output schemas, see the [evaluation guide](evaluation/README.md).

## Repository map

| Module | Purpose | Documentation |
| :-- | :-- | :-- |
| `data_generation/` | Diplomacy generation, narrative transformations, QA, distractor selection, and context construction | [Guide](data_generation/README.md) |
| `parametric_filtering/` | PKS calibration, three-class filtering, and cross-model checks | [Guide](parametric_filtering/README.md) |
| `evaluation/` | Generator inference, equivalence judging, and diagnostic scoring | [Guide](evaluation/README.md) |

<details>
<summary><strong>Show the complete repository structure</strong></summary>

```text
multihop-rag-benchmark/
|-- data_generation/
|   |-- llm_diplomacy/              # LLM game simulation and message extraction
|   |-- source_collection/          # Source-collection utilities
|   |-- narrative_generation/       # Game messages to narratives
|   |-- entity_remapping/           # Entity and temporal transformations
|   |-- qa_generation/              # Base QA generation
|   |-- schema_adapters/            # Input/output schema alignment
|   |-- distractor_selection/       # Answer-screened distractors
|   |-- dataset_validation/         # Gold evidence and answer checks
|   |-- multihop_construction/      # Recursive 1-4 hop chains
|   `-- context_formation/          # Controlled A/B contexts and released JSON adapter
|-- parametric_filtering/
|   |-- baseline_calibration/       # PKS baseline scores
|   |-- pks_filtering/              # Question-level screening
|   |-- validation/                 # Reference-model checks
|   `-- pks_core.py                 # Shared score/classification logic
|-- evaluation/
|   |-- run_inference.py            # Inference across fixed contexts
|   |-- providers.py                # API and local model backends
|   |-- prompting.py                # Grounded QA prompts
|   |-- contexts.py                 # Context loading and checks
|   `-- scoring/
|       |-- matching.py             # Exact matching
|       |-- judge_answers.py        # Semantic evaluation
|       `-- aggregate.py            # Aggregation and diagnostics
|-- assets/readme/                  # Visual identity used in this README
|-- Supplementary Material.pdf
`-- README.md
```

</details>

## Supplementary material

[**Supplementary Material (PDF)**](Supplementary%20Material.pdf) provides worked multi-hop examples, dataset statistics, PKS calibration and robustness checks, full prompts and model settings, human validation, statistical analyses, live-retrieval validation, chain completion, and cross-domain replication. It also documents supplementary experiments that are separate from the core dataset-generation and evaluation code.


<p align="center"><sub>Built for controlled evidence-grounded reasoning research, not retriever benchmarking.</sub></p>
