<h1 align="center">Multi-Hop RAG Diagnostics</h1>

<p align="center">
  <strong>A framework for diagnosing where multi-hop reasoning breaks in RAG generators</strong><br />
</p>

<p align="center">
  <a href="#overview">Overview</a> 
  <a href="#how-it-works">Method</a> 
  <a href="#evidence-controls">Evidence controls</a>
  <a href="#what-the-paper-finds">Findings</a> 
  <a href="#quick-start">Quick start</a>
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <strong>600</strong> source narratives &nbsp,&nbsp,|&nbsp,&nbsp,
  <strong>528</strong> retained instances &nbsp,&nbsp,|&nbsp,&nbsp,
  <strong>1-4</strong> hops &nbsp,&nbsp,|&nbsp,&nbsp,
  <strong>11</strong> documents per context &nbsp,&nbsp,|&nbsp,&nbsp,
  <strong>12</strong> evaluated models
</p>

---

## Overview

**Can a language model connect the right evidence across multiple documents when retrieval is no longer the problem?**

This repository provides the dataset-construction, parametric knowledge filtering, and generator-evaluation code associated with *A Framework for Diagnosing Where Multi-Hop Reasoning Breaks in RAG Generators*. The benchmark tests whether language models can combine evidence across documents when the retrieved context is **fixed**.

By controlling reasoning depth, evidence position, ordering, and distractor content, it isolates failures in the **generator** rather than confounding them with retrieval quality. **The primary benchmark is not a live-retrieval evaluation.** A separate retrieval validation is included in the [supplementary material](Supplementary%20Material.pdf).

### Benchmark at a glance

| Property | Design |
| :-- | :-- |
| Source material | 600 Diplomacy-derived source samples: 300 LLM-vs-LLM game samples and 300 from the QANTA Diplomacy corpus |
| Retained after PKS filtering | 528 question instances, as reported in the paper |
| Reasoning complexity | 1, 2, 3, and 4 hops |
| Context budget | Exactly **11 documents** per question |
| Evidence / distractors | 1-4 gold documents plus 10-7 screened distractors |
| Single-hop conditions | A1 (first), A2 (middle), A3 (last) |
| Multi-hop conditions | B1 (canonical), B2 (reversed), B3 (randomised) |
| Expected answers | Short, document-grounded answers of 1-3 words |
| Scoring | Deterministic generation, exact match, semantic equivalence, and diagnostic metrics |

## How it works

<p align="center">
  <img src="assets/readme/framework.svg" alt="Diagram showing source collection, narrative transformation, entity remapping, PKS filtering, controlled multi-hop composition, and generator evaluation." width="100%" />
</p>

The source samples are turned into historical-style narratives, named entities and temporal markers are remapped to make surface-level memorisation less useful. Short, span-supported questions are expanded into multi-hop chains by replacing bridge entities with descriptions that must be resolved from additional documents. A calibrated **Parametric Knowledge Score (PKS)** screens out instances with unusually strong question-answer associations under a reference model.

Finally, each retained question is placed inside a **fixed 11-document context**, with controlled gold-evidence placement and screened distractors, for generator-only evaluation.

<details>
<summary><strong>Learn more: source generation and PKS filtering</strong></summary>

### Source material and transformation

- **300 LLM-vs-LLM Diplomacy game samples**, generated using the [GoodStartLabs AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy) environment.
- **300 QANTA Diplomacy corpus samples**, based on the [Diplomacy dataset](https://sites.google.com/view/qanta/projects/diplomacy) (Peskov et al., 2020).

Gameplay conversations are transformed into historical-style narratives, followed by stochastic entity and temporal remapping, short question-answer generation, answer-span validation, distractor screening, and recursive multi-hop construction. The `data_generation/source_collection/webdiplomacy/` scripts are supplementary collection utilities, the original-source benchmark partition is based on QANTA Diplomacy.

See [data generation](data_generation/README.md) for input formats, dependencies, and stage-specific commands.

### Parametric knowledge filtering

The framework computes a no-context, PMI-inspired association score:

$$
\mathrm{PKS}(q,a) = L_q + L_a - L_{aq}
$$

Here, each $L$ is a summed token-level negative log-likelihood under a fixed reference model. The score is calibrated using three distributions: **Known & Plausible**, **Unknown & Plausible**, and **Unknown & Implausible**. The nearest class is determined by standardised distance, the **Known & Plausible** category is excluded.

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
| **Where do incorrect answers come from?** | The majority are *distracted* answers, approximately 2.3% are classified as *fabricated*. |
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

**Requirements:** Python 3.10+, stage-specific dependencies, model access or credentials, and input/calibration files. Dataset-generation modules have their own dependencies, consult their [guide](data_generation/README.md). GPU-compatible PyTorch/Transformers builds may be needed for local open-weight inference.

```bash
# Clone the repository
git clone https://github.com/rag-diagnostics/multihop-rag-benchmark.git
cd multihop-rag-benchmark

# Create an environment
python -m venv .venv
source .venv/bin/activate

# Install the PKS and evaluation requirements
python -m pip install -r parametric_filtering/requirements.txt
python -m pip install -r evaluation/requirements.txt
```

API-backed steps read credentials from environment variables (for example `DEEPINFRA_TOKEN` and `GOOGLE_API_KEY`). Keep secrets out of the repository, do not commit `.env` files or service tokens.

### Reproduce the evaluation pipeline

The commands below start from a constructed multi-hop dataset (`data_generation/work/filtered_multihop.json`). Provide the source corpus, calibration QA files, model checkpoints, and API access separately. Run commands from the repository root.

**1. Calibrate the PKS reference distributions**

```bash
python parametric_filtering/baseline_calibration/para11.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data-dir calibration_data \
  --output-dir parametric_filtering/outputs/calibration
```

**2. Filter the multi-hop QA set**

```bash
python parametric_filtering/pks_filtering/00_para14.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input data_generation/work/filtered_multihop.json \
  --calibration-dir parametric_filtering/outputs/calibration \
  --output-dir parametric_filtering/outputs/filtering
```

Retained instances are written to `parametric_filtering/outputs/filtering/000_abl_filtered_unknowns.json`, `pks_filter_summary.json` records filtering outcomes.

**3. Freeze the 11-document contexts**

```bash
python data_generation/context_formation/construct_contexts.py \
  --input parametric_filtering/outputs/filtering/000_abl_filtered_unknowns.json \
  --output data_generation/work/contexts_11doc.json \
  --seed 42
```

**4. Validate and run model generation**

```bash
# Validate configuration without running generation
python -m evaluation.run_inference \
  --input data_generation/work/contexts_11doc.json \
  --models evaluation/model_config.example.json \
  --output evaluation/work/generations.jsonl \
  --dry-run

# Remove --dry-run from the command above to execute inference.
```

**5. Judge non-exact answers and aggregate metrics**

```bash
python -m evaluation.scoring.judge_answers \
  --input evaluation/work/generations.jsonl \
  --output evaluation/work/judged.jsonl \
  --judge-model gemini-2.5-pro

python -m evaluation.scoring.aggregate \
  --input evaluation/work/judged.jsonl \
  --output-dir evaluation/work/summary
```

For configuration details, provider support, and output schemas, see the [evaluation guide](evaluation/README.md).

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
|   `-- context_formation/          # Controlled A/B document arrangements
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

[**Supplementary Material (PDF)**](Supplementary%20Material.pdf) provides worked multi-hop examples, full experimental details, dataset statistics, PKS calibration and robustness checks, prompts, model settings, human validation, position and context ablations, statistical analyses, live-retrieval validation, chain completion, and cross-domain replication.
When an official publication or preprint URL is available, link it here and add the canonical BibTeX entry (or a `CITATION.cff` file) for easy reuse.

---

<p align="center"><sub>Built for controlled evidence-grounded reasoning research, not retriever benchmarking.</sub></p>
