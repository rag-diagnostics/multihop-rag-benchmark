# Dataset construction

This directory contains the dataset-construction components for **A Framework for Diagnosing Where Multi-Hop Reasoning Breaks in RAG Generators**. The pipeline combines two Diplomacy sources, generates narrative documents and base question–answer pairs, validates the evidence, composes multi-hop questions and constructs controlled 11-document contexts.

## Source composition

The source pool contains **600 selected game samples**:

- **300 randomly selected LLM-versus-LLM game samples**, generated with the [GoodStartLabs AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy) environment.
- **300 samples from the [QANTA Diplomacy dataset](https://sites.google.com/view/qanta/projects/diplomacy)**, associated with Peskov et al. (2020).

The selected samples are converted into narrative documents before entity transformation and question–answer generation. Record selected game identifiers and any sampling seed alongside each dataset build to preserve a reproducible source manifest.


## Construction stages

| Stage | Directory | Inputs → outputs |
| --- | --- | --- |
| LLM game simulation | [`llm_diplomacy/`](llm_diplomacy/README.md) | Model configurations → saved Diplomacy games |
| LLM message extraction | [`llm_diplomacy/game_log_extraction/`](llm_diplomacy/game_log_extraction/README.md) | `lmvsgame.json` → `game_<id>_messages.txt` |
| Narrative conversion | [`narrative_generation/`](narrative_generation/README.md) | Selected game messages → narrative summaries in a workbook |
| Entity remapping | [`entity_remapping/`](entity_remapping/README.md) | Narrative workbook → transformed `Summary` column |
| Base QA generation | [`qa_generation/`](qa_generation/README.md) | Transformed narratives → `question` and `answer` columns |
| Workbook interface | [`schema_adapters/`](schema_adapters/README.md) | `Summary` → `Document` column for downstream scripts |
| Random distractors | [`distractor_selection/`](distractor_selection/README.md) | QA workbook → 10 random, answer-screened distractors per item |
| Automated gold validation | [`dataset_validation/`](dataset_validation/README.md) | Base QA/distractor JSON → gold-validated JSON |
| Multi-hop composition | [`multihop_construction/`](multihop_construction/README.md) | Validated base QA → n=1–4 question chains |
| Evidence placement | [`context_formation/`](context_formation/README.md) | Multi-hop chains → A1–A3 and B1–B3 contexts |

**Parametric-answerability filtering (§3.4)** is a separate benchmark-curation step and should be applied to the base QA pool used for the final retained benchmark. Model evaluation and results analysis are outside this directory.

## Input and output conventions

The stage-specific scripts use file-based interfaces rather than a single monolithic command. The core fields are:

| Field | Meaning |
| --- | --- |
| `Serial` | Identifier for a base document/QA row |
| `Summary` | Narrative text consumed by the entity remapping and QA scripts |
| `question`, `answer` | Generated base question and answer |
| `Document` | Narrative text consumed by the random distractor and multi-hop scripts |
| `golden_document_text`, `golden_answer` | Gold source and answer in the distractor JSON |
| `distractors` | Randomly selected, answer-screened non-gold documents |
| `hops.n1`–`hops.n4` | Multi-hop chain data, gold documents and remaining distractors |

The entity-remapping script expects `final_dataset.xlsx` with a `Summary` column and writes `0modified_final_dataset_004.xlsx`. The QA script updates that workbook in place. The adapter in `schema_adapters/` creates the `Document` column required by distractor selection while retaining the existing `Summary` column and QA fields.

### Example: QA workbook → validated multi-hop contexts

Run from the repository root once the Gemini-generated QA workbook has been saved:

```bash
python data_generation/schema_adapters/prepare_distractor_input.py \
  --input data_generation/work/0modified_final_dataset_004.xlsx \
  --output data_generation/work/qa_with_document.xlsx

python data_generation/distractor_selection/select_random_distractors.py \
  --input-xlsx data_generation/work/qa_with_document.xlsx \
  --output data_generation/work/rag_benchmark_clean_new.json \
  --audit data_generation/work/random_distractor_audit.json \
  --count 10 --seed 42 --check-mode llm

python data_generation/dataset_validation/validate_gold.py \
  --input data_generation/work/rag_benchmark_clean_new.json \
  --output data_generation/work/rag_benchmark_gold_validated_new.json \
  --errors data_generation/work/gold_validation_errors.json

python data_generation/multihop_construction/multi_create.py \
  --input data_generation/work/rag_benchmark_gold_validated_new.json \
  --global-docs data_generation/work/qa_with_document.xlsx \
  --output data_generation/work/000_new_multihop_dataset_final.json \
  --errors data_generation/work/multihop_errors.json

python -c 'from data_generation.multihop_construction.filter_multihop import filter_json_remove_entries_with_hops_error; filter_json_remove_entries_with_hops_error("data_generation/work/000_new_multihop_dataset_final.json", "data_generation/work/filtered_multihop.json")'

python data_generation/context_formation/construct_contexts.py \
  --input data_generation/work/filtered_multihop.json \
  --output data_generation/work/contexts_11doc.json --seed 42
```

The commands above require appropriate local inputs and API credentials for LLM-assisted screening, validation and multi-hop construction. See each module's README for dependency installation, model selection, and verification. The random selection stage excludes the gold document and candidate distractors containing gold-answer variants, then checks candidate answerability with Gemini.
