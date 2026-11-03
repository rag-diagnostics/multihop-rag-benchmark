# Dataset construction

To evaluate the already released 528-chain JSON, use [the direct evaluation quick start](../README.md#quick-start). Do not regenerate narratives, reapply entity mappings, or rerun PKS on that release.

For a **new** benchmark, follow [Build a new dataset](../docs/BUILD_DATASET.md). The executable orchestrator is:

```bash
python -m data_generation.run_pipeline --input work/selected_sources.json --workdir runs/construction --seed 42
```

It preserves stable source identifiers and full JSON text through narrative generation, entity remapping, QA, random screened distractors, gold validation, recursive multi-hop construction and strict chain filtering. PKS is a separate required final stage unless explicitly configured in the orchestrator. No fixed retention count is fabricated.

The source design is 300 randomly selected LLM-vs-LLM Diplomacy transcripts plus 300 QANTA Diplomacy samples. `select_sources.py` records a reproducible new sample. QANTA transcript exports and original calibration data remain required research inputs. The WebDiplomacy scraper is supplementary, not the canonical source partition.

## Modules

| Module | Entry point |
|---|---|
| Game simulation | [Engine and launcher instructions](llm_diplomacy/docs/INSTALL_AND_USAGE.md) |
| Message extraction | `llm_diplomacy/game_log_extraction/extract_game_messages.py` |
| Source selection | `select_sources.py` |
| Narratives | `narrative_generation/generate_narratives.py` |
| Transformations | `entity_remapping/remap_entities.py` |
| QA | `qa_generation/generate_qa.py` |
| Schema alignment | `schema_adapters/prepare_distractor_input.py` |
| Random distractors | `distractor_selection/select_random_distractors.py` |
| Gold validation | `dataset_validation/validate_gold.py` |
| Multi-hop | `multihop_construction/multi_create.py`, `filter_multihop.py` |
| Context formation | `context_formation/construct_contexts.py` (also used by direct evaluation) |

Install `requirements-generation.txt` from the repository root and the NLP resources listed in the construction guide. Use environment credentials only. JSON is preferred for full-length text; XLSX refuses cells exceeding its supported text limit. See [fixes and method boundaries](../docs/FIXES.md).
