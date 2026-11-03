# LLM-played Diplomacy: game-generation code and reproducibility notes


This subproject documents the **LLM-vs-LLM Diplomacy game-simulation stage** used as a source for part of the multi-hop RAG benchmark described in *A Framework for Diagnosing Where Multi-Hop Reasoning Breaks in RAG Generators*.

> **Licensing checkpoint:** The archived upstream engine is from [GoodStartLabs/AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy). 

## What is in this directory?

| Path | Purpose |
| --- | --- |
| `scripts/generate_model_configs.py` | Portable, deterministic reconstruction of the archived model-assignment generator; produces seven model entries per game. |
| `scripts/run_games.py` | Python batch wrapper that runs the **separate** AI_Diplomacy checkout; `--dry-run` available. |
| `scripts/check_local_endpoints.py` | Validate local OpenAI-compatible vLLM servers and model IDs. |
| `scripts/audit_games.py` | Audit local game logs and saved states; any generated CSV remains on your machine. |
| `examples/local_vllm_commands.sh` | Commented, nonexecuting examples for two GPU-backed vLLM servers. |
| `docs/INSTALL_AND_USAGE.md` | Environment setup, exact commands, output locations, and HPC notes. |
| `docs/API_AND_CREDENTIALS.md` | Dedicated API/local-endpoint and credential instructions. |
| `docs/PROMPTS.md` | Identified prompt directory, prompt selection flags, and files. |
| `docs/HISTORICAL_PROVENANCE.md` | What the **actual archived scripts** did; differences from this portable wrapper. |
| `docs/UPSTREAM_AND_LICENSING.md` | Third-party attribution and licensing checkpoint. |
| `docs/RELEASE_NOTES.md` | What remains unverified and what is missing from the paper pipeline. |
| `requirements/` | Upstream archived dependency snapshot and optional GPU-server dependency guidance. |
| `manifests/prompt_files_sha256.csv` | Hashes/paths for the archived prompt templates (no prompt bodies). |

## What the archived simulation did

The source includes local, OpenAI-compatible vLLM endpoints for two open-weight models.

Model assignments were sampled across the seven Diplomacy powers (Austria, England, France, Germany, Italy, Russia, Turkey). The upstream `lm_game.py` performed the simulation and wrote per-game logs and JSONs to `results/game_###/`. The historical scripts used **2 negotiation rounds**, and the most relevant archived batch ran with `--max_year 1903 --max_tokens 100` (a setting associated with order extraction failures in the saved logs).

The archived game runner's default `--simple_prompts true` selected the upstream `ai_diplomacy/prompts_simple` templates, not the larger `ai_diplomacy/prompts` tree. Its phase-summary option was **disabled**, so this is **not** the later Gemini 2.5 Pro narrative-conversion stage from the paper.

## Quick start: 

```bash
cd data_generation/llm_diplomacy
python scripts/generate_model_configs.py --games 1 --seed 42 --output /tmp/diplomacy_one_game.txt
python scripts/run_games.py \
  --engine-dir /path/to/AI_Diplomacy \
  --config /tmp/diplomacy_one_game.txt \
  --output-dir /tmp/diplomacy_simulations \
  --dry-run
```
