# Installation and usage

This document combines requirements visible in the archived source with the upstream [GoodStartLabs README](https://github.com/GoodStartLabs/AI_Diplomacy#readme). It has **not** been validated by executing a GPU simulation in this environment. Review the [licensing checkpoint](UPSTREAM_AND_LICENSING.md) first.

## 1. Requirements

- A GPU environment capable of serving **two** approximately 7B/8B instruction-tuned models (original jobs requested two 80-GB A100 GPUs). Exact memory needs depend on serving configuration and model backend.
- Python **3.13 or later** for the archived AI_Diplomacy application, per its `pyproject.toml` (`requires-python = ">=3.13"`). A different compatible environment may be needed for vLLM depending on your installed vLLM/PyTorch versions.
- Git, [uv](https://docs.astral.sh/uv/) for the upstream project, and a CUDA-compatible [vLLM](https://docs.vllm.ai/) installation for local model serving.
- Model weights available from Hugging Face, subject to their respective model licences and access conditions. `HF_TOKEN` may be required for authenticated downloads.
- No paid OpenAI API is required for the **historical local vLLM simulation**. See [API_AND_CREDENTIALS.md](API_AND_CREDENTIALS.md).

The original upstream dependency declaration is represented by `requirements/upstream_requirements_archive.txt`. It is reference information **not** a promise that these packages alone reproduce the exact old environment. The upstream archive also contained `uv.lock` (not bundled here). Prefer the engine's own `uv sync` for its application dependencies.

## 2. Acquire the upstream engine separately

```bash
git clone https://github.com/GoodStartLabs/AI_Diplomacy.git
cd AI_Diplomacy
# Archived Git HEAD: 287d845d4ccc8ae67d47ab5131199dfbd5aee532
# If it is available and authorised for use, consider checking out that revision.
uv sync
```

**Do not blindly copy the large `AI_Diplomacy` directory from the old ZIP:** it includes `.venv`, `.git`, game results, and third-party code. The project's `.env.example` lists multiple services that were optional, not necessarily used in this experiment.

## 3. Serve the models on separate GPUs

In separate terminal sessions / GPU allocations with vLLM installed:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 --port 8000 --max-model-len 8192
```

```bash
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8001 --max-model-len 4096
```

These are *portable example commands*, not exact historical commands. The original scripts used `python -m vllm.entrypoints.openai.api_server`, specific HPC paths, GPU allocation and cache flags; see [HISTORICAL_PROVENANCE.md](HISTORICAL_PROVENANCE.md). The modern `vllm serve` syntax depends on your installed version.

**Do not expose model servers to the public internet without access control.** Keep them on `127.0.0.1` when possible.

Once both servers are ready:

```bash
cd /path/to/your/benchmark-repo/data_generation/llm_diplomacy
python scripts/check_local_endpoints.py
```

The `id` returned by `/v1/models` must match the model ID sent by `lm_game.py`. If a server advertises an alias, pass that exact ID to the config generator using `--llama-id` or `--qwen-id`.

## 4. Set the local-client environment variable

The archived `ai_diplomacy/clients.py` checks for an `OPENAI_API_KEY` even when using a local vLLM base URL. For unauthenticated localhost servers, set a **dummy** value:

```bash
export OPENAI_API_KEY=local-vllm-placeholder
```

This is **not** an OpenAI credential and does not cause OpenAI billing when the model spec directs requests to localhost. Never put a real token in a `model_configs_500.txt` line; the upstream parser supports inline `#key` but that may leak into logs/process lists.

## 5. Generate per-game model assignments

```bash
python scripts/generate_model_configs.py \
  --games 1 --seed 42 --output /tmp/diplomacy_one_game.txt
```

There are seven comma-separated model specs per line, in the order Austria, England, France, Germany, Italy, Russia, Turkey. By default each power is assigned Llama or Qwen with 50/50 probability. `--seed` makes new experiment configurations reproducible; the original `generate_configs.py` **did not set a seed**.

For larger exploratory runs, choose `--games 500` and a new output path. This creates only the configuration file, not 500 games.

## 6. Verify the commands before actually running

```bash
python scripts/run_games.py \
  --engine-dir /path/to/AI_Diplomacy \
  --config /tmp/diplomacy_one_game.txt \
  --output-dir /tmp/diplomacy_simulations \
  --limit 1 --max-year 1903 --negotiation-rounds 2 \
  --dry-run
```

If the command looks correct, **and the licence allows your intended use**, remove `--dry-run` to execute it. The script prevents accidental overwrites by skipping nonempty output directories.

By default it uses the upstream **simple prompt set** and disables phase summaries and unformatted/formatter prompts, matching the archived batch behavior. To use another prompt folder:

```bash
python scripts/run_games.py \
  --engine-dir /path/to/AI_Diplomacy \
  --config /tmp/diplomacy_one_game.txt \
  --prompts-dir /path/to/your/prompts \
  --output-dir /tmp/diplomacy_simulations --dry-run
```

### Outputs (local only)

Each game normally produces a directory like:

```text
/tmp/diplomacy_simulations/game_000/
  lmvsgame.json
  llm_responses.csv
  general_game.log
  overview.jsonl
```

Some options and revisions may also create additional files. These outputs are ignored by the suggested `.gitignore` rules; this package does not publish them.

### Offline audit of prior runs

```bash
python scripts/audit_games.py \
  --results-dir /path/to/AI_Diplomacy/results \
  --output /tmp/diplomacy_audit.csv
```

This reads saved files only; it does not contact models or modify the games.

## 7. HPC and SLURM

The archive contains three different batch-launcher variants. The most relevant archived `AI_Diplomacy/run_everything.slurm` requested **2 A100 80GB GPUs**, bound Llama to GPU 0 and Qwen to GPU 1, waited for startup, then iterated games 66 to 499 with `--max_year 1903 --num_negotiation_rounds 2 --max_tokens 100`. Its project-specific paths, Hugging Face authentication command, and host bindings should not be copied unmodified into a public repository. Use your cluster's own job template and paths.

For each archived failure and environment distinction, see [HISTORICAL_PROVENANCE.md](HISTORICAL_PROVENANCE.md).
