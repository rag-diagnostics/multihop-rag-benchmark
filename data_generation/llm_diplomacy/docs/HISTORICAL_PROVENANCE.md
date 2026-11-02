# Reconstruction of the archived LLM Diplomacy simulation

**Source examined:** The user-supplied `llm_dip.zip` archive, not an independently reproduced run. We distinguish original scripts from the cleaned example scripts in this package. No game data is published here.

## Files and their roles

| Archived file | Role | Observations |
| --- | --- | --- |
| `llm_dip/generate_configs.py` | Samples seven Llama/Qwen assignments for 500 games | Uses `random.choices()` without an explicit seed; contains hard-coded cluster hostnames and a malformed Qwen URL containing a space. |
| `llm_dip/run_all_games.slurm` | Sequentially launches configurations | Requests one GPU, loads Conda `rgb`, loops all 500 lines using `--max_year 1915 --num_negotiation_rounds 2`; assumes servers already available. |
| `llm_dip/run_everything.slurm` | Starts both vLLM servers and runs games | Requests two A100 GPUs; code actually loops `0..9`, although comments mention 500; Qwen model string contains a space and may fail as written. |
| `llm_dip/AI_Diplomacy/run_everything.slurm` | More complete all-in-one batch | Requests two A100 80GB GPUs; binds models to GPUs 0/1, configures ports 8000/8001, and loops **66..499**, with `--max_year 1903 --num_negotiation_rounds 2 --max_tokens 100`. |
| `llm_dip/AI_Diplomacy/lm_game.py` | Actual simulation entry point | Defaults to `simple_prompts=true`, `generate_phase_summaries=false`, and `use_unformatted_prompts=false`. Writes saved games, responses and logs. |
| `llm_dip/AI_Diplomacy/ai_diplomacy/` | Upstream LLM agents and prompt logic | GoodStartLabs code, not the benchmark authors' new framework. |
| `llm_dip/AI_Diplomacy/results/` | Existing saved simulations | The archive includes 472 `game_*` subdirectories with saved JSONs; excluded from this code-only package. |
| `llm_dip/AI_Diplomacy/.git/refs/heads/main` | Checkout provenance | SHA `287d845d4ccc8ae67d47ab5131199dfbd5aee532`; the archived worktree may not match this commit exactly. |

## Common model settings in the relevant batch

- Llama: `meta-llama/Llama-3.1-8B-Instruct`, vLLM local port `8000`, max model context length `8192`.
- Qwen: `Qwen/Qwen2.5-7B-Instruct`, vLLM local port `8001`, max model context length `4096`.
- GPU: 2 x A100 80GB, one model server per GPU, on the original cluster.
- Game: `--max_year 1903`, `--num_negotiation_rounds 2`, `--max_tokens 100`.
- Prompt selection: `ai_diplomacy/prompts_simple/`, default, with no explicit custom path.
- Phase summaries: disabled, so no evidence of the later full-narrative Gemini transformation from this job.

The archived model configurations contain endpoint hostnames or `#not_needed` inline placeholder values in different copies. **Do not publish those files as-is**; the clean generator substitutes localhost URLs and environment-based authentication.

## Known anomalies and limits of the evidence

1. A prior audit of the 472 archived games found 19,798 logged order-generation attempts marked `Failure: No moves extracted`. This likely affected gameplay, with the simulator's fallback moves invoked. The logs prove extraction failures, not necessarily why they occurred. The very short `--max_tokens 100` is one plausible contributing factor, but not a demonstrated cause.
2. Some stored orders are non-HOLD, but logged parsing failures mean those cannot be reliably attributed to successful LLM order parsing without further investigation.
3. The two `model_configs_500.txt` files in the archive do not fully describe the saved game assignments. Each saved game is a stronger provenance source for its own model IDs.
4. Several SLURM variants exist. The archived folders alone do **not** prove the precise execution order of every game or which run was ultimately selected into the 300 source narratives.
5. The new `generate_model_configs.py` adds a seed; it provides *repeatable new assignments*, **not** byte-for-byte recreation of old unseeded runs.
6. The new runner defaults to `--max-tokens 2048` for an exploratory run; **the old selected batch used 100**. This change must be stated in any newly reported experiments.
7. Neither archive identifies the script and Gemini 2.5 Pro prompts that converted saved games into 300 standalone narrative documents. That stage is missing.

## Correct research attribution

The reusable contribution claimed in the ECIR paper is the controlled RAG benchmark and its downstream construction/evaluation protocol. The LLM game simulation engine and most game-agent prompts are based on **GoodStartLabs/AI_Diplomacy** (Duffy et al.) and should be clearly credited. This subproject documents a source-data workflow and does not imply that the benchmark authors wrote the upstream engine.
