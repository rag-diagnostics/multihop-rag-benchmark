# Which prompts were used?

**The archived prompts exist, and the effective default for the selected batch was the upstream `ai_diplomacy/prompts_simple/` directory.** We do not reproduce the third-party prompt contents here pending licensing clarification.

## Evidence from the archived code

1. `lm_game.py` declares `--simple_prompts` with default **true**.
2. When true and `--prompts_dir` is absent, it sets `args.prompts_dir` to `<engine>/ai_diplomacy/prompts_simple`.
3. Your archived SLURM commands supplied neither `--prompts_dir` nor `--simple_prompts false`.
4. `--use_unformatted_prompts` defaulted to **false**; the normal formatted prompts were used instead of the optional two-stage Gemini formatter path.
5. `--country_specific_prompts` defaulted to **false**, and phase-summary generation was disabled.

## Prompt families found in the archive

`ai_diplomacy/prompts_simple/` contains **29 files**, including its `unformatted/` subdirectory. The top-level files include:

- Seven nation-specific `*_system_prompt.txt` files (Austria, England, France, Germany, Italy, Russia, Turkey).
- `conversation_instructions.txt` and `negotiation_diary_prompt.txt`.
- `order_instructions_movement_phase.txt`, `order_instructions_retreat_phase.txt`, `order_instructions_adjustment_phase.txt`, and `order_diary_prompt.txt`.
- `state_update_prompt.txt`, `initial_state_prompt.txt`, `phase_result_diary_prompt.txt`, `diary_consolidation_prompt.txt`.
- `context_prompt.txt`, `planning_instructions.txt`, `few_shot_example.txt`, and `system_prompt.txt`.

**Two top-level files in this snapshot are empty:** `initial_state_prompt.txt` and `system_prompt.txt`. The upstream code may provide fallbacks or assemble prompts elsewhere. The presence of a file does not mean that every prompt was invoked in every run.

The archive also includes `ai_diplomacy/prompts/` (larger tree with variant prompt sets) and `ai_diplomacy/prompts_simple_optim_v1.0/`; these were **not selected** by default in the commands shown. The snapshot contains no separate, evidenced Gemini 2.5 Pro document-to-narrative prompt for the benchmark's later conversion step.

## Version/provenance checks

`manifests/prompt_files_sha256.csv` includes relative paths, byte lengths, and SHA-256 hashes for all archived `prompts_simple/` files. It allows a researcher with a **licensed upstream checkout** to check whether those prompt files match this snapshot without redistributing the prompt bodies.

For source retrieval and attribution, see [GoodStartLabs/AI_Diplomacy](https://github.com/GoodStartLabs/AI_Diplomacy) and [UPSTREAM_AND_LICENSING.md](UPSTREAM_AND_LICENSING.md). Do not attribute the upstream prompt text to this benchmark's authors.
