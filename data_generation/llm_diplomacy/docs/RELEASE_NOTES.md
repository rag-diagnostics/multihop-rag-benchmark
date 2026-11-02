# Code-only package: scope and follow-up checklist

## Included / verified from the supplied archive

- Original experiment's local Llama and Qwen model IDs, ports, vLLM serving method, SLURM launcher parameters and source-code entry points.
- Identified the `prompts_simple` directory as the effective default prompt set for the selected scripts.
- Built a portable seeded config generator and a small batch wrapper with a `--dry-run` mode.
- Extracted the archived upstream Python requirements for documentation.
- Documented optional APIs and required local placeholder `OPENAI_API_KEY` behavior.
- Included an offline audit utility and prompt-file SHA256 inventory.

## Not included

- Raw game JSONs, prompts and completions, game response logs, 300 transformed narratives, generated QA, or benchmark results.
- Any real `.env`, model weights, third-party engine source or third-party prompt contents.
- Full upstream `.venv`, `uv.lock`, unrelated RGB evaluation code or other RAG benchmark code.

## Unresolved before claiming full reproducibility

1. **Rights/licensing**: clarify the upstream engine's dataset-gathering restrictions, plus whether game outputs and transformed narratives may be redistributed.
2. **Exact engine revision**: determine any local modifications relative to recorded upstream Git HEAD and preserve an authorised, immutable version reference.
3. **Historical assignments**: reconcile the alternative config files with saved per-game model IDs and record the selected 300 games, if applicable.
4. **Order parsing**: investigate all `No moves extracted` records and describe their effect on the games. Do not silently substitute revised model outputs for the historical data.
5. **Game -> narrative**: locate the Gemini 2.5 Pro transformation code, prompts, model/version, sampling settings and output format. This code is **not** identified in either uploaded ZIP.
6. **Validation**: run a small authorised end-to-end smoke test with model servers and verify output consistency. The scripts here were syntax-checked and dry-run tested, but not GPU-tested.
7. **ECIR resource access**: ensure a lawful, usable benchmark resource is available for reviewers, potentially via a separate archive rather than committing large data to GitHub.

## Suggested follow-up structure in the main project repository

```text
multihop-rag-benchmark/
  README.md
  data_generation/
    llm_diplomacy/        # this package
    human_diplomacy/      # later: 300 public-log-derived narratives
    narrative_conversion/ # later: Gemini 2.5 Pro scripts, prompts, outputs schema
    benchmark_construction/ # later: remapping, QA, PKS, multi-hop chains
  evaluation/             # later: model runs and diagnostics
  docs/
```

Only add folders when their source code is found and provenance verified. Preserve distinctions between original historical methods and later, repaired implementations.
