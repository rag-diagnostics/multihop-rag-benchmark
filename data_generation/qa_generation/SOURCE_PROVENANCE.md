# Current release note

[FIXES.md](../../docs/FIXES.md) documents the current I/O, validation, resume, and fail-closed changes. The following earlier provenance record describes the preceding release and must not override the current commands.

---

# Source provenance and exact scope of the Gemini adaptation

- Source archive: user-provided `16-12-2025_final paper(1).zip`, which should remain private.
- Original notebook: `dataset_creation.ipynb`, code cell 1 (0-based index; the second code cell).
- Prior public-ready package: `qa_generation_github_ready.zip`, a credential-sanitized transcription of that cell.
- This release: same core QA prompt and dataframe workflow, Gemini API adapter instead of DeepSeek API.

## Changes from the prior package

1. Python client import: `openai.OpenAI` → official `google-genai` (`google.genai`).
2. Credential retrieval: private environment variable `GEMINI_API_KEY`; no credential literal in code.
3. Model name: `deepseek-chat` → `gemini-2.5-pro`, matching the model named in the manuscript.
4. API initialization and preflight check: Gemini client and model access lookup.
5. Inference call: `client.models.generate_content` with original user prompt, original system instruction and original temperature 0.0.
6. Text checkpoint output folder renamed to `gemini_out_6_004` so historical DeepSeek checkpoints cannot be mixed with Gemini outputs.
7. Some provider-name comments and error logs adapted (exception type only, to avoid sensitive log disclosure).
8. `openai` dependency replaced with `google-genai`; notebook, README, environment template and gitignore updated.

**Preserved without changes:** The entire embedded QA prompt (including examples and the 1–2-word answer rule); the complete function bodies of `process_and_save_row()`, `generate_text_files_concurrently()` and `update_xlsx_from_text_files()`; question-type order; 10-worker parallelism; three retries and backoff; `QUESTION:`/`ANSWER:` parsing; input workbook name; Excel column names; in-place update behavior.

These identical-function checks were verified programmatically against the previous released script, and the prompt block was compared byte-for-byte.

## Research provenance distinction

This *adapted* release implements what the paper states about the model choice, but the recovered historical notebook used DeepSeek. Its presence does not prove either model produced the final experiment data. Do not claim the paper's existing dataset or reported numbers were generated with this adapted script unless that is verified independently.

Also, the original prompt limits answers to 1–2 words whereas the draft paper says 1–3 words. The prompt has intentionally **not** been changed at the user's request.
