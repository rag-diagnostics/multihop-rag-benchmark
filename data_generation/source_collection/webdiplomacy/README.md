# Original-source WebDiplomacy collection and cleaning

These scripts are reconstructed from the user-provided `code.ipynb`, `game_messages_clean.ipynb`, and `cleaning_code.ipynb`. They recover a historical **2025 finished-game web-scraping workflow**, not the GoodStartLabs LLM-game simulator or the Gemini narrative generator.

## Requirements

- Python 3.10 or later (recommended)
- `requests`, `beautifulsoup4` (for `collect_games.py`)
- Standard library only (for deduplication and cleaning)

```bash
python -m pip install -r data_generation/source_collection/webdiplomacy/requirements.txt
```

## Execute from repository root

**Check the source site's current access permissions, robots guidance, terms, and research ethics requirements before collecting any data.** The original scraper depends on historical page layout and has not been verified against the current site.

```bash
python data_generation/source_collection/webdiplomacy/collect_games.py \
  --output-dir data_generation/source_collection/webdiplomacy/work \
  --max-listing-pages 1 --max-message-pages 15

python data_generation/source_collection/webdiplomacy/deduplicate_messages.py \
  --input-dir data_generation/source_collection/webdiplomacy/work/game_messages \
  --output-dir data_generation/source_collection/webdiplomacy/work/new_text_cleaned

python data_generation/source_collection/webdiplomacy/clean_messages.py \
  --input-dir data_generation/source_collection/webdiplomacy/work/new_text_cleaned \
  --output-dir data_generation/source_collection/webdiplomacy/work/new_text_cleaned_2
```

Outputs: `work/game_metadata.csv` and `work/game_messages/game_<id>_messages.txt`; two subsequent cleaned directories retain these same filenames. Feed the final text directory into the **existing** `data_generation/narrative_generation/summary_generation.ipynb` via `DIPLOMACY_TEXT_DIR` to create the source workbook and generate narratives (that notebook uses Gemini and requires a separate Gemini API key via environment variable).

## Important historical caveats

1. `collect_games.py` filters for games finishing in **2025**, as in the original notebook, rather than the paper's referenced Peskov et al. Diplomacy corpus. **Do not silently claim this source is the cited corpus.** Establish the final input data provenance and any permission to redistribute it.
2. The historical `format_messages_by_conversation()` groups each message into every country pair containing its sender. **It does not retrieve recipient identities**, and this may cause the same message to appear in multiple conversation groups. This behavior is preserved for fidelity, but must be reviewed before claiming faithful pairwise chat reconstruction.
3. The historical deduplicator drops repeated identical nonempty lines. Some legitimate repeated utterances may be removed. The metadata cleaner removes date and separator lines. Retained as originally implemented; document these transformations if they were used for the final benchmark.
4. The original scraper used a default of 5000 listing pages; the public command defaults to one page to avoid unexpectedly extensive scraping. You can explicitly choose a page limit after checking source permissions and requirements.
5. Both source layout and chronological ordering depend on the site and available timestamps; this has not been tested on the live site.
6. No code for selecting the final 300 original-source items has been identified in these notebooks.

## API usage and credentials

The collector makes ordinary HTTP GET requests to WebDiplomacy via `requests` and requires **no LLM API key**. The cleaning stages are entirely offline. Do not commit cookies, tokens, private messages, .env files, or scraped outputs. The later Gemini narrative-generation code is documented in `data_generation/narrative_generation/README.md`.

## Code provenance

- `collect_games.py` ← `code.ipynb`: HTTP extraction, parsing, the 2025 filter, grouping, and export logic retained; output locations and command-line page limits made configurable.
- `deduplicate_messages.py` ← `game_messages_clean.ipynb`: original line-deduplication function retained; CLI arguments replace hardcoded folder path.
- `clean_messages.py` ← `cleaning_code.ipynb`: original regex cleaning and rewriting function retained; CLI arguments replace hardcoded paths.
- `llm_war.ipynb` creates model assignments but is already represented in `data_generation/llm_diplomacy/` and is not needed here.

Only code and documentation are included in this public package; not user/site messages or execution outputs.
