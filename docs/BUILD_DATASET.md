# Build a new benchmark from source transcripts

This is a separate path from evaluating the released JSON. New stochastic games, narrative outputs, remapping, questions, or model-based filtering will not automatically reproduce the historical 600-to-528 sample. The original calibration QA files and historical selection manifest are research inputs, not generated substitutes.

## Environment and data contract

From the repository root:

```bash
python -m pip install -r requirements-generation.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet omw-1.4
```

Provide `GEMINI_API_KEY` or `GOOGLE_API_KEY` through the environment. Select model identifiers to which the project has access; there is no automatic model fallback. The narrative default is the recovered notebook's `gemini-2.5-flash`, QA is `gemini-2.5-pro`, validation is `gemini-2.0-flash`, and multi-hop generation is `gemini-2.5-flash`. 

## 1. Simulate games and extract messages

Acquire and configure the upstream GoodStartLabs AI_Diplomacy engine in its own environment following [the existing engine guide](../data_generation/llm_diplomacy/docs/INSTALL_AND_USAGE.md). Its code, weights, licences and provider/local-serving dependencies are separate. The benchmark repository supplies configuration/launcher/extraction code; it does not vendor the engine or model weights.

With the engine and two local serving endpoints ready:

```bash
python data_generation/llm_diplomacy/scripts/generate_model_configs.py \
  --games 2 --seed 42 --output runs/game_configs.txt
python data_generation/llm_diplomacy/scripts/run_games.py \
  --engine-dir /path/to/AI_Diplomacy --config runs/game_configs.txt \
  --output-dir runs/games --limit 2 --max-year 1903 --negotiation-rounds 2 --dry-run
```

Remove `--dry-run` to play new games. Inspect the saved logs, then:

```bash
python data_generation/llm_diplomacy/game_log_extraction/extract_game_messages.py \
  --input-dir runs/games --output-dir work/llm_messages
```

Extraction preserves UTF-8 private messages and season/year from the recorded phase, writes an extraction manifest, and refuses malformed games instead of inventing a demonstration game. GLOBAL messages remain excluded as in the original private-message extraction.

## 2. Select source transcripts

Place the chosen QANTA Diplomacy corpus message exports in `work/qanta_messages/`. Preserve the actual source identifiers and corpus attribution. The WebDiplomacy scraper is supplementary and is not a substitute for this partition.

```bash
python -m data_generation.select_sources \
  --llm-dir work/llm_messages --qanta-dir work/qanta_messages \
  --llm-count 300 --qanta-count 300 --seed 42 --output work/selected_sources.json
```

The input directories must contain the requested number of real source transcripts. `--pattern` defaults to `*.txt`. The selection manifest records filenames and SHA-256 hashes; it describes this new draw, not unrecorded historical IDs. For a small construction test, use smaller source counts totalling at least 11 distinct documents; additional sources are often needed because safe distractors and linking evidence must actually exist.

The selected JSON has `Serial`, `text`, `source`, `source_filename`, and `source_sha256`. A preselected JSON with `Serial` and `text` can also be supplied directly. The pipeline does not reparse or automatically retrieve the QANTA corpus from its website.

## 3. Run construction in order

```bash
python -m data_generation.run_pipeline \
  --input work/selected_sources.json --workdir runs/construction \
  --narrative-model gemini-2.5-flash --qa-model gemini-2.5-pro \
  --validation-model gemini-2.0-flash --multihop-model gemini-2.5-flash --seed 42
```

This runs:

`messages -> narratives -> entity remapping -> base QA -> Document alignment -> random screened distractors -> gold validation -> recursive multi-hop construction -> strict chain validation`

The stage outputs and checkpoints are explicit JSON files. No stage replaces `Summary` with the unrelated raw `text` column. Remapping runs before any QA, and refuses assembled question/decomposition records. `Summary` and `Document` are kept aligned. The exact recovered remapping routine is retained, including its token-wise limitations; the seed/entity map is saved.

The original recovered QA prompt still requests **1-2-word answers**, not the manuscript's 1-3-word limit. That prompt has not been silently edited. The validator accepts 1-3 words, so all accepted 1-2-word QA outputs fit it. Changing the prompt is a new experimental choice.

Random distractors are sampled without replacement, excluding the gold document, duplicate texts, literal gold-answer variants, and model-answerable candidates. Operational errors are not treated as safe. A lack of ten safe candidates is a real data limitation: the pipeline never pads with duplicates or unverified generated text.

A new hop must replace an entity actually in the previous question, use its descriptive clue, preserve the gold answer under full evidence, require the new document, and require each earlier gold document under leave-one-document-out model checks. These checks are stricter than the recovered script. They reduce invalid output but remain model-based checks; human review is still needed. Invalid chains are saved to a rejection report, not published as successful four-hop chains.

Narrative and QA checkpoints are keyed by complete source/protocol identities. Repeated identical runs reuse completed outputs; changed input/model/prompt settings require a new directory. Source/model errors stop the pipeline. Do not remove checkpoints and overwrite old research data to force a resume.

## 4. Calibrate and apply PKS

Without `--pks-model`, construction stops at `multihop_validated.json` and records `pks_applied: false`. It does **not** label that file as a final PKS-filtered benchmark.

Supply the three original research calibration QA arrays (`question`, `answer` per record):

```text
calibration_data/
  7common_knowledge.json
  7plausible_but_false.json
  7nonsense_knowledge.json
```

These inputs are not included in this code package. Toy test distributions are not valid replacements for the research calibration.

```bash
python -m pip install -r requirements-local-models.txt
python parametric_filtering/baseline_calibration/para11.py \
  --model Qwen/Qwen2.5-7B-Instruct --data-dir calibration_data \
  --output-dir runs/pks_calibration
python parametric_filtering/pks_filtering/00_para14.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input runs/construction/multihop_validated.json \
  --calibration-dir runs/pks_calibration --output-dir runs/construction/pks --on-error error
```

The retained file is `runs/construction/pks/000_abl_filtered_unknowns.json`. The same reference model identifier must be used for scoring and calibration. A failed score aborts; it is not assigned to the unknown class. Alternatively, supply `--pks-model` and `--calibration-dir` to `run_pipeline` **from the first run**; after successful filtering that route writes `benchmark.json`. Changing PKS settings on an existing construction directory requires a new directory, or use the standalone filter above.

The PKS formula and nearest absolute Z-distance are unchanged. Model-dependent scores, source populations, stronger validation, and stochastic outputs mean a new run need not retain exactly 528 records.

## 5. Evaluate new chains

Use the same direct evaluation command as for the release, but point `--dataset` to the new retained JSON and choose `--distractor-family random --length-policy none`. Supply the actual expected count or omit it. Do not mix a new generated benchmark with the historical release under the same run directory.

## Individual entry points

```bash
python -m data_generation.narrative_generation.generate_narratives --input work/selected_sources.json --output work/narratives.json
python -m data_generation.entity_remapping.remap_entities --input work/narratives.json --output work/transformed.json --seed 42
python -m data_generation.qa_generation.generate_qa --input work/transformed.json --output work/qa.json
python -m data_generation.schema_adapters.prepare_distractor_input --input work/qa.json --output work/documents_qa.json
python -m data_generation.distractor_selection.select_random_distractors --input work/documents_qa.json --output work/base.json --audit work/base_audit.json --seed 42
python -m data_generation.dataset_validation.validate_gold --input work/base.json --output work/gold.json --errors work/gold_errors.json
python -m data_generation.multihop_construction.multi_create --input work/gold.json --global-docs work/documents_qa.json --output work/chains.json --errors work/chain_errors.json
python -m data_generation.multihop_construction.filter_multihop --input work/chains.json --output work/multihop_validated.json
```

The orchestrator is preferred because it checks stage order and source identity. The notebook entry points import these Python modules rather than embedding a second divergent implementation.
