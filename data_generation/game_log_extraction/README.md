# Extract messages from LLM Diplomacy game histories

This module reconstructs the message-extraction step found in the uploaded historical `game_dip.ipynb` notebook. It reads the `lmvsgame.json` files produced by the LLM-played Diplomacy simulator in `../` and writes one text file for each game. It uses only Python's standard library; **no API key, model inference, or network access is required**.

## Run

Run from your repository root (adjust the input directory for your actual stored games):

```bash
python data_generation/llm_diplomacy/game_log_extraction/extract_game_messages.py \
  --input-dir /path/to/AI_Diplomacy/results \
  --output-dir /path/to/private-work/llm_messages
```

Expected input structure:

```text
results/
  game_000/lmvsgame.json
  game_001/lmvsgame.json
```

Expected output (by default):

```text
llm_messages/
  game_000_messages.txt
  game_001_messages.txt
```

The `--naming original` option instead reproduces the original `game_000.txt` filename convention. **The `summary` default makes names compatible with the file-discovery pattern in `data_generation/narrative_generation/summary_generation.ipynb`.** It does not itself create prose summaries.

## What gets extracted

The historical code traverses `phases[].messages[]`, includes only messages whose `recipient` is not `GLOBAL`, and formats each record as:

```text
FRANCE, Spring 1901 to GERMANY: An example message.
```

It intentionally does not export public/global messages, movement orders, or phase summaries. The original year/season interpretation is preserved. If the input folder does not exist, the cleaned script now raises an error instead of silently generating an example game.
