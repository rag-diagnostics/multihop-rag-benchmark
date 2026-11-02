## Provenance and scope

- **Input:** text files named `game_<id>_messages.txt`, each containing dialogue from a Diplomacy game.
- **Staging:** combines text into `real_game_data_2025.xlsx` with columns `game_id` and `text`.
- **Narrative conversion:** uses the historical prompt to ask Gemini to rewrite the dialogue as in-universe historical prose.
- **Output:** writes generated prose into the `Summary` column of the **same workbook**, saving progress after each row when `CONCURRENT_WORKERS=1`.


## Setup

Install Python 3 and the historical notebook's dependencies:

```bash
python -m pip install pandas openpyxl google-generativeai jupyterlab
```

Set the `DIPLOMACY_TEXT_DIR` environment variable to the directory containing the input `.txt` files. If unset, the notebook looks for `game_messages` in its current working directory.

For Gemini access, configure the `GEMINI_API_KEY` environment variable **outside source control**. 
Open `summary_generation.ipynb` and run its two cells in order in a directory where you want the workbook written. The second cell updates `real_game_data_2025.xlsx` in place; back up any existing workbook before running.