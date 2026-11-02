# QA generation (Gemini 2.5 Pro) 
## What it does

1. Opens `0modified_final_dataset_004.xlsx`, produced by the earlier entity-remapping stage.
2. Uses each non-empty `Summary` field to ask `gemini-2.5-pro` for a factual, anchored question and concise answer.
3. Saves `QUESTION:`/`ANSWER:` text files in `gemini_out_6_004/` so interrupted runs can resume.
4. Fills *missing* `question` and `answer` cells in the **same** workbook by reading those text files.

No original workbook, generated dataset or API key is included.

## Requirements

Python 3.10+ (recommended). Install requirements from this directory:

```bash
python -m pip install -r requirements.txt
```

The supported SDK is Google's `google-genai`, using the `genai.Client` and `client.models.generate_content` API. See the [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/) documentation.

## Run

1. Back up your input workbook **before running**; the program edits it in place.
2. Ensure it has a unique `Serial` column, an entity-remapped `Summary` column with strings, and blank/missing `question` and `answer` entries where you want new QA pairs.
3. Supply a valid `GEMINI_API_KEY` in a **private environment variable** (see `API_USAGE.md`). Do not insert the key into any file in Git.
4. From this folder run:

   ```bash
   python generate_qa.py
   ```

Alternatively, run `generate_qa.ipynb` from this same working directory. The notebook and script are equivalent. Do not run both simultaneously on the same workbook.
