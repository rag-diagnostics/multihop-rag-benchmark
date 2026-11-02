# Entity remapping

## What it does

- Shuffles fixed pools of replacement countries/adjectives and cities and creates a run-level mapping.
- Replaces the configured named entities and regions, transforms season names, and replaces recognised four-digit numbers with generated AD/BC dates.
- Uses WordNet to optionally substitute nouns and verbs with probability **0.1**.
- Applies the transformation to the Excel column **`Summary`**, retaining other workbook columns.



```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet
```

The spaCy language model and NLTK WordNet corpus are external resources and are not bundled in the repository.

## Input and output (historical names retained)

| File/column | Role |
|---|---|
| `final_dataset.xlsx` | Create the dataset and use it for entity transformation |
| `Summary` | Required column containing original narrative text |
| `0modified_final_dataset_004.xlsx` | Output workbook with transformed `Summary` values |

Put `final_dataset.xlsx` in your current working directory and run:

```bash
python remap_entities.py
```

You can alternatively open `remap_entities.ipynb` and run its single cell with the same working directory. Do not execute **both** if you intend to compare their outputs: the unseeded random remapping will vary across runs.



## Relation to the paper

This covers the **implementation of Section 3.2, Randomised Entity Transformation**.