# WebDiplomacy collection and cleaning


## Execute from repository root


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