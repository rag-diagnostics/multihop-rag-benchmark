"""Preserved entity/season/year/synonym transformation with explicit seed and lossless I/O."""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rag_utils.tables import read_records, write_records, ensure_serials
from rag_utils.io import atomic_json, fingerprint, require_new_manifest
nlp = None
wordnet = None
master_entity_map = {}
SYNONYM_REPLACEMENT_PROBABILITY = 0.1

REPLACEMENT_COUNTRIES_AND_ADJECTIVES = [
("Bath", "Bathian"), ("York", "Yorkian"), ("Durham", "Durhamian"), ("Exeter", "Exonian"), ("Norwich", "Norwichian"), ("Oxford", "Oxfordian"), ("Cambridge", "Cambridgian"), ("Salisbury", "Salisburian"), ("Hereford", "Herefordian"), ("Worcester", "Worcesterian"), ("Gloucester", "Gloucesterian"), ("Carlisle", "Carlislean"), ("Lancaster", "Lancastrian"), ("Blackpool", "Blackpoolian"), ("Bolton", "Boltonian"), ("Wigan", "Wiganer"), ("Rochdale", "Rochdalian"), ("Oldham", "Oldhamer"), ("Huddersfield", "Huddersfieldian"), ("Halifax", "Halifaxian"), ("Doncaster", "Doncasterian"), ("Rotherham", "Rotherhamer"), ("Sheffield", "Sheffieldian"), ("Leeds", "Leedsian"), ("Liverpool", "Liverpudlian"), ("Preston", "Prestonian"), ("Chester", "Chesterian"), ("Birkenhead", "Birkenheadian"), ("Middlesbrough", "Middlesbroughian"), ("Sunderland", "Sunderlandian"), ("Newcastle", "Newcastlian"), ("Brighton", "Brightonian"), ("Hove", "Hovian"), ("Eastbourne", "Eastbournian"), ("Southampton", "Southamptonian"), ("Portsmouth", "Portsmouthian"), ("Swansea", "Swansean"), ("Cardiff", "Cardiffian"), ("Newport", "Newportian"), ("Bangor", "Bangorian"), ("Belfast", "Belfastian"), ("Londonderry", "Londonderrian"), ("Llandudno", "Llandudnian"), ("Aberystwyth", "Aberystwythian"), ("Wrexham", "Wrexhamian"), ("Stornoway", "Stornowayan"), ("Inverness", "Invernessian"), ("Stirling", "Stirlingian"), ("Perth", "Perthan"), ("Dundee", "Dundonian"), ("Aberdeen", "Aberdonian"), ("Stonehaven", "Stonehavenian"), ("Ayr", "Ayrian"), ("Kilmarnock", "Kilmarnockian"), ("Hamilton", "Hamiltonian"), ("Falkirk", "Falkirian"), ("Greenock", "Greenockian"), ("Grangemouth", "Grangemouthian"), ("Coatbridge", "Coatbridgian"), ("Lisburn", "Lisburnian")
]

REPLACEMENT_CITIES = [
"Boston", "Chicago", "Houston", "Phoenix", "Dallas", "Atlanta", "Denver", "Seattle", "Portland", "Cleveland", "Pittsburgh", "Detroit", "Orlando", "Tampa", "Buffalo", "Albany", "Rochester", "Hartford", "Providence", "Wilmington", "Mobile", "Savannah", "Knoxville", "Lexington", "Nashville", "Memphis", "Boulder", "Salem", "Eugene", "Spokane", "Tacoma", "Fresno"
]

original_countries = {
    "Egypt": "Egyptian", "Ghana": "Ghanaian", "Kenya": "Kenyan", "Libya": "Libyan",
    "South Africa": "South African", "Nigeria": "Nigerian", "Ethiopia": "Ethiopian", "Algeria": "Algerian",
    "USA": "American", "Canada": "Canadian", "Mexico": "Mexican", "Argentina": "Argentine", "Brazil": "Brazilian",
    "Peru": "Peruvian", "Chile": "Chilean", "Cuba": "Cuban", "China": "Chinese", "India": "Indian",
    "Japan": "Japanese", "Vietnam": "Vietnamese", "Indonesia": "Indonesian", "Australia": "Australian",
    "New Zealand": "New Zealander", "Philippines": "Philippine", "Saudi Arabia": "Saudi", "Iran": "Iranian",
    "Persia": "Persian", "Austria": "Austrian", "Britain": "British", "England": "English", "France": "French",
    "Germany": "German", "Italy": "Italian", "Poland": "Polish", "Russia": "Russian", "Spain": "Spanish",
    "Turkey": "Turkish", "Ukraine": "Ukrainian", "Greece": "Greek", "Portugal": "Portuguese",
    "Hungary": "Hungarian", "Iceland": "Icelandic", "Scandinavia": "Scandinavian", "Soviet Union": "Soviet"
}

original_cities = [
    "New-York", "Texas", "California", "Florida", "Washington", "Chicago", "Lima", "Utah",
    "Rome", "Athens", "Naples", "Milan", "Moscow", "St. Petersburg", "Barcelona", "Madrid",
    "Seville", "Gibraltar", "Crimea", "Rumania", "Baghdad", "Byzantium", "Crete"
]

region_map = {
    "Greenland": "Siberia", "The Outback": "Near East", "Patagonia": "The Andes",
    "Baltic states": "The Amazon", "Ottoman Empire": "The Congo", "Heartland": "Antarctica",
    "Pacific Russia": "The Sahara"
}

season_map = {
    "Fall": "Winter",
    "Spring": "Summer",
    "Summer": "Fall",
    "Winter": "Spring"
}

def initialize(seed=42, nlp_model="en_core_web_sm"):
    global nlp, wordnet, master_entity_map
    import spacy
    from nltk.corpus import wordnet as wn
    random.seed(seed)
    nlp = spacy.load(nlp_model)
    # Fail before modifying any records if the optional linguistic resource is missing.
    if SYNONYM_REPLACEMENT_PROBABILITY > 0: wn.ensure_loaded()
    wordnet = wn
    countries = list(REPLACEMENT_COUNTRIES_AND_ADJECTIVES)
    cities = list(REPLACEMENT_CITIES)
    if len(countries) < len(original_countries) or len(cities) < len(original_cities):
        raise ValueError("Replacement pools are too small")
    random.shuffle(countries); random.shuffle(cities)
    country_map = {}
    for (country, adj), (new_country, new_adj) in zip(original_countries.items(), countries):
        country_map[country], country_map[adj] = new_country, new_adj
    master_entity_map = {**country_map, **dict(zip(original_cities, cities)), **region_map}

def get_synonym(word):
    synsets = wordnet.synsets(word.lemma_)
    for synset in synsets:
        synonyms = [lemma.name() for lemma in synset.lemmas() if lemma.name().lower() != word.lemma_.lower() and '_' not in lemma.name()]
        if synonyms: return random.choice(synonyms)
    return None

def transform_text(text: str, last_number: int, ad_mode: bool):
    if not isinstance(text, str): 
        return text, last_number
    doc = nlp(text)
    new_text_parts = []
    current_number = last_number
    for token in doc:
        is_title = token.is_title
        original_word = token.text
        replacement_word = original_word

        if original_word in master_entity_map:
            replacement_word = master_entity_map[original_word]
        elif original_word in season_map:
            replacement_word = season_map[original_word]
        elif token.like_num and len(token.text) == 4:  # Handle years
            increment = random.randint(1, 15)
            current_number += increment
            if ad_mode:
                replacement_word = f"AD {current_number}"
            else:
                replacement_word = f"{current_number} BC"
        elif token.pos_ in ["NOUN", "VERB"] and random.random() < SYNONYM_REPLACEMENT_PROBABILITY:
            synonym = get_synonym(token)
            if synonym: 
                replacement_word = synonym

        if is_title and replacement_word.lower() != original_word.lower():
            replacement_word = replacement_word.capitalize()

        new_text_parts.append(replacement_word + token.whitespace_)
    return "".join(new_text_parts), current_number

def process_records(rows, seed=42, column_name="Summary", initialize_fn=initialize):
    rows = ensure_serials(rows)
    for row in rows:
        if any(row.get(k) for k in ("question", "answer", "hops", "decomposition")):
            raise ValueError("Entity remapping must happen BEFORE QA and multi-hop construction, never on an assembled benchmark")
        if not isinstance(row.get(column_name), str) or not row[column_name].strip():
            raise ValueError("Missing source narrative")
    initialize_fn(seed)
    result = []
    for i, row in enumerate(rows):
        out = dict(row)
        transformed, _ = transform_text(row[column_name], random.randint(2050, 20000), i % 2 == 0)
        out[column_name] = transformed
        out["Document"] = transformed
        result.append(out)
    return result


def process_excel_file(input_path, output_path, column_name="Summary", seed=42):
    if Path(input_path).resolve() == Path(output_path).resolve(): raise ValueError("Preserve the source file; use separate output")
    rows = read_records(input_path)
    result = process_records(rows, seed, column_name)
    write_records(output_path, result)
    atomic_json(str(output_path) + ".transformation.json", {"seed": seed, "entity_map": master_entity_map,
        "season_map": season_map, "synonym_probability": SYNONYM_REPLACEMENT_PROBABILITY,
        "input_sha256": fingerprint(rows), "output_sha256": fingerprint(result),
        "algorithm": "original token-wise transformation; multiword-key behaviour unchanged"})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--column", default="Summary"); p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(); process_excel_file(a.input, a.output, a.column, a.seed)

if __name__ == "__main__": main()
