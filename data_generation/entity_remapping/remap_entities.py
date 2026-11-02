import pandas as pd
import spacy
import random
from nltk.corpus import wordnet

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# --- CONTROL PANEL ---
SYNONYM_REPLACEMENT_PROBABILITY = 0.1 # Adjust 0.0 (off) to 1.0 (always on)

# --- 1. REPLACEMENT POOLS (EXPANDED TO PREVENT ERRORS) ---
# This list has been expanded to be larger than the number of original countries.
REPLACEMENT_COUNTRIES_AND_ADJECTIVES = [
("Bath", "Bathian"), ("York", "Yorkian"), ("Durham", "Durhamian"), ("Exeter", "Exonian"), ("Norwich", "Norwichian"), ("Oxford", "Oxfordian"), ("Cambridge", "Cambridgian"), ("Salisbury", "Salisburian"), ("Hereford", "Herefordian"), ("Worcester", "Worcesterian"), ("Gloucester", "Gloucesterian"), ("Carlisle", "Carlislean"), ("Lancaster", "Lancastrian"), ("Blackpool", "Blackpoolian"), ("Bolton", "Boltonian"), ("Wigan", "Wiganer"), ("Rochdale", "Rochdalian"), ("Oldham", "Oldhamer"), ("Huddersfield", "Huddersfieldian"), ("Halifax", "Halifaxian"), ("Doncaster", "Doncasterian"), ("Rotherham", "Rotherhamer"), ("Sheffield", "Sheffieldian"), ("Leeds", "Leedsian"), ("Liverpool", "Liverpudlian"), ("Preston", "Prestonian"), ("Chester", "Chesterian"), ("Birkenhead", "Birkenheadian"), ("Middlesbrough", "Middlesbroughian"), ("Sunderland", "Sunderlandian"), ("Newcastle", "Newcastlian"), ("Brighton", "Brightonian"), ("Hove", "Hovian"), ("Eastbourne", "Eastbournian"), ("Southampton", "Southamptonian"), ("Portsmouth", "Portsmouthian"), ("Swansea", "Swansean"), ("Cardiff", "Cardiffian"), ("Newport", "Newportian"), ("Bangor", "Bangorian"), ("Belfast", "Belfastian"), ("Londonderry", "Londonderrian"), ("Llandudno", "Llandudnian"), ("Aberystwyth", "Aberystwythian"), ("Wrexham", "Wrexhamian"), ("Stornoway", "Stornowayan"), ("Inverness", "Invernessian"), ("Stirling", "Stirlingian"), ("Perth", "Perthan"), ("Dundee", "Dundonian"), ("Aberdeen", "Aberdonian"), ("Stonehaven", "Stonehavenian"), ("Ayr", "Ayrian"), ("Kilmarnock", "Kilmarnockian"), ("Hamilton", "Hamiltonian"), ("Falkirk", "Falkirian"), ("Greenock", "Greenockian"), ("Grangemouth", "Grangemouthian"), ("Coatbridge", "Coatbridgian"), ("Lisburn", "Lisburnian")
]

REPLACEMENT_CITIES = [
"Boston", "Chicago", "Houston", "Phoenix", "Dallas", "Atlanta", "Denver", "Seattle", "Portland", "Cleveland", "Pittsburgh", "Detroit", "Orlando", "Tampa", "Buffalo", "Albany", "Rochester", "Hartford", "Providence", "Wilmington", "Mobile", "Savannah", "Knoxville", "Lexington", "Nashville", "Memphis", "Boulder", "Salem", "Eugene", "Spokane", "Tacoma", "Fresno"
]

# --- 2. DYNAMIC MAPPING CREATION ---
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

# Robustness Check: Ensure replacement pools are large enough
if len(REPLACEMENT_COUNTRIES_AND_ADJECTIVES) < len(original_countries):
    raise ValueError(f"Replacement country pool is too small. Need {len(original_countries)}, have {len(REPLACEMENT_COUNTRIES_AND_ADJECTIVES)}.")
if len(REPLACEMENT_CITIES) < len(original_cities):
    raise ValueError(f"Replacement city pool is too small. Need {len(original_cities)}, have {len(REPLACEMENT_CITIES)}.")

# Shuffle the replacement lists
random.shuffle(REPLACEMENT_COUNTRIES_AND_ADJECTIVES)
random.shuffle(REPLACEMENT_CITIES)

# Create the new random country map
country_map = {}
shuffled_replacements = REPLACEMENT_COUNTRIES_AND_ADJECTIVES[:len(original_countries)]
for i, (country_key, adj_key) in enumerate(original_countries.items()):
    new_country, new_adj = shuffled_replacements[i]
    country_map[country_key] = new_country
    country_map[adj_key] = new_adj

city_map = dict(zip(original_cities, REPLACEMENT_CITIES))
region_map = {
    "Greenland": "Siberia", "The Outback": "Near East", "Patagonia": "The Andes",
    "Baltic states": "The Amazon", "Ottoman Empire": "The Congo", "Heartland": "Antarctica",
    "Pacific Russia": "The Sahara"
}
master_entity_map = {**country_map, **city_map, **region_map}
season_map = {
    "Fall": "Winter",
    "Spring": "Summer",
    "Summer": "Fall",
    "Winter": "Spring"
}
# --- TEXT TRANSFORMATION FUNCTION ---
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


def process_excel_file(input_path, output_path, column_name):
    try:
        print(f"🔄 Reading data from '{input_path}'...")
        df = pd.read_excel(input_path)
        if column_name not in df.columns:
            print(f"❌ Error: Column '{column_name}' not found in the file.")
            return

        print(f"⚙️ Transforming text in the '{column_name}' column...")
        new_summaries = []
        ad_mode = True  # Start with AD for first row

        for summary_text in df[column_name]:
            starting_number = random.randint(2050, 20000)
            transformed_summary, _ = transform_text(summary_text, starting_number, ad_mode)
            new_summaries.append(transformed_summary)

            # Alternate AD/BC mode for next row
            ad_mode = not ad_mode

        df[column_name] = new_summaries
        df.to_excel(output_path, index=False)
        print(f"✅ Successfully created modified file at '{output_path}'!")

    except FileNotFoundError:
        print(f"❌ Error: The file '{input_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- RUN THE PROCESS ---
if __name__ == "__main__":
    input_file = "final_dataset.xlsx"
    output_file = "0modified_final_dataset_004.xlsx"
    target_column = "Summary"
    process_excel_file(input_file, output_file, target_column)