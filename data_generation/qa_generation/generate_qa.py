import pandas as pd
from google import genai
from google.genai import types
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- Configuration ---
# IMPORTANT: Do not paste API keys into source code or notebooks.
# Set GEMINI_API_KEY as an environment variable before running.
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro"
XLSX_FILE_PATH = '0modified_final_dataset_004.xlsx'
OUTPUT_FOLDER = 'gemini_out_6_004'
CONCURRENT_WORKERS = 10 # Number of parallel API calls

# Global client object
client = None

def configure_gemini_api():
    """Configures the Gemini API client using an environment variable."""
    global client
    if not API_KEY:
        print("ERROR: Set GEMINI_API_KEY in your private environment before running.")
        raise SystemExit(1)
    try:
        client = genai.Client(api_key=API_KEY)
        # Match the original preflight connection check using Gemini's model lookup.
        client.models.get(model=MODEL_NAME)
        print("Gemini API client configured successfully.")
    except Exception as e:
        # Avoid printing response bodies or authorization information in logs.
        print(f"Error configuring Gemini API ({type(e).__name__}). "
              "Check model access, credentials and connectivity.")
        raise SystemExit(1)


def generate_qa(summary_text, question_type):
    """
    Generates a SIMPLE but ANCHORED question-answer pair using the Gemini API.
    """
    if not summary_text or pd.isna(summary_text):
        return None, None

    # --- This prompt is model-agnostic and should work well ---
    prompt = f"""
    You are an expert at creating question-answer pairs for a benchmark. Your task is to generate a question that is both **simple and unambiguously answerable** only from the provided text.

    **RULE 1: The Question Must Be SIMPLE and FACTUAL.**
    * The structure of the question should be a direct query. Focus on "Who", "What", "Where", "When".
    * AVOID "Why" or "How" questions that require complex reasoning.

    **RULE 2: The Question Must Be UNAMBIGUOUSLY ANCHORED.**
    * To prevent ambiguity in a large dataset, the question **MUST include a second piece of context** from the text to act as an "anchor".
    * This anchor should be another entity (like a town or place), a specific action (like "proposing a pact"), or a season (NOT YEAR, BECAUSE YEAR WILL BE EASY FOR THE SYSTEM TO DO SIMPLE KEYWORD SEARCH). This ensures the question can only be answered by this specific document.

    * **Bad Example (Too Vague):**
        -   Summary: "Gabon extended an invitation to Congo, proposing the formation of a robust coalition to secure crucial supplying centers across the Mediterranean Sea."
        -   Vague Question: "Which town proposed a coalition?" (Many countries could do this).
    * **Bad Example (Year Specified, hence it is too easy, do not use year as an anchor or identifier in the question):**
        -   Question: Which season did Aberystwyth announce its intention to advance into Munich in 8065 BC?

    * **Good Example (Simple and Anchored):**
        -   Summary: "Gabon extended an invitation to Congo, proposing the formation of a robust coalition to secure crucial supplying centers across the Mediterranean Sea."
        -   Anchored Question: "Which town proposed a coalition to Congo to secure the Mediterranean Sea?"
        -   Answer: Gabon

    **RULE 3: The Answer Must Be EXTREMELY CONCISE.**
    * The answer must be a single entity (name, place, etc.) extracted directly from the text.
    * **STRICT LIMIT: One to two words maximum.**

    ---
    **YOUR TASK:**
    Generate one **simple but anchored** question-answer pair from the summary below.

    **Additional Instruction:** The question must begin with a phrase like "{question_type}".

    **Summary:** "{summary_text}"

    **Format your output strictly as follows:**
    QUESTION: [Your simple and anchored question here]
    ANSWER: [Your 1-2 word answer here]
    """

    # Retry logic for API calls
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are an expert in creating factual question-answer pairs.",
                    temperature=0.0 # Preserve the original generation setting
                ),
            )
            response_text = response.text

            # Parse the generated text
            lines = response_text.strip().split('\n')
            question = None
            answer = None

            for line in lines:
                if line.startswith("QUESTION:"):
                    question = line.replace("QUESTION:", "").strip()
                elif line.startswith("ANSWER:"):
                    answer = line.replace("ANSWER:", "").strip()

            if question and answer:
                return question, answer # Success

        except Exception as e:
            print(f"API call attempt {attempt + 1} failed ({type(e).__name__}). Retrying...")
            time.sleep(2 ** attempt) # Exponential backoff

    print(f"API call failed after multiple retries for summary: {summary_text[:100]}...")
    return None, None


def process_and_save_row(serial, summary, question_type):
    """
    Takes serial, summary, and question_type, generates Q&A, and saves to a text file.
    """
    question, answer = generate_qa(summary, question_type)

    if question and answer:
        output_filepath = os.path.join(OUTPUT_FOLDER, f"{serial}.txt")
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(f"QUESTION: {question}\n")
                f.write(f"ANSWER: {answer}\n")
            return f"Success: Saved Q&A for Serial {serial}"
        except Exception as e:
            return f"Error: Failed to write file for Serial {serial}: {e}"
    else:
        return f"Error: Failed to generate Q&A for Serial {serial}."


def generate_text_files_concurrently(file_path):
    """
    Reads the XLSX, identifies rows to be processed, and runs the generation concurrently.
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    if 'Serial' not in df.columns:
        print("Error: The XLSX file must have a 'Serial' column for naming output files.")
        return

    QUESTION_TYPES = ["What", "Where", "Which year", "Which season", "Which town"]

    tasks = []
    task_counter = 0
    for _, row in df.iterrows():
        serial = row['Serial']
        summary = row.get('Summary')
        output_filepath = os.path.join(OUTPUT_FOLDER, f"{serial}.txt")

        if os.path.exists(output_filepath):
            continue

        if pd.isna(summary) or not summary.strip():
            continue

        question_type = QUESTION_TYPES[task_counter % len(QUESTION_TYPES)]
        tasks.append((serial, summary, question_type))
        task_counter += 1

    if not tasks:
        print(f"No new rows to process. All corresponding text files already exist in '{OUTPUT_FOLDER}'.")
        return

    print(f"Found {len(tasks)} new rows to process.")

    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        future_to_serial = {
            executor.submit(process_and_save_row, serial, summary, q_type): serial
            for serial, summary, q_type in tasks
        }

        for future in tqdm(as_completed(future_to_serial), total=len(tasks), desc="Processing Rows"):
            result = future.result()
            if "Error" in result:
                # Errors are printed from within the generate_qa function, so we don't need to print again
                pass

    print("\nConcurrent text file generation complete.")


def update_xlsx_from_text_files(file_path):
    """
    Reads the generated text files and updates the main XLSX file.
    (This function remains unchanged)
    """
    print("Starting final update of the XLSX file...")
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find '{file_path}' to update.")
        return

    if 'question' not in df.columns:
        df['question'] = None
    if 'answer' not in df.columns:
        df['answer'] = None

    df['question'] = df['question'].astype(object)
    df['answer'] = df['answer'].astype(object)

    updated_rows = 0
    for index, row in df.iterrows():
        serial = row['Serial']
        txt_filepath = os.path.join(OUTPUT_FOLDER, f"{serial}.txt")

        if os.path.exists(txt_filepath):
            try:
                with open(txt_filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                question = None
                answer = None

                for line in lines:
                    if line.startswith("QUESTION:"):
                        question = line.replace("QUESTION:", "").strip()
                    elif line.startswith("ANSWER:"):
                        answer = line.replace("ANSWER:", "").strip()

                if question and answer and (pd.isna(row.get('question')) or pd.isna(row.get('answer'))):
                    df.at[index, 'question'] = question
                    df.at[index, 'answer'] = answer
                    updated_rows += 1

            except Exception as e:
                print(f"Could not process file {txt_filepath}: {e}")

    if updated_rows > 0:
        print(f"Updating {updated_rows} rows in the Excel file.")
        try:
            df.to_excel(file_path, index=False)
            print(f"Successfully saved updated data to '{file_path}'.")
        except Exception as e:
            print(f"CRITICAL: Failed to save the final XLSX file: {e}")
            print("Please check file permissions or if the file is open elsewhere.")
    else:
        print("No new information to add to the Excel file.")


if __name__ == "__main__":
    configure_gemini_api()

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output directory: '{OUTPUT_FOLDER}'")

    print("--- Step 1: Generating Q&A and saving to text files ---")
    generate_text_files_concurrently(XLSX_FILE_PATH)

    print("\n--- Step 2: Updating main XLSX file from text files ---")
    update_xlsx_from_text_files(XLSX_FILE_PATH)

    print("\nAll processing is complete!")