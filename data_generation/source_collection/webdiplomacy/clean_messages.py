import os
import re

def clean_text_files(input_folder, output_folder):
    """
    Reads all .txt files from an input folder, removes specific metadata and
    separator lines, and saves the cleaned files to an output folder.

    Args:
        input_folder (str): The path to the folder containing the files to clean.
        output_folder (str): The path to the folder where cleaned files will be saved.
    """
    # --- 1. Create the output folder if it doesn't exist ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # --- 2. Get the list of files to process ---
    try:
        files_to_process = [
            f for f in os.listdir(input_folder) if f.endswith('.txt')
        ]
        if not files_to_process:
            print(f"No .txt files found in '{input_folder}'.")
            return
    except FileNotFoundError:
        print(f"Error: The input folder '{input_folder}' was not found.")
        return

    print(f"Found {len(files_to_process)} .txt files to clean...")

    # --- 3. Compile regex patterns for lines to be removed ---
    # We compile them once for efficiency. The patterns are designed to match
    # against lines that have had their leading/trailing whitespace stripped.
    patterns_to_remove = [
        re.compile(r'^================================================================================$'),
        re.compile(r'^--- Game ID: .*? \| Players: .*? ---$'),
        re.compile(r'^Date: .*'),
        re.compile(r'^\d{1,2} \w+ \d{4}$'),
        re.compile(r'^\[Total messages in this conversation: \d+\]$')
    ]

    # --- 4. Loop through each file, clean it, and save it ---
    for filename in files_to_process:
        input_filepath = os.path.join(input_folder, filename)
        output_filepath = os.path.join(output_folder, filename)

        try:
            kept_lines = []
            with open(input_filepath, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    # Remove leading/trailing whitespace to simplify matching
                    stripped_line = line.strip()

                    # Skip lines that are empty after stripping
                    if not stripped_line:
                        continue
                    
                    # Check if the stripped line matches any of the patterns for removal
                    should_remove = False
                    for pattern in patterns_to_remove:
                        if pattern.match(stripped_line):
                            should_remove = True
                            break
                    
                    # If the line is not marked for removal, add it to our list of kept lines
                    if not should_remove:
                        kept_lines.append(stripped_line)

            # Join the kept lines back into a single string, separated by newlines
            cleaned_content = '\n'.join(kept_lines)

            with open(output_filepath, 'w', encoding='utf-8') as f_out:
                f_out.write(cleaned_content)

            print(f"Successfully cleaned '{filename}' and saved to '{output_folder}'")

        except Exception as e:
            print(f"Could not process file {filename}. Error: {e}")

    print("\nCleaning process complete!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Clean metadata and separators from WebDiplomacy text files")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if os.path.abspath(args.input_dir) == os.path.abspath(args.output_dir):
        parser.error("Input and output directories must be different")
    clean_text_files(args.input_dir, args.output_dir)
