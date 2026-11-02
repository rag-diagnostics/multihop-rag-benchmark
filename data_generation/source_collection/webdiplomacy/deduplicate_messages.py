import pathlib

def remove_all_duplicates(file_path, output_folder):
    """
    Reads a file, removes all duplicate lines throughout the entire file,
    and saves the result to a new file. Only the first occurrence of any
    given line is kept.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        cleaned_lines = []
        seen_lines = set()  # This set will act as our memory for unique lines

        for line in original_lines:
            # We strip the line to handle potential whitespace differences.
            # E.g., "  hello" and "hello  " are treated as the same content.
            stripped_line = line.strip()
            
            # An empty line after stripping means it was just whitespace or a blank line.
            # We only check non-empty lines for duplicates.
            if stripped_line and stripped_line in seen_lines:
                continue  # Skip this line, we've already seen its content
            
            # If we're here, it's a new line or a blank one we should keep.
            cleaned_lines.append(line)
            if stripped_line:
                seen_lines.add(stripped_line) # Add the content to our memory

        # Define the path for the new, cleaned file
        output_file_path = output_folder / file_path.name
        
        # Write the cleaned lines to the new file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
            
    except Exception as e:
        print(f"  -> Could not process file {file_path.name}. Error: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remove exact duplicate lines from each message file")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    folder_path = pathlib.Path(args.input_dir)
    output_folder = pathlib.Path(args.output_dir)
    if not folder_path.is_dir():
        parser.error(f"Input directory does not exist: {folder_path}")
    if folder_path.resolve() == output_folder.resolve():
        parser.error("Input and output directories must be different")
    output_folder.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(folder_path.glob('*.txt'))
    for file in txt_files:
        remove_all_duplicates(file, output_folder)
    print(f"Processed {len(txt_files)} files into {output_folder}")
