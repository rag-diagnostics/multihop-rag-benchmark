import os
import json
import argparse

def process_game_logs(input_base_dir="llm_dip_results", output_base_dir="game_text", naming="summary"):
    """
    Reads Diplomacy game logs from JSON files, extracts and formats messages,
    and writes them to text files.
    """
    json_filename = 'lmvsgame.json'

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"Created directory: {output_base_dir}")

    # Do not manufacture demonstration records: a missing source is an error.
    if not os.path.isdir(input_base_dir):
        raise FileNotFoundError(f"Input directory '{input_base_dir}' not found")

    # Iterate over each subdirectory in the base directory
    for subdir_name in os.listdir(input_base_dir):
        subdir_path = os.path.join(input_base_dir, subdir_name)

        if os.path.isdir(subdir_path):
            json_file_path = os.path.join(subdir_path, json_filename)

            if os.path.exists(json_file_path):
                print(f"Processing {json_file_path}...")
                try:
                    with open(json_file_path, 'r') as f:
                        game_data = json.load(f)

                    conversations = []
                    if 'phases' in game_data:
                        for phase in game_data['phases']:
                            if 'messages' in phase:
                                for message in phase['messages']:
                                    if message.get('recipient') != 'GLOBAL':
                                        sender = message.get('sender', 'Unknown Sender')
                                        recipient = message.get('recipient', 'Unknown Recipient')
                                        phase_str = message.get('phase', 'S1900M')
                                        text = message.get('message', '')

                                        # Determine season and year from phase
                                        season_char = phase_str[0]
                                        year = phase_str[1:5]
                                        season = "Spring" if season_char == 'S' else "Fall"

                                        formatted_message = f"{sender}, {season} {year} to {recipient}: {text}"
                                        conversations.append(formatted_message)

                    # Write the formatted conversations to a new text file
                    output_txt_filename = (f"{subdir_name}_messages.txt" if naming == "summary"
                                           else f"{subdir_name}.txt")
                    output_txt_path = os.path.join(output_base_dir, output_txt_filename)

                    with open(output_txt_path, 'w') as f:
                        f.write('\n'.join(conversations))

                    print(f"Successfully created {output_txt_path}")

                except json.JSONDecodeError:
                    print(f"Error: Could not decode JSON from {json_file_path}")
                except Exception as e:
                    print(f"An unexpected error occurred while processing {json_file_path}: {e}")
            else:
                print(f"Warning: '{json_filename}' not found in {subdir_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract private messages from LLM Diplomacy game histories")
    parser.add_argument("--input-dir", required=True,
                        help="Folder containing game_*/lmvsgame.json from the original simulator")
    parser.add_argument("--output-dir", required=True,
                        help="Destination folder for extracted text files")
    parser.add_argument("--naming", choices=("summary", "original"), default="summary",
                        help="summary: game_000_messages.txt (for summary_generation.ipynb); original: game_000.txt")
    args = parser.parse_args()
    process_game_logs(args.input_dir, args.output_dir, naming=args.naming)

