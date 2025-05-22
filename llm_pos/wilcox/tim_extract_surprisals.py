# llm_pos/wilcox/tim_extract_surprisals.py
import sys
import os
import re
import csv
import time
import math # For log base conversion

# Get the directory of the current script (wilcox/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (llm_pos/)
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to Python's search path for modules
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lib # Uses the existing, unmodified lib.py

# --- Configuration ---
# Input CSV file with columns: source_doc_name, item, condition, full_sentence
INPUT_STIMULI_CSV = "tims_data/extract_test.csv"  # Example name, place your CSV here

MODEL_TO_USE = "gpt2"

# Output directory will be created inside llm_pos/wilcox/ if it doesn't exist
OUTPUT_DIRECTORY = "tims_results" # New output directory
# Output CSV filename will be generated dynamically
# --- End Configuration ---

def load_stimuli_from_csv(filepath):
    """Loads stimuli from a CSV file.
    Expects columns: source_doc_name, item, condition, full_sentence
    """
    stimuli = []
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f: # utf-8-sig handles potential BOM
            reader = csv.DictReader(f)
            required_cols = ['source_doc_name', 'item', 'condition', 'full_sentence']
            
            if not reader.fieldnames:
                print(f"Error: CSV file '{filepath}' appears to be empty or has no header.")
                return None
            if not all(key in reader.fieldnames for key in required_cols):
                print(f"Error: Input CSV '{filepath}' must contain columns: {', '.join(required_cols)}")
                print(f"Found columns: {reader.fieldnames}")
                return None
            
            for row_idx, row in enumerate(reader):
                # Check for missing essential data in the row
                if any(row.get(key) is None for key in required_cols):
                    print(f"Warning: Missing data in required columns at row {row_idx + 2} in {filepath}. Skipping row.")
                    continue

                sentence_text = re.sub(r'\s+', ' ', row['full_sentence']).strip()
                if sentence_text:
                    stimuli.append({
                        "source_doc_name": row['source_doc_name'],
                        "item": row['item'],
                        "condition": row['condition'],
                        "sentence_text": sentence_text, # This will be passed to the model
                        "original_full_sentence_text_for_csv": row['full_sentence'].strip() # Preserve for CSV output
                    })
                else:
                    print(f"Warning: Empty full_sentence found at row {row_idx + 2} in {filepath} (after cleaning). Skipping.")
        
        if not stimuli:
            print(f"Warning: No valid stimuli found or loaded from {filepath}")
    except FileNotFoundError:
        print(f"Error: Stimuli file not found at {os.path.abspath(filepath)}")
        return None
    except Exception as e:
        print(f"Error reading CSV file '{filepath}': {e}")
        import traceback
        traceback.print_exc()
        return None
    return stimuli

def main():
    # Construct absolute path for input file if it's relative to script dir
    input_csv_path = INPUT_STIMULI_CSV
    if not os.path.isabs(input_csv_path):
        input_csv_path = os.path.join(script_dir, input_csv_path)

    print(f"Starting surprisal extraction for stimuli file: {input_csv_path}")
    print(f"Using model: {MODEL_TO_USE}")

    stimuli_list = load_stimuli_from_csv(input_csv_path)
    if not stimuli_list:
        print("Exiting due to issues loading stimuli.")
        return

    # Extract just the sentence texts to pass to lib.get_surprisals_per_model
    sentence_texts_to_process = tuple(item['sentence_text'] for item in stimuli_list)

    print(f"Loaded {len(sentence_texts_to_process)} sentences for processing.")
    script_start_time = time.time()

    print("Fetching surprisal data...")
    surprisal_data_per_model = lib.get_surprisals_per_model(
        sentences=sentence_texts_to_process,
        models=(MODEL_TO_USE,)
    )
    print("Surprisal data fetched.")

    # --- Prepare Output ---
    output_dir_path = OUTPUT_DIRECTORY
    if not os.path.isabs(output_dir_path):
        output_dir_path = os.path.join(script_dir, output_dir_path)

    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f"Created output directory: {output_dir_path}")
    
    source_doc_name_for_filename = os.path.basename(input_csv_path)
    model_name_for_cols_and_filename = MODEL_TO_USE # e.g., "gpt2"

    output_csv_filename = f"{source_doc_name_for_filename.split('.')[0]}_surprisals_{model_name_for_cols_and_filename}.csv"
    csv_filepath = os.path.join(output_dir_path, output_csv_filename)
    print(f"Attempting to save surprisal data to CSV: {csv_filepath}")

    headers = [
        "source_doc_name", "item", "condition", "full_sentence_text",
        "bpe_token_index", "bpe_token_id", "bpe_token_str",
        f"log_prob_{model_name_for_cols_and_filename}",
        f"surprisal_bits_{model_name_for_cols_and_filename}"
    ]

    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)

            if MODEL_TO_USE in surprisal_data_per_model:
                sentence_surprisals_tuple = surprisal_data_per_model[MODEL_TO_USE]
                
                if len(sentence_surprisals_tuple) != len(stimuli_list):
                    print(f"CRITICAL WARNING: Mismatch between input stimuli ({len(stimuli_list)}) "
                          f"and processed sentences ({len(sentence_surprisals_tuple)}). "
                          "Output might be misaligned or incomplete.")
                
                for i, sentence_surprisal_obj in enumerate(sentence_surprisals_tuple):
                    if i >= len(stimuli_list):
                        print(f"Warning: More surprisal objects than stimuli at index {i}. Stopping.")
                        break 
                    
                    stimulus_info = stimuli_list[i]
                    
                    for token_seq_idx, token_obj in enumerate(sentence_surprisal_obj.tokens):
                        # BPE Token ID cannot be retrieved without lib.py modification
                        bpe_id_to_write = "N/A" 

                        if token_obj.surprisal is not None:
                            natural_log_surprisal = token_obj.surprisal
                            log_prob_val = -natural_log_surprisal
                            surprisal_bits_val = natural_log_surprisal / math.log(2)
                        else:
                            log_prob_val = "N/A"
                            surprisal_bits_val = "N/A"

                        csv_writer.writerow([
                            stimulus_info['source_doc_name'],
                            stimulus_info['item'],
                            stimulus_info['condition'],
                            stimulus_info['original_full_sentence_text_for_csv'], # Use preserved full sentence
                            token_seq_idx,          # bpe_token_index
                            bpe_id_to_write,        # bpe_token_id (placeholder)
                            token_obj.text,         # bpe_token_str
                            f"{log_prob_val:.8f}" if isinstance(log_prob_val, float) else log_prob_val,
                            f"{surprisal_bits_val:.8f}" if isinstance(surprisal_bits_val, float) else surprisal_bits_val
                        ])
            else:
                print(f"Warning: No surprisal data found for model {MODEL_TO_USE} in the results dictionary.")
           
            print(f"Surprisal data saved to CSV: {csv_filepath}")
    except IOError as e:
        print(f"Error writing CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    script_end_time = time.time()
    print(f"Script finished in {script_end_time - script_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()