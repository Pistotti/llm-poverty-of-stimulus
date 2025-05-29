# llm_pos/wilcox/tim_extract_surprisals.py
import sys
import os
import re
import csv
import time
import math # For log base conversion
# from collections import defaultdict # Not needed for sentence-by-sentence

# Get the directory of the current script (wilcox/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (llm_pos/)
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to Python's search path for modules
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import lib # Uses the existing, unmodified lib.py

# --- Configuration ---
INPUT_STIMULI_CSV = os.path.join("data", "processed", "master_stimuli_list.csv")
MODEL_TO_USE = "gpt2"
OUTPUT_DIRECTORY = "tims_results"
# --- End Configuration ---

def load_stimuli_from_csv(filepath):
    """Loads stimuli from a CSV file.
    Expects columns: sentence_type, item_id, condition, full_sentence
    Outputs list of dicts with keys: source_doc_name (from sentence_type), item (from item_id), condition, sentence_text, original_full_sentence_text_for_csv
    """
    stimuli = []
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            required_cols = ['sentence_type', 'item_id', 'condition', 'full_sentence']
            
            if not reader.fieldnames:
                print(f"Error: CSV file '{filepath}' appears to be empty or has no header.")
                return None
            
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                print(f"Error: Input CSV '{filepath}' must contain columns: {', '.join(required_cols)}")
                print(f"Found columns: {reader.fieldnames}. Missing: {', '.join(missing_cols)}")
                return None
            
            for row_idx, row in enumerate(reader):
                if any(row.get(key) is None or row.get(key).strip() == "" for key in required_cols):
                    print(f"Warning: Missing or empty data in required columns at row {row_idx + 2} in {filepath}. Skipping row. Data: {row}")
                    continue

                sentence_text = re.sub(r'\s+', ' ', row['full_sentence']).strip()
                
                stimuli.append({
                    "source_doc_name": row['sentence_type'],
                    "item": row['item_id'],
                    "condition": row['condition'],
                    "sentence_text": sentence_text,
                    "original_full_sentence_text_for_csv": row['full_sentence'].strip()
                })
        
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
    start_time_total = time.time()

    input_csv_path_from_config = INPUT_STIMULI_CSV
    if not os.path.isabs(input_csv_path_from_config):
        input_csv_path = os.path.join(script_dir, input_csv_path_from_config)
    else:
        input_csv_path = input_csv_path_from_config

    print(f"Starting surprisal extraction for stimuli file: {os.path.abspath(input_csv_path)}")
    print(f"Using model: {MODEL_TO_USE}")

    all_stimuli_list = load_stimuli_from_csv(input_csv_path)
    if not all_stimuli_list:
        print("Exiting due to issues loading stimuli.")
        return
    
    print(f"Loaded {len(all_stimuli_list)} total sentences for processing.")

    # --- Prepare Output File ---
    output_dir_abs_path = OUTPUT_DIRECTORY
    if not os.path.isabs(output_dir_abs_path):
        output_dir_abs_path = os.path.join(script_dir, output_dir_abs_path)

    if not os.path.exists(output_dir_abs_path):
        try:
            os.makedirs(output_dir_abs_path)
            print(f"Created output directory: {output_dir_abs_path}")
        except OSError as e:
            print(f"Error creating output directory {output_dir_abs_path}: {e}")
            return
            
    base_input_filename = os.path.basename(input_csv_path)
    model_name_for_filename = MODEL_TO_USE
    output_csv_filename = f"{base_input_filename.split('.')[0]}_surprisals_{model_name_for_filename}.csv"
    csv_filepath = os.path.join(output_dir_abs_path, output_csv_filename)

    headers = [
        "source_doc_name", "item", "condition", "full_sentence_text",
        "bpe_token_index", "bpe_token_id", "bpe_token_str",
        f"log_prob_{model_name_for_filename}",
        f"surprisal_bits_{model_name_for_filename}"
    ]

    # Write header once
    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(headers)
        print(f"Initialized output file with headers: {os.path.abspath(csv_filepath)}")
    except IOError as e:
        print(f"Error initializing output file '{csv_filepath}': {e}")
        return

    # Process each sentence individually
    for idx, stimulus_info in enumerate(all_stimuli_list):
        print(f"\nProcessing sentence {idx + 1}/{len(all_stimuli_list)}: "
              f"Type='{stimulus_info['source_doc_name']}', Item='{stimulus_info['item']}', Cond='{stimulus_info['condition']}'")
        print(f"  Sentence: '{stimulus_info['sentence_text'][:100]}...'") # Print first 100 chars
        
        sentence_text_to_process = (stimulus_info['sentence_text'],) # Comma makes it a tuple of one

        batch_start_time = time.time()
        surprisal_data_for_sentence = None
        try:
            surprisal_data_for_sentence = lib.get_surprisals_per_model(
                sentences=sentence_text_to_process,
                models=(MODEL_TO_USE,)
            )
        except Exception as e:
            print(f"  CRITICAL ERROR during lib.get_surprisals_per_model for sentence:")
            print(f"    Type: {stimulus_info['source_doc_name']}, Item: {stimulus_info['item']}, Condition: {stimulus_info['condition']}")
            print(f"    Full Sentence: {stimulus_info['original_full_sentence_text_for_csv']}")
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nStopping script due to error.")
            return # Stop script

        if not surprisal_data_for_sentence or MODEL_TO_USE not in surprisal_data_for_sentence or not surprisal_data_for_sentence[MODEL_TO_USE]:
            print(f"  CRITICAL ERROR: No surprisal data returned from lib.get_surprisals_per_model for sentence:")
            print(f"    Type: {stimulus_info['source_doc_name']}, Item: {stimulus_info['item']}, Condition: {stimulus_info['condition']}")
            print(f"    Full Sentence: {stimulus_info['original_full_sentence_text_for_csv']}")
            print("\nStopping script due to missing data for this sentence.")
            return # Stop script

        sentence_surprisal_obj = surprisal_data_for_sentence[MODEL_TO_USE][0] # Should be only one result

        # Check for mismatch (should ideally be 1 input, 1 output)
        if len(surprisal_data_for_sentence[MODEL_TO_USE]) != 1:
             print(f"  CRITICAL WARNING for sentence: Type='{stimulus_info['source_doc_name']}', Item='{stimulus_info['item']}', Cond='{stimulus_info['condition']}'")
             print(f"  Expected 1 surprisal object, got {len(surprisal_data_for_sentence[MODEL_TO_USE])}. This should not happen.")
             print("\nStopping script due to unexpected result structure.")
             return


        try:
            with open(csv_filepath, 'a', newline='', encoding='utf-8') as csvfile_append:
                csv_writer_append = csv.writer(csvfile_append)
                
                for token_seq_idx, token_obj in enumerate(sentence_surprisal_obj.tokens):
                    bpe_id_to_write = "N/A" 
                    if token_obj.surprisal is not None:
                        natural_log_surprisal = token_obj.surprisal
                        log_prob_val = -natural_log_surprisal
                        surprisal_bits_val = natural_log_surprisal / math.log(2)
                    else:
                        log_prob_val = "N/A"
                        surprisal_bits_val = "N/A"

                    csv_writer_append.writerow([
                        stimulus_info['source_doc_name'],
                        stimulus_info['item'],
                        stimulus_info['condition'],
                        stimulus_info['original_full_sentence_text_for_csv'],
                        token_seq_idx,
                        bpe_id_to_write,
                        token_obj.text,
                        f"{log_prob_val:.8f}" if isinstance(log_prob_val, float) else log_prob_val,
                        f"{surprisal_bits_val:.8f}" if isinstance(surprisal_bits_val, float) else surprisal_bits_val
                    ])
        except IOError as e:
            print(f"  CRITICAL ERROR writing to output file for sentence: {stimulus_info}")
            print(f"  Error: {e}")
            print("\nStopping script due to IO error.")
            return
        except Exception as e:
            print(f"  CRITICAL UNEXPECTED ERROR during output writing for sentence: {stimulus_info}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            print("\nStopping script due to unexpected error.")
            return

        batch_end_time = time.time()
        print(f"  Successfully processed and saved sentence {idx + 1} in {batch_end_time - batch_start_time:.2f} seconds.")

    total_time_end = time.time()
    print(f"\nAll sentences processed. Total script finished in {total_time_end - start_time_total:.2f} seconds.")
    print(f"Full output saved to: {os.path.abspath(csv_filepath)}")

if __name__ == "__main__":
    main()