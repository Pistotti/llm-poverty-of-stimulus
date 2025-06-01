# llm-poverty-of-stimulus/llm_pos/wilcox/preprocess_lan_stimuli.py
import pandas as pd
import os
import csv

# --- Configuration ---
# Assumes this script is run from 'llm-poverty-of-stimulus/llm_pos/wilcox/'
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Input file: llm-poverty-of-stimulus/llm_pos/wilcox/data/processed/lan_data_full.csv
INPUT_LAN_CSV = os.path.join(CURRENT_SCRIPT_DIR, "data", "processed", "lan_data_full.csv")

# Output file path
OUTPUT_LONG_FORMAT_CSV = os.path.join(CURRENT_SCRIPT_DIR, "data", "processed", "lan_data_long_format.csv")

# Define the sentence type for this dataset (e.g., "lan_parasitic_gap", "lan_atb_movement")
# You might run this script multiple times if Lan's data for different phenomena are in separate files,
# or if one file contains multiple types distinguishable by some other means (not covered here).
DEFAULT_SENTENCE_TYPE = "lan_parasitic_gap" # Adjust as needed

# Define the header for the output CSV file
OUTPUT_CSV_HEADER = ["sentence_type", "item_id", "condition", "full_sentence"]

def transform_lan_data_to_long_format(input_filepath, output_filepath, sentence_type_label):
    """
    Transforms Lan et al.'s wide format data (one item per row, conditions as columns)
    to a long format (one sentence per row with type, item_id, condition, sentence).
    """
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at '{os.path.abspath(input_filepath)}'")
        return

    output_rows = []

    try:
        print(f"Reading wide format data from: {input_filepath}")
        wide_df = pd.read_csv(input_filepath)
        
        if wide_df.empty:
            print(f"Warning: Input file '{input_filepath}' is empty.")
            return

        print(f"Processing {len(wide_df)} items (rows) from input file...")

        # The column headers of the input CSV are assumed to be the condition names
        # e.g., "PLUS_FILLER_PLUS_GAP", "MINUS_FILLER_PLUS_GAP", etc.
        condition_columns = wide_df.columns.tolist()

        for item_idx, row in wide_df.iterrows():
            item_id = item_idx + 1 # Simple 1-based item ID from row number

            for condition_name in condition_columns:
                full_sentence = row[condition_name]

                if pd.isna(full_sentence) or not isinstance(full_sentence, str) or not full_sentence.strip():
                    # print(f"  Warning: Skipping empty or non-string sentence for item {item_id}, condition '{condition_name}'.")
                    continue
                
                # Clean up potential leading/trailing whitespace from sentence
                full_sentence_cleaned = full_sentence.strip()
                
                # Add a period if it doesn't end with one (optional, but consistent with reconstruct_from_tokens.py)
                if not full_sentence_cleaned.endswith(('.', '?', '!')):
                    full_sentence_cleaned += " ."


                output_rows.append({
                    "sentence_type": sentence_type_label,
                    "item_id": str(item_id), # Keep item_id as string for consistency
                    "condition": condition_name,
                    "full_sentence": full_sentence_cleaned
                })
        
        if not output_rows:
            print("No sentences were extracted. Output file will not be created.")
            return

        # Write to output CSV
        output_df = pd.DataFrame(output_rows, columns=OUTPUT_CSV_HEADER)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as e:
                print(f"Error creating output directory {output_dir}: {e}")
                return

        output_df.to_csv(output_filepath, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"\nSuccessfully transformed {len(wide_df)} items into {len(output_df)} sentence rows.")
        print(f"Long format data saved to: {os.path.abspath(output_filepath)}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {os.path.abspath(input_filepath)}")
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_filepath}' is empty or not a valid CSV.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # You might want to make INPUT_LAN_CSV, OUTPUT_LONG_FORMAT_CSV, 
    # and DEFAULT_SENTENCE_TYPE command-line arguments for more flexibility
    # if you have multiple Lan et al. files for different sentence types.
    
    # For now, using the configured defaults:
    print(f"Input Lan et al. data: {os.path.abspath(INPUT_LAN_CSV)}")
    print(f"Output long format data: {os.path.abspath(OUTPUT_LONG_FORMAT_CSV)}")
    print(f"Assigned sentence_type: {DEFAULT_SENTENCE_TYPE}")
    
    transform_lan_data_to_long_format(INPUT_LAN_CSV, OUTPUT_LONG_FORMAT_CSV, DEFAULT_SENTENCE_TYPE)