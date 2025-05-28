import csv
import os
import pandas as pd

# --- Configuration ---
# Script is in: llm-poverty-of-stimulus/llm_pos/wilcox/
# Current working directory when running the script will be llm-poverty-of-stimulus/llm_pos/wilcox/
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is robust

# Input file: llm-poverty-of-stimulus/llm_pos/wilcox/data/processed/test_items.csv
INPUT_TOKEN_FILE = os.path.join(CURRENT_SCRIPT_DIR, "data", "processed", "test_items.csv")

# Output file: llm-poverty-of-stimulus/llm_pos/wilcox/data/processed/master_stimuli_list.csv
# (Placing output in the same 'processed' directory as the input)
OUTPUT_SENTENCE_FILE = os.path.join(CURRENT_SCRIPT_DIR, "data", "processed", "master_stimuli_list.csv")

# Define the header for the output CSV file
OUTPUT_CSV_HEADER = ["sentence_type", "item_id", "condition", "full_sentence"]

# Columns from input to group by (ensure these match your test_items.csv headers)
# 'test' column from input will be 'sentence_type' in output
GROUPING_COLUMNS = ["test", "item", "condition"]
TOKEN_COLUMN = "token" # Column containing the actual word/token

def reconstruct_sentences(input_filepath, output_filepath):
    """
    Reads a tokenized CSV file, reconstructs full sentences,
    and writes them to a new CSV with specified metadata.
    """
    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at '{os.path.abspath(input_filepath)}'")
        return

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return
            
    try:
        # Read the tokenized data using pandas for easier grouping
        df = pd.read_csv(input_filepath)
        print(f"Successfully read '{input_filepath}'. Processing {len(df)} token rows...")

        # Ensure required columns exist
        required_input_cols = GROUPING_COLUMNS + [TOKEN_COLUMN]
        missing_cols = [col for col in required_input_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Input file '{input_filepath}' is missing required columns: {', '.join(missing_cols)}")
            print(f"Found columns: {df.columns.tolist()}")
            return

        # Assuming tokens are already correctly ordered within each group in the input CSV.
        # If your test_items.csv has a token index column (e.g., 'token_pos' or 'token_id_in_sentence'),
        # you should sort by it here to ensure correct sentence reconstruction:
        # Example: df.sort_values(GROUPING_COLUMNS + ['token_pos'], inplace=True)

        # Convert relevant columns to string to ensure consistent grouping
        for col in GROUPING_COLUMNS:
            if col in df.columns: # Check if column exists before astype
                 df[col] = df[col].astype(str)
        if TOKEN_COLUMN in df.columns: # Check if column exists
            df[TOKEN_COLUMN] = df[TOKEN_COLUMN].astype(str)


        def join_tokens(tokens_series):
            return " ".join(tokens_series.dropna().astype(str))

        # Using sort=False in groupby to maintain original order of groups if it matters
        # This is important if your test_items.csv groups are already in a specific order you want to preserve
        reconstructed_data = df.groupby(GROUPING_COLUMNS, sort=False, as_index=False)[TOKEN_COLUMN].apply(join_tokens)
        
        # After apply, pandas might create a multi-index if as_index=False wasn't perfect or depending on pandas version.
        # If `reconstructed_data` doesn't have TOKEN_COLUMN directly but it's the result of apply,
        # and GROUPING_COLUMNS are the index, we need to handle it.
        # A more robust way if apply returns a Series:
        if isinstance(reconstructed_data, pd.Series): # If apply returned a Series (common)
            reconstructed_data = reconstructed_data.reset_index()
            # The joined tokens are likely in the last column, often unnamed or named 0 if not specified
            # Let's rename it if needed, assuming it's the column with joined strings
            if TOKEN_COLUMN not in reconstructed_data.columns:
                 # Heuristic: find the column that's not in GROUPING_COLUMNS
                new_token_col_name = [col for col in reconstructed_data.columns if col not in GROUPING_COLUMNS][0]
                reconstructed_data.rename(columns={new_token_col_name: TOKEN_COLUMN}, inplace=True)


        reconstructed_data.rename(columns={"test": "sentence_type", "item": "item_id"}, inplace=True)
        
        reconstructed_data['full_sentence'] = reconstructed_data[TOKEN_COLUMN] + " ."
        
        output_df = reconstructed_data[["sentence_type", "item_id", "condition", "full_sentence"]]

    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_filepath}' is empty.")
        return
    except Exception as e:
        print(f"An error occurred while processing the input file '{input_filepath}': {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return

    if output_df.empty:
        print("No sentences were reconstructed. Output file will not be created.")
        return

    try:
        output_df.to_csv(output_filepath, index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"\nSuccessfully generated '{os.path.abspath(output_filepath)}' with {len(output_df)} sentences.")
    except IOError as e:
        print(f"Error writing to output file '{output_filepath}': {e}")

if __name__ == "__main__":
    # This script is intended to be run when test_items.csv exists.
    # The dummy creation logic is removed as paths are now specific.
    if not os.path.exists(INPUT_TOKEN_FILE):
        print(f"Error: Main input file '{INPUT_TOKEN_FILE}' not found at expected location: {os.path.abspath(INPUT_TOKEN_FILE)}")
        print("Please ensure the file exists and the path is correct in the script.")
    else:
        reconstruct_sentences(INPUT_TOKEN_FILE, OUTPUT_SENTENCE_FILE)