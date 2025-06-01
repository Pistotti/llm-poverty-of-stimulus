# extract_lan_critical_surprisals.py
import pandas as pd
import os
import math

# --- Configuration ---
# Assume this script is in a directory like llm_pos/wilcox/
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define base paths relative to this script's location
# Adjust these if your script is elsewhere or data is structured differently
# Assuming 'data/processed/' and 'tims_results/' are subdirectories 
# relative to where this script is located.
DATA_PROCESSED_DIR = os.path.join(CURRENT_SCRIPT_DIR, "data", "processed")
BPE_RESULTS_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results")

MODEL_NAME = "gpt2" # Used to form the surprisal column name

# --- INPUT FILES ---
# This is your file with specific critical words for LAN sentences
CRITICAL_REGION_DEFINITIONS_CSV_BASENAME = "lan_critical_region.csv"
CRITICAL_REGION_DEFINITIONS_CSV = os.path.join(DATA_PROCESSED_DIR, CRITICAL_REGION_DEFINITIONS_CSV_BASENAME)

# This is the BPE-level (or word-level, based on your example) surprisal CSV
# It should contain pre-calculated surprisals for tokens/words.
MASTER_SURPRISALS_CSV_BASENAME = f"master_stimuli_list_surprisals_{MODEL_NAME}.csv"
MASTER_SURPRISALS_CSV = os.path.join(BPE_RESULTS_INPUT_DIR, MASTER_SURPRISALS_CSV_BASENAME)

# --- OUTPUT FILE ---
# Where this script saves its output
EXTRACTED_LAN_RESULTS_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results") # Or a new directory
OUTPUT_CSV_BASENAME = f"lan_extracted_critical_surprisals_{MODEL_NAME}.csv"
EXTRACTED_LAN_OUTPUT_CSV = os.path.join(EXTRACTED_LAN_RESULTS_OUTPUT_DIR, OUTPUT_CSV_BASENAME)

# --- Column Names ---
# Columns in CRITICAL_REGION_DEFINITIONS_CSV
ITEM_COL_CR = "item"
CONDITION_COL_CR = "condition"
CRITICAL_WORD_COL_CR = "critical_region" # This column contains the target word text

# Columns in MASTER_SURPRISALS_CSV
SOURCE_DOC_COL_MS = "source_doc_name" # Expected to be like 'lan_parasitic_gap'
ITEM_COL_MS = "item"
CONDITION_COL_MS = "condition"
FULL_SENTENCE_COL_MS = "full_sentence_text"
TOKEN_STRING_COL_MS = "bpe_token_str" # This should match the CRITICAL_WORD_COL_CR
SURPRISAL_COL_MS_PATTERN = f"surprisal_bits_{MODEL_NAME}" # e.g., surprisal_bits_gpt2
# Optional: if you need to disambiguate multiple occurrences of the same word
# TOKEN_INDEX_COL_MS = "bpe_token_index" # Example showed 11, 12

def main():
    print(f"--- LAN Critical Region Surprisal Extraction for {MODEL_NAME} ---")

    if not os.path.exists(EXTRACTED_LAN_RESULTS_OUTPUT_DIR):
        os.makedirs(EXTRACTED_LAN_RESULTS_OUTPUT_DIR)
        print(f"Created output directory: {EXTRACTED_LAN_RESULTS_OUTPUT_DIR}")

    # --- File Existence Checks ---
    input_files_to_check = {
        "Critical Region Definitions CSV": CRITICAL_REGION_DEFINITIONS_CSV,
        "Master Surprisals CSV": MASTER_SURPRISALS_CSV,
    }
    missing_files = False
    for name, fpath in input_files_to_check.items():
        if not os.path.exists(fpath):
            print(f"Critical Error: Input file '{name}' not found at '{fpath}'")
            missing_files = True
    if missing_files:
        print("Please ensure all required input files are present and paths are correct.")
        return

    # --- Load Data ---
    print("Loading data...")
    try:
        critical_defs_df = pd.read_csv(CRITICAL_REGION_DEFINITIONS_CSV)
        master_surprisals_df = pd.read_csv(MASTER_SURPRISALS_CSV)
    except FileNotFoundError as e:
        print(f"Critical Error: Could not load one of the input CSV files. {e}")
        return
    except Exception as e:
        print(f"Critical Error: Error loading input CSV files. {e}")
        return
    
    print(f"Loaded {len(critical_defs_df)} definitions from {CRITICAL_REGION_DEFINITIONS_CSV_BASENAME}")
    print(f"Loaded {len(master_surprisals_df)} rows from {MASTER_SURPRISALS_CSV_BASENAME}")

    # --- Validate Columns ---
    # Validate critical_defs_df
    expected_cr_cols = [ITEM_COL_CR, CONDITION_COL_CR, CRITICAL_WORD_COL_CR]
    for col in expected_cr_cols:
        if col not in critical_defs_df.columns:
            print(f"Critical Error: Column '{col}' missing from {CRITICAL_REGION_DEFINITIONS_CSV_BASENAME}.")
            return
            
    # Validate master_surprisals_df
    expected_ms_cols = [SOURCE_DOC_COL_MS, ITEM_COL_MS, CONDITION_COL_MS, FULL_SENTENCE_COL_MS, TOKEN_STRING_COL_MS, SURPRISAL_COL_MS_PATTERN]
    for col in expected_ms_cols:
        if col not in master_surprisals_df.columns:
            # For surprisal column, it's a pattern, check if any column matches
            if col == SURPRISAL_COL_MS_PATTERN and not any(c.startswith("surprisal_bits_") for c in master_surprisals_df.columns):
                 print(f"Critical Error: Surprisal column like '{SURPRISAL_COL_MS_PATTERN}' missing from {MASTER_SURPRISALS_CSV_BASENAME}.")
                 print(f"Available columns: {master_surprisals_df.columns.tolist()}")
                 return
            elif col != SURPRISAL_COL_MS_PATTERN : # only print error if it's not the pattern and missing
                 print(f"Critical Error: Column '{col}' missing from {MASTER_SURPRISALS_CSV_BASENAME}.")
                 return
    
    # Determine the actual surprisal column name to use from master_surprisals_df
    actual_surprisal_col_ms = ""
    if SURPRISAL_COL_MS_PATTERN in master_surprisals_df.columns:
        actual_surprisal_col_ms = SURPRISAL_COL_MS_PATTERN
    else: # Fallback to first column starting with "surprisal_bits_"
        potential_s_cols = [c for c in master_surprisals_df.columns if c.startswith("surprisal_bits_")]
        if potential_s_cols:
            actual_surprisal_col_ms = potential_s_cols[0]
            print(f"Info: Using surprisal column '{actual_surprisal_col_ms}' from {MASTER_SURPRISALS_CSV_BASENAME}.")
        else: # This case should have been caught above, but as a safeguard
            print(f"Critical Error: No suitable surprisal column found in {MASTER_SURPRISALS_CSV_BASENAME}.")
            return


    # --- Prepare Data for Merging/Lookup ---
    # Convert join keys to string to ensure consistent matching
    critical_defs_df[ITEM_COL_CR] = critical_defs_df[ITEM_COL_CR].astype(str)
    critical_defs_df[CONDITION_COL_CR] = critical_defs_df[CONDITION_COL_CR].astype(str)
    critical_defs_df[CRITICAL_WORD_COL_CR] = critical_defs_df[CRITICAL_WORD_COL_CR].astype(str).str.strip()

    master_surprisals_df[ITEM_COL_MS] = master_surprisals_df[ITEM_COL_MS].astype(str)
    master_surprisals_df[CONDITION_COL_MS] = master_surprisals_df[CONDITION_COL_MS].astype(str)
    master_surprisals_df[TOKEN_STRING_COL_MS] = master_surprisals_df[TOKEN_STRING_COL_MS].astype(str).str.strip()
    # Assuming SOURCE_DOC_COL_MS is mostly 'lan_parasitic_gap' for these items
    # If there are other source_docs for LAN items, this needs to be handled or filtered.
    # For now, we will rely on item, condition, and critical word string.

    extracted_results = []

    print(f"\nProcessing critical region definitions...")
    for index, cr_row in critical_defs_df.iterrows():
        target_item = cr_row[ITEM_COL_CR]
        target_condition = cr_row[CONDITION_COL_CR]
        target_critical_word = cr_row[CRITICAL_WORD_COL_CR]

        # Find the matching entry in the master surprisals file
        # This assumes that the (item, condition, critical_word_string) uniquely identifies
        # the critical word's entry in master_surprisals_df.
        # We also assume a dominant source_doc_name for LAN data, e.g., 'lan_parasitic_gap'
        # If multiple source_doc_names could apply, the filter might need adjustment.
        
        # Filter for the specific sentence and the specific critical word token
        match_df = master_surprisals_df[
            (master_surprisals_df[ITEM_COL_MS] == target_item) &
            (master_surprisals_df[CONDITION_COL_MS] == target_condition) &
            (master_surprisals_df[TOKEN_STRING_COL_MS] == target_critical_word) &
            (master_surprisals_df[SOURCE_DOC_COL_MS].str.contains("lan", case=False, na=False)) # Heuristic for LAN data
        ]
        
        if match_df.empty:
            print(f"  Warning: No match found in '{MASTER_SURPRISALS_CSV_BASENAME}' for: "
                  f"Item='{target_item}', Condition='{target_condition}', CriticalWord='{target_critical_word}' (LAN source). Skipping.")
            extracted_results.append({
                "sentence_type": "N/A (No BPE match)",
                "item_id": target_item,
                "condition": target_condition,
                "critical_region_name": target_critical_word, # Use the word as name
                "critical_region_text": target_critical_word,
                "aggregated_surprisal_bits": math.nan,
                "original_full_sentence": "N/A (No BPE match)"
            })
            continue
        
        if len(match_df) > 1:
            # This case needs careful handling. If the critical word appears multiple times
            # and lan_critical_region.csv doesn't specify which one (e.g., via an index),
            # we might be picking the wrong one or an arbitrary one.
            # Your example output implies a unique mapping.
            print(f"  Warning: Multiple matches ({len(match_df)}) found for: "
                  f"Item='{target_item}', Condition='{target_condition}', CriticalWord='{target_critical_word}'. "
                  f"Using the first match. Surprisals for all matches: {match_df[actual_surprisal_col_ms].tolist()}")
            # To be robust, you might need an additional key if this happens often,
            # e.g., the bpe_token_index from your example if it's reliable.
            # For now, take the first one as per simplified approach.
        
        matched_row = match_df.iloc[0] # Take the first match

        sentence_type = matched_row[SOURCE_DOC_COL_MS]
        full_sentence = matched_row[FULL_SENTENCE_COL_MS]
        extracted_surprisal = matched_row[actual_surprisal_col_ms]

        if pd.isna(extracted_surprisal):
            print(f"  Warning: Extracted surprisal is NaN for: "
                  f"Item='{target_item}', Condition='{target_condition}', CriticalWord='{target_critical_word}'.")
            extracted_surprisal = math.nan # Ensure it's a float NaN

        extracted_results.append({
            "sentence_type": sentence_type,
            "item_id": target_item,
            "condition": target_condition,
            "critical_region_name": target_critical_word, # Using the word itself as the 'name'
            "critical_region_text": target_critical_word,
            "aggregated_surprisal_bits": extracted_surprisal,
            "original_full_sentence": full_sentence
        })

    if not extracted_results:
        print("\nNo results were extracted. Check input files and matching logic.")
        return

    output_df = pd.DataFrame(extracted_results)
    
    # Ensure correct order of columns as requested
    output_columns_ordered = [
        "sentence_type", "item_id", "condition", 
        "critical_region_name", "critical_region_text", 
        "aggregated_surprisal_bits", "original_full_sentence"
    ]
    # Filter to only include desired columns, in order, handling missing ones gracefully
    final_output_df = pd.DataFrame()
    for col in output_columns_ordered:
        if col in output_df.columns:
            final_output_df[col] = output_df[col]
        else:
            print(f"Warning: Expected output column '{col}' was not generated. Filling with NaN.")
            final_output_df[col] = math.nan


    try:
        final_output_df.to_csv(EXTRACTED_LAN_OUTPUT_CSV, index=False, float_format='%.8f')
        print(f"\nSuccessfully extracted LAN critical surprisals to: {EXTRACTED_LAN_OUTPUT_CSV}")
        print(f"Generated {len(final_output_df)} rows.")
        print(f"First few results:\n{final_output_df.head()}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")

if __name__ == "__main__":
    main()
