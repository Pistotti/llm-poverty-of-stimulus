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

    if not os.path.exists(EXTRACTED_LAN_RESULTS_OUTPUT_DIR): # Ensure output dir from config
        os.makedirs(EXTRACTED_LAN_RESULTS_OUTPUT_DIR)
        print(f"Created output directory: {EXTRACTED_LAN_RESULTS_OUTPUT_DIR}")

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
        print("Please ensure all required input files are present.")
        return

    print("Loading data...")
    try:
        critical_defs_df = pd.read_csv(CRITICAL_REGION_DEFINITIONS_CSV)
        master_surprisals_df = pd.read_csv(MASTER_SURPRISALS_CSV)
    except Exception as e:
        print(f"Critical Error: Error loading input CSV files. {e}")
        return
    
    print(f"Loaded {len(critical_defs_df)} definitions from {os.path.basename(CRITICAL_REGION_DEFINITIONS_CSV)}")
    print(f"Loaded {len(master_surprisals_df)} rows from {os.path.basename(MASTER_SURPRISALS_CSV)}")

    # --- Validate Columns & Prepare Data (similar to your original script) ---
    expected_cr_cols = [ITEM_COL_CR, CONDITION_COL_CR, CRITICAL_WORD_COL_CR]
    for col in expected_cr_cols:
        if col not in critical_defs_df.columns:
            print(f"Critical Error: Column '{col}' missing from {os.path.basename(CRITICAL_REGION_DEFINITIONS_CSV)}."); return
            
    actual_surprisal_col_ms = ""
    if SURPRISAL_COL_MS_PATTERN in master_surprisals_df.columns:
        actual_surprisal_col_ms = SURPRISAL_COL_MS_PATTERN
    else:
        potential_s_cols = [c for c in master_surprisals_df.columns if c.startswith("surprisal_bits_")]
        if potential_s_cols: actual_surprisal_col_ms = potential_s_cols[0]
        else: print(f"Critical Error: No surprisal column like '{SURPRISAL_COL_MS_PATTERN}' found in {MASTER_SURPRISALS_CSV_BASENAME}. Avail: {master_surprisals_df.columns.tolist()}"); return

    expected_ms_cols = [SOURCE_DOC_COL_MS, ITEM_COL_MS, CONDITION_COL_MS, FULL_SENTENCE_COL_MS, TOKEN_STRING_COL_MS, actual_surprisal_col_ms]
    for col in expected_ms_cols:
        if col not in master_surprisals_df.columns: print(f"Critical Error: Column '{col}' missing from {MASTER_SURPRISALS_CSV_BASENAME}."); return

    critical_defs_df[ITEM_COL_CR] = critical_defs_df[ITEM_COL_CR].astype(str)
    critical_defs_df[CONDITION_COL_CR] = critical_defs_df[CONDITION_COL_CR].astype(str)
    critical_defs_df[CRITICAL_WORD_COL_CR] = critical_defs_df[CRITICAL_WORD_COL_CR].astype(str).str.strip()

    master_surprisals_df[ITEM_COL_MS] = master_surprisals_df[ITEM_COL_MS].astype(str)
    master_surprisals_df[CONDITION_COL_MS] = master_surprisals_df[CONDITION_COL_MS].astype(str)
    master_surprisals_df[TOKEN_STRING_COL_MS] = master_surprisals_df[TOKEN_STRING_COL_MS].astype(str).str.strip()
    master_surprisals_df[SOURCE_DOC_COL_MS] = master_surprisals_df[SOURCE_DOC_COL_MS].astype(str) # for .str.contains

    # --- MODIFIED SECTION: Replace loop with merge ---
    print(f"\nFiltering master surprisals for LAN-related entries...")
    # Pre-filter master_surprisals_df for LAN entries to make the merge more targeted
    lan_master_df = master_surprisals_df[
        master_surprisals_df[SOURCE_DOC_COL_MS].str.contains("lan", case=False, na=False)
    ].copy() # Use .copy() to avoid SettingWithCopyWarning on potential later modifications
    
    print(f"Found {len(lan_master_df)} LAN-related rows in master surprisals.")

    if lan_master_df.empty:
        print("Warning: No LAN-related rows found in master surprisals after filtering. Output will likely be empty or have many NaNs.")
        # Fallback: create an empty df with expected columns for schema consistency if needed later
        # For now, we'll let merge produce NaNs for non-matches

    print(f"Attempting to merge {len(critical_defs_df)} critical definitions with {len(lan_master_df)} filtered LAN master surprisal rows...")
    
    # Define columns for merging
    left_on_cols = [ITEM_COL_CR, CONDITION_COL_CR, CRITICAL_WORD_COL_CR]
    right_on_cols = [ITEM_COL_MS, CONDITION_COL_MS, TOKEN_STRING_COL_MS]
    
    # Select only necessary columns from lan_master_df to avoid large redundant data
    cols_to_bring_from_master = right_on_cols + [SOURCE_DOC_COL_MS, FULL_SENTENCE_COL_MS, actual_surprisal_col_ms]
    # Ensure uniqueness in case merge keys are also in other_cols_from_master
    unique_cols_to_bring_from_master = list(pd.Series(cols_to_bring_from_master).unique())


    merged_df = pd.merge(
        critical_defs_df,
        lan_master_df[unique_cols_to_bring_from_master],
        left_on=left_on_cols,
        right_on=right_on_cols,
        how='left'  # Keep all definitions from critical_defs_df
    )
    print(f"Merge completed. Resulting table before duplicate handling has {len(merged_df)} rows.")

    # Handle cases where one definition might match multiple master rows (e.g. if a critical word appears >1 times in a sentence)
    # We want to keep only the first match for each original critical definition row.
    # The 'left_on_cols' are the keys from critical_defs_df that define a unique definition.
    merged_df.drop_duplicates(subset=left_on_cols, keep='first', inplace=True)
    print(f"After dropping potential duplicates (keeping first match per definition), table has {len(merged_df)} rows.")

    # Prepare the final output DataFrame
    output_df = pd.DataFrame()
    output_df["source_doc_name"] = merged_df[SOURCE_DOC_COL_MS]
    output_df["item"] = merged_df[ITEM_COL_CR] 
    output_df["condition"] = merged_df[CONDITION_COL_CR] 
    output_df["critical_word_text"] = merged_df[CRITICAL_WORD_COL_CR] 
    output_df["aggregated_surprisal_bits"] = merged_df[actual_surprisal_col_ms]
    # Add full sentence if it was part of your intended output schema (it's in merged_df[FULL_SENTENCE_COL_MS])
    # output_df["original_full_sentence"] = merged_df[FULL_SENTENCE_COL_MS] # Uncomment if needed

    # --- End MODIFIED SECTION ---

    if not output_df.empty:
        # Check for rows where surprisal is NaN, indicating no match was found in master file for a definition
        num_unmatched = output_df["aggregated_surprisal_bits"].isnull().sum()
        if num_unmatched > 0:
            print(f"Warning: {num_unmatched} out of {len(critical_defs_df)} critical definitions did not find a match "
                  f"or had NaN surprisal in the master surprisals file (surprisal will be NaN).")
        
        try:
            output_df.to_csv(EXTRACTED_LAN_OUTPUT_CSV, index=False, float_format='%.8f') # EXTRACTED_LAN_OUTPUT_CSV from config
            print(f"\nSuccessfully extracted surprisals to {os.path.abspath(EXTRACTED_LAN_OUTPUT_CSV)}")
            print(f"Generated {len(output_df)} rows.")
            print(f"First few results:\n{output_df.head()}")
        except Exception as e:
            print(f"Error saving output CSV: {e}")
    else:
        print("\nNo results were extracted (merged DataFrame was empty).")

if __name__ == "__main__":
    main()