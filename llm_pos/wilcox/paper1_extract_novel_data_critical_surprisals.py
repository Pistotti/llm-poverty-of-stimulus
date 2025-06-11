# llm-poverty-of-stimulus/llm_pos/wilcox/extract_novel_data_critical_surprisals.py
import pandas as pd
import os
import math
# NOTE: This script, as per your request to follow the 'lan_data based script',
# does NOT use a Hugging Face tokenizer for matching critical regions.
# It relies on direct string match of CR_TEXT with BPE_TOKEN_STR.

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED_DIR = os.path.join(CURRENT_SCRIPT_DIR, "data", "processed")
BPE_RESULTS_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results")
AGGREGATED_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "aggregated")

MODEL_NAME = "gpt2"
NOVEL_SENTENCE_TYPE_IDENTIFIER = "subject_gap" 

CRITICAL_REGION_DEFINITIONS_CSV_BASENAME = "novel_data_critical_regions.csv"
CRITICAL_REGION_DEFINITIONS_CSV = os.path.join(DATA_PROCESSED_DIR, CRITICAL_REGION_DEFINITIONS_CSV_BASENAME)

MASTER_SURPRISALS_CSV_BASENAME = f"master_stimuli_list_surprisals_{MODEL_NAME}.csv"
MASTER_SURPRISALS_CSV = os.path.join(BPE_RESULTS_INPUT_DIR, MASTER_SURPRISALS_CSV_BASENAME)

OUTPUT_CSV_BASENAME = f"novel_data_extracted_critical_surprisals_{MODEL_NAME}.csv"
EXTRACTED_NOVEL_OUTPUT_CSV = os.path.join(AGGREGATED_OUTPUT_DIR, OUTPUT_CSV_BASENAME)

ITEM_COL_CR = "item_id" 
CONDITION_COL_CR = "condition"
CRITICAL_WORD_COL_CR = "critical_region"

SOURCE_DOC_COL_MS = "source_doc_name"
ITEM_COL_MS = "item"                 
CONDITION_COL_MS = "condition"
FULL_SENTENCE_COL_MS = "full_sentence_text"
TOKEN_STRING_COL_MS = "bpe_token_str"     
SURPRISAL_COL_MS_PATTERN = f"surprisal_bits_{MODEL_NAME}"

def main():
    print(f"--- Novel Data ('{NOVEL_SENTENCE_TYPE_IDENTIFIER}') Critical Region Surprisal Extraction for {MODEL_NAME} ---")

    if not os.path.exists(AGGREGATED_OUTPUT_DIR):
        os.makedirs(AGGREGATED_OUTPUT_DIR)
        print(f"Created output directory: {AGGREGATED_OUTPUT_DIR}")

    input_files_to_check = {
        "Critical Region Definitions CSV": CRITICAL_REGION_DEFINITIONS_CSV,
        "Master Surprisals CSV": MASTER_SURPRISALS_CSV,
    }
    missing_files = False
    for name, fpath in input_files_to_check.items():
        if not os.path.exists(fpath):
            print(f"Critical Error: Input file '{name}' not found at '{os.path.abspath(fpath)}'")
            missing_files = True
    if missing_files: return

    print("Loading data...")
    try:
        critical_defs_df = pd.read_csv(CRITICAL_REGION_DEFINITIONS_CSV)
        master_surprisals_df = pd.read_csv(MASTER_SURPRISALS_CSV)
    except Exception as e: print(f"Critical Error: Error loading input CSV files. {e}"); return
    
    print(f"Loaded {len(critical_defs_df)} critical region definitions from {os.path.basename(CRITICAL_REGION_DEFINITIONS_CSV)}")
    print(f"Loaded {len(master_surprisals_df)} BPE surprisal rows from {os.path.basename(MASTER_SURPRISALS_CSV)}")

    expected_cr_cols = [ITEM_COL_CR, CONDITION_COL_CR, CRITICAL_WORD_COL_CR]
    for col in expected_cr_cols:
        if col not in critical_defs_df.columns:
            print(f"Critical Error: Column '{col}' missing from {os.path.basename(CRITICAL_REGION_DEFINITIONS_CSV)}. Found headers: {critical_defs_df.columns.tolist()}"); return
            
    actual_surprisal_col_ms = SURPRISAL_COL_MS_PATTERN
    if actual_surprisal_col_ms not in master_surprisals_df.columns:
        print(f"Critical Error: Surprisal column '{actual_surprisal_col_ms}' not found in {MASTER_SURPRISALS_CSV_BASENAME}. Avail: {master_surprisals_df.columns.tolist()}"); return

    expected_ms_cols_subset = [SOURCE_DOC_COL_MS, ITEM_COL_MS, CONDITION_COL_MS, TOKEN_STRING_COL_MS, actual_surprisal_col_ms, FULL_SENTENCE_COL_MS]
    for col in expected_ms_cols_subset:
        if col not in master_surprisals_df.columns: print(f"Critical Error: Column '{col}' missing from {MASTER_SURPRISALS_CSV_BASENAME}. Avail: {master_surprisals_df.columns.tolist()}"); return

    if ITEM_COL_CR != 'item' and ITEM_COL_CR in critical_defs_df.columns:
        critical_defs_df.rename(columns={ITEM_COL_CR: 'item'}, inplace=True)
    elif 'item' not in critical_defs_df.columns and ITEM_COL_CR not in critical_defs_df.columns:
         print(f"Critical Error: Column for item ID ('item' or '{ITEM_COL_CR}') not found in {os.path.basename(CRITICAL_REGION_DEFINITIONS_CSV)}."); return

    critical_defs_df['item'] = critical_defs_df['item'].astype(str)
    critical_defs_df[CONDITION_COL_CR] = critical_defs_df[CONDITION_COL_CR].astype(str)
    critical_defs_df[CRITICAL_WORD_COL_CR] = critical_defs_df[CRITICAL_WORD_COL_CR].astype(str).str.strip()

    master_surprisals_df[SOURCE_DOC_COL_MS] = master_surprisals_df[SOURCE_DOC_COL_MS].astype(str)
    master_surprisals_df[ITEM_COL_MS] = master_surprisals_df[ITEM_COL_MS].astype(str)
    master_surprisals_df[CONDITION_COL_MS] = master_surprisals_df[CONDITION_COL_MS].astype(str)
    master_surprisals_df[TOKEN_STRING_COL_MS] = master_surprisals_df[TOKEN_STRING_COL_MS].astype(str).str.strip()
    
    print(f"\nFiltering master surprisals for sentence_type '{NOVEL_SENTENCE_TYPE_IDENTIFIER}' entries...")
    novel_data_bpe_df = master_surprisals_df[
        master_surprisals_df[SOURCE_DOC_COL_MS] == NOVEL_SENTENCE_TYPE_IDENTIFIER
    ].copy()
    
    print(f"Found {len(novel_data_bpe_df)} BPE rows for '{NOVEL_SENTENCE_TYPE_IDENTIFIER}'.")

    if novel_data_bpe_df.empty:
        print(f"Warning: No BPE rows for '{NOVEL_SENTENCE_TYPE_IDENTIFIER}' found. Output will be empty.")
        return
        
    print(f"Attempting to merge critical definitions with filtered BPE surprisal rows for '{NOVEL_SENTENCE_TYPE_IDENTIFIER}'...")
    
    merged_df = pd.merge(
        critical_defs_df, 
        novel_data_bpe_df,
        left_on=['item', CONDITION_COL_CR, CRITICAL_WORD_COL_CR],
        right_on=[ITEM_COL_MS, CONDITION_COL_MS, TOKEN_STRING_COL_MS],
        how='left'
    )
    print(f"Merge completed. Resulting table has {len(merged_df)} rows.")
    
    # --- MODIFIED/IMPROVED PART for output_df construction ---
    if not merged_df.empty:
        data_for_output = {
            "sentence_type": NOVEL_SENTENCE_TYPE_IDENTIFIER, # Assign the known type for all rows
            "item_id": merged_df['item'],  # 'item' from critical_defs_df (after potential rename)
            "condition": merged_df[CONDITION_COL_CR],
            "critical_region_text": merged_df[CRITICAL_WORD_COL_CR],
            "aggregated_surprisal_bits": merged_df.get(actual_surprisal_col_ms, pd.Series([math.nan] * len(merged_df))), # Get surprisal, default to NaN if column missing from a row after left merge
            "original_full_sentence": merged_df.get(FULL_SENTENCE_COL_MS, pd.Series(["N/A"] * len(merged_df))) # Get full sentence
        }
        output_df = pd.DataFrame(data_for_output)
        
        # Ensure specific column order for output
        final_output_cols = ["sentence_type", "item_id", "condition", "critical_region_text", 
                             "aggregated_surprisal_bits", "original_full_sentence"]
        output_df = output_df[final_output_cols] # Reorder to desired final output
    else:
        # If merged_df is empty, create an empty DataFrame with correct columns
        output_df = pd.DataFrame(columns=["sentence_type", "item_id", "condition", "critical_region_text", 
                                          "aggregated_surprisal_bits", "original_full_sentence"])
    # --- END MODIFIED/IMPROVED PART ---

    num_unmatched = output_df["aggregated_surprisal_bits"].isnull().sum()
    if num_unmatched > 0 and not merged_df.empty : # only print warning if there were rows to process
        print(f"WARNING: {num_unmatched} out of {len(critical_defs_df)} critical region definitions "
              f"did NOT find an exact match for '{CRITICAL_WORD_COL_CR}' in the 'bpe_token_str' column "
              f"of the BPE surprisal data (or the surprisal was already NaN).")
        print("  This means the critical word might be split into multiple BPEs, or there's a text mismatch (e.g., 'soon' vs 'Ġsoon').")
        print("  For such cases, 'aggregated_surprisal_bits' will be NaN.")

    if not output_df.empty:
        try:
            output_df.to_csv(EXTRACTED_NOVEL_OUTPUT_CSV, index=False, float_format='%.8f')
            print(f"\nSuccessfully extracted surprisals to {os.path.abspath(EXTRACTED_NOVEL_OUTPUT_CSV)}")
            print(f"Generated {len(output_df)} rows.")
            print(f"First few results:\n{output_df.head()}")
        except Exception as e:
            print(f"Error saving output CSV: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo results were extracted (output DataFrame is empty or only headers).")

if __name__ == "__main__":
    main()