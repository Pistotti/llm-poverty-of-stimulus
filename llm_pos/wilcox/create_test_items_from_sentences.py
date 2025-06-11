# llm-poverty-of-stimulus/llm_pos/wilcox/create_test_items_from_sentences.py
import pandas as pd
import os
import re

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED_DIR = os.path.join(CURRENT_SCRIPT_DIR, "data", "processed")

# Input 1: The CSV with your full sentences (e.g., 320 rows for the full paradigm)
SOURCE_SENTENCES_CSV = os.path.join(DATA_PROCESSED_DIR, "novel_pg_full_paradigm.csv")

# Input 2: The CSV defining which text is the critical region for each sentence
CRITICAL_REGIONS_DEF_FILE = os.path.join(DATA_PROCESSED_DIR, "novel_pg_full_paradigm_critical_regions.csv")

# Output: The new test_items file with token-per-row format and preliminary region labels
OUTPUT_TEST_ITEMS_CSV = os.path.join(DATA_PROCESSED_DIR, "test_items_novel_pg_full.csv")

def simple_tokenize(sentence_text):
    """
    A simple tokenizer that splits on spaces and handles sentence-final punctuation.
    """
    sentence_text = sentence_text.strip()
    # Ensure period is treated as a separate token if it's attached to the last word
    sentence_text = re.sub(r'(\w)\s*\.$', r'\1 .', sentence_text)
    if not sentence_text.endswith('.'):
        sentence_text += ' .'
    
    # Split by space and filter out any empty strings that might result from multiple spaces
    tokens = [token for token in sentence_text.split(' ') if token]
    return tokens

def main():
    print("--- Generating test_items file from full sentences ---")

    if not os.path.exists(SOURCE_SENTENCES_CSV) or not os.path.exists(CRITICAL_REGIONS_DEF_FILE):
        print(f"Error: One or more input files not found. Check paths.")
        print(f"  - Sentences file exists: {os.path.exists(SOURCE_SENTENCES_CSV)} -> {SOURCE_SENTENCES_CSV}")
        print(f"  - CR Defs file exists: {os.path.exists(CRITICAL_REGIONS_DEF_FILE)} -> {CRITICAL_REGIONS_DEF_FILE}")
        return
        
    print("Loading data...")
    try:
        sentences_df = pd.read_csv(SOURCE_SENTENCES_CSV)
        cr_defs_df = pd.read_csv(CRITICAL_REGIONS_DEF_FILE)
    except Exception as e:
        print(f"Error loading CSV files: {e}"); return
        
    # --- Prepare Data ---
    for col in ['item_id', 'condition']:
        sentences_df[col] = sentences_df[col].astype(str)
        cr_defs_df[col] = cr_defs_df[col].astype(str)

    all_token_rows = []
    print(f"Processing {len(sentences_df)} sentences...")

    for index, row in sentences_df.iterrows():
        sentence_type = row['sentence_type']
        item_id = row['item_id']
        condition = row['condition']
        full_sentence = row['full_sentence']

        tokens = simple_tokenize(full_sentence)
        
        # Get the critical region text for this specific sentence
        # FIX: The cr_defs_df does not have a sentence_type column, so we don't use it in the filter
        cr_def_row = cr_defs_df[
            (cr_defs_df['item_id'] == item_id) &
            (cr_defs_df['condition'] == condition)
        ]
        
        critical_region_tokens = []
        if not cr_def_row.empty:
            critical_region_text = cr_def_row['critical_region_text'].iloc[0]
            # Tokenize the critical region text itself for matching
            critical_region_tokens = str(critical_region_text).split()
        
        cr_start_index = -1
        if critical_region_tokens:
            for i in range(len(tokens) - len(critical_region_tokens) + 1):
                if tokens[i:i+len(critical_region_tokens)] == critical_region_tokens:
                    cr_start_index = i
                    break
        
        for token_idx, token_text in enumerate(tokens):
            region_label = "context" # Default label
            
            # --- Automatic Labeling Logic ---
            if token_text.lower() in ["who", "that"]:
                region_label = "filler_comp"
            elif cr_start_index != -1 and cr_start_index <= token_idx < cr_start_index + len(critical_region_tokens):
                # The condition name tells us the role of the CR
                if "minusG2" in condition:
                    region_label = "g2_filled_object"
                elif "plusG2" in condition:
                    region_label = "g2_gapped_adverbial"
            elif token_text == ".":
                region_label = "punctuation"

            all_token_rows.append({
                "sentence_type": sentence_type,
                "item_id": item_id,
                "condition": condition,
                "token_index": token_idx + 1, # 1-based index
                "token": token_text,
                "region": region_label # Use 'region' as the column header
            })
            
    if not all_token_rows:
        print("No token rows were generated."); return

    output_df = pd.DataFrame(all_token_rows)
    output_df.to_csv(OUTPUT_TEST_ITEMS_CSV, index=False)
    print(f"\nSuccessfully generated {len(output_df)} token rows.")
    print(f"File saved to: {os.path.abspath(OUTPUT_TEST_ITEMS_CSV)}")
    print("\nNext step: You can now run `extract_novel_data_critical_surprisals.py`.")

if __name__ == "__main__":
    main()