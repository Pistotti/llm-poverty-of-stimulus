# llm-poverty-of-stimulus/llm_pos/wilcox/extract_novel_data_critical_surprisals.py
import pandas as pd
import os
import math
from transformers import AutoTokenizer

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED_DIR = os.path.join(CURRENT_SCRIPT_DIR, "data", "processed")
BPE_RESULTS_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results")
AGGREGATED_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "aggregated")

MODEL_NAME = "gpt2"
HF_MODEL_NAME_TOKENIZER = "gpt2"
# This should match the 'sentence_type' in your BPE surprisal file for this dataset
NOVEL_SENTENCE_TYPE_IDENTIFIER = "subject_pg_full"

# --- INPUT FILES ---
# Input 1: BPE surprisals for your new 80-sentence paradigm
BPE_SURPRISALS_INPUT_FILE = os.path.join(BPE_RESULTS_INPUT_DIR, "novel_pg_full_paradigm_surprisals_gpt2.csv")

# Input 2: The tokenized and region-labeled file you will create
WORD_TOKEN_MAP_FILE = os.path.join(DATA_PROCESSED_DIR, "test_items_novel_pg_full.csv")

# Input 3: The file defining the text of the critical region for each sentence
CRITICAL_REGIONS_DEF_FILE = os.path.join(DATA_PROCESSED_DIR, "novel_pg_full_paradigm_critical_regions.csv")

# --- OUTPUT FILE ---
OUTPUT_CSV_FILE = os.path.join(AGGREGATED_OUTPUT_DIR, "novel_pg_full_paradigm_aggregated_surprisals.csv")

# --- Initialize Tokenizer ---
tokenizer = None
try:
    print(f"Loading tokenizer: {HF_MODEL_NAME_TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME_TOKENIZER)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
except Exception as e:
    print(f"Critical Error: Could not load tokenizer '{HF_MODEL_NAME_TOKENIZER}'. Exception: {e}")

def get_word_level_surprisals(current_bpe_df_group, current_word_map_df_group, tokenizer_instance, bpe_surprisal_col):
    """
    Aligns BPE tokens to words from the word map and sums their surprisals.
    """
    if tokenizer_instance is None: return []

    sentence_word_data = []
    bpes_for_sentence = current_bpe_df_group.to_dict('records')
    target_words = current_word_map_df_group['token'].tolist()
    bpe_cursor = 0

    for word_idx, target_word in enumerate(target_words):
        word_bpes = []
        reconstructed = ""
        start_cursor = bpe_cursor
        
        while bpe_cursor < len(bpes_for_sentence):
            word_bpes.append(bpes_for_sentence[bpe_cursor]['bpe_token_str'])
            reconstructed = tokenizer_instance.convert_tokens_to_string(word_bpes).strip()
            bpe_cursor += 1
            if reconstructed.lower() == str(target_word).lower():
                break
        
        if reconstructed.lower() == str(target_word).lower():
            bpe_span = bpes_for_sentence[start_cursor:bpe_cursor]
            word_surprisal = sum(bpe[bpe_surprisal_col] for bpe in bpe_span if pd.notna(bpe[bpe_surprisal_col]))
            if any(pd.isna(bpe[bpe_surprisal_col]) for bpe in bpe_span):
                word_surprisal = math.nan
            sentence_word_data.append({'word_index': word_idx, 'word': target_word, 'surprisal': word_surprisal})
        else:
            bpe_cursor = start_cursor # Backtrack if word not matched
            sentence_word_data.append({'word_index': word_idx, 'word': target_word, 'surprisal': math.nan})
            bpe_cursor += 1 # Try to advance cursor past the failed word to avoid getting stuck

    return sentence_word_data

def main():
    if tokenizer is None: print("Tokenizer not loaded. Exiting."); return
    if not os.path.exists(AGGREGATED_OUTPUT_DIR): os.makedirs(AGGREGATED_OUTPUT_DIR)

    print("Loading input files...")
    try:
        bpe_df = pd.read_csv(BPE_SURPRISALS_INPUT_FILE)
        cr_def_df = pd.read_csv(CRITICAL_REGIONS_DEF_FILE)
        word_map_df = pd.read_csv(WORD_TOKEN_MAP_FILE)
    except FileNotFoundError as e:
        print(f"Error: An input file was not found. {e}"); return
    
    # Prepare dataframes for processing
    bpe_df.rename(columns={'source_doc_name': 'sentence_type', 'item': 'item_id'}, inplace=True, errors='ignore')
    for col in ['sentence_type', 'item_id', 'condition']:
        if col in bpe_df.columns: bpe_df[col] = bpe_df[col].astype(str)
        if col in cr_def_df.columns: cr_def_df[col] = cr_def_df[col].astype(str)
        if col in word_map_df.columns: word_map_df[col] = word_map_df[col].astype(str)
        
    aggregated_results = []
    bpe_surprisal_col_name = f"surprisal_bits_{MODEL_NAME}"
    
    grouped_bpe_data = bpe_df[bpe_df['sentence_type'] == NOVEL_SENTENCE_TYPE_IDENTIFIER].groupby(['item_id', 'condition'])
    print(f"Processing {len(grouped_bpe_data)} unique sentences for type '{NOVEL_SENTENCE_TYPE_IDENTIFIER}'...")

    for (item_id, condition_str), current_sentence_bpes_df in grouped_bpe_data:
        # Get the target words for this sentence from the word map
        current_word_map = word_map_df[
            (word_map_df['item_id'] == item_id) & 
            (word_map_df['condition'] == condition_str) &
            (word_map_df['sentence_type'] == NOVEL_SENTENCE_TYPE_IDENTIFIER)
        ].sort_values(by='token_index')

        if current_word_map.empty:
            print(f"    Warning: No token map found for Item='{item_id}', Cond='{condition_str}'. Skipping.")
            continue

        sentence_word_surprisals = get_word_level_surprisals(current_sentence_bpes_df, current_word_map, tokenizer, bpe_surprisal_col_name)
        if not sentence_word_surprisals:
            print(f"    Warning: BPE-to-word alignment failed for Item='{item_id}', Cond='{condition_str}'. Skipping.")
            continue

        cr_text_row = cr_def_df[(cr_def_df['item_id'] == item_id) & (cr_def_df['condition'] == condition_str)]
        if cr_text_row.empty: continue
            
        target_cr_text = cr_text_row['critical_region_text'].iloc[0]
        target_cr_words = str(target_cr_text).split()
        
        aggregated_surprisal = math.nan
        
        reconstructed_words = [str(d['word']) for d in sentence_word_surprisals]
        cr_word_start_idx = -1
        # Find the start of the CR word sequence in our reconstructed words
        for i in range(len(reconstructed_words) - len(target_cr_words) + 1):
            window = reconstructed_words[i:i+len(target_cr_words)]
            if window == target_cr_words:
                cr_word_start_idx = i
                break
        
        if cr_word_start_idx != -1:
            cr_word_data = sentence_word_surprisals[cr_word_start_idx : cr_word_start_idx + len(target_cr_words)]
            surprisals_for_cr = [d['surprisal'] for d in cr_word_data]
            if any(math.isnan(s) for s in surprisals_for_cr):
                aggregated_surprisal = math.nan
            else:
                aggregated_surprisal = sum(surprisals_for_cr)
        else:
            print(f"    Warning: Failed to find CR '{target_cr_text}' in reconstructed words for Item='{item_id}', Cond='{condition_str}'.")

        aggregated_results.append({
            "sentence_type": NOVEL_SENTENCE_TYPE_IDENTIFIER,
            "item_id": item_id,
            "condition": condition_str,
            "critical_region_text": target_cr_text,
            "aggregated_surprisal_bits": aggregated_surprisal,
            "original_full_sentence": current_sentence_bpes_df['full_sentence_text'].iloc[0]
        })

    if aggregated_results:
        output_df = pd.DataFrame(aggregated_results)
        output_df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.8f')
        print(f"\nSuccessfully aggregated surprisals for {len(output_df)} sentences to {os.path.abspath(OUTPUT_CSV_FILE)}")
        print(f"First few results:\n{output_df.head()}")
    else:
        print("\nNo results were aggregated.")

# --- THIS IS THE FIX ---
# This block ensures that the main() function is called when you run the script.
if __name__ == "__main__":
    main()