# llm-poverty-of-stimulus/llm_pos/wilcox/aggregate_surprisals.py
import pandas as pd
import os
import math
from transformers import AutoTokenizer

# --- Configuration ---
# Assume this script is in llm_pos/wilcox/
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define base paths relative to this script's location
DATA_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_data") # For word_region_mapping & sentence_components
BPE_RESULTS_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results") # BPE surprisals from your other script
AGGREGATED_RESULTS_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results") # Where this script saves output

MODEL_NAME_FOR_FILES = "gpt2"  # Matches your tim_extract_surprisals.py
HF_MODEL_NAME_TOKENIZER = "gpt2" # Tokenizer should match the model used for BPEs

# INPUT FILES (ensure these exist in DATA_INPUT_DIR or BPE_RESULTS_INPUT_DIR)

# This is the CSV file that defines the structure of your sentences (e.g., columns for prep, np2)
# It's used to determine which parts form the critical region.
SENTENCE_COMPONENTS_INPUT_CSV_BASENAME = "test_set.csv" # e.g., your test_set.csv
SENTENCE_COMPONENTS_INPUT_CSV = os.path.join(DATA_INPUT_DIR, SENTENCE_COMPONENTS_INPUT_CSV_BASENAME)

# This is the CSV file that was *actually fed into* tim_extract_surprisals.py
# Its basename is used to construct the name of the BPE surprisal file.
# The *content* of its first column becomes the 'source_doc_name' *within* the BPE surprisal CSV.
PROCESSED_STIMULI_FILE_BASENAME_FOR_BPE = "extract_test.csv"

# This is the BPE-level surprisal CSV generated by tim_extract_surprisals.py
BPE_SURPRISALS_CSV_BASENAME = f"{PROCESSED_STIMULI_FILE_BASENAME_FOR_BPE.split('.')[0]}_surprisals_{MODEL_NAME_FOR_FILES}.csv"
BPE_SURPRISALS_CSV = os.path.join(BPE_RESULTS_INPUT_DIR, BPE_SURPRISALS_CSV_BASENAME)

WORD_REGION_MAPPING_CSV_BASENAME = "word_region_mapping.csv" # You'll need this file
WORD_REGION_MAPPING_CSV = os.path.join(DATA_INPUT_DIR, WORD_REGION_MAPPING_CSV_BASENAME)

# OUTPUT FILE for this script - now includes components file base name
AGGREGATED_OUTPUT_CSV = os.path.join(
    AGGREGATED_RESULTS_OUTPUT_DIR,
    f"{SENTENCE_COMPONENTS_INPUT_CSV_BASENAME.split('.')[0]}_{MODEL_NAME_FOR_FILES}_critical_region_surprisals_aggregated.csv"
)

# --- Initialize Tokenizer ---
try:
    print(f"Loading tokenizer: {HF_MODEL_NAME_TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME_TOKENIZER)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            print(f"Tokenizer: Setting pad_token to eos_token ('{tokenizer.eos_token}')")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Tokenizer: Adding new pad_token '<|pad|>'")
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
except Exception as e:
    print(f"Critical Error: Could not load tokenizer '{HF_MODEL_NAME_TOKENIZER}'. Exception: {e}")
    exit()

def main():
    if not os.path.exists(AGGREGATED_RESULTS_OUTPUT_DIR):
        os.makedirs(AGGREGATED_RESULTS_OUTPUT_DIR)
        print(f"Created output directory: {AGGREGATED_RESULTS_OUTPUT_DIR}")

    input_files_to_check = {
        "Sentence Components CSV": SENTENCE_COMPONENTS_INPUT_CSV,
        "BPE Surprisals CSV": BPE_SURPRISALS_CSV,
        "Word Region Mapping CSV": WORD_REGION_MAPPING_CSV
    }
    missing_files = False
    for name, fpath in input_files_to_check.items():
        if not os.path.exists(fpath):
            print(f"Critical Error: Input file '{name}' not found at '{fpath}'")
            missing_files = True
    if missing_files:
        print("Please ensure all required input files are present and paths are correct in the script.")
        return

    print("Loading data...")
    try:
        bpe_df = pd.read_csv(BPE_SURPRISALS_CSV)
        word_map_df = pd.read_csv(WORD_REGION_MAPPING_CSV)
        components_df = pd.read_csv(SENTENCE_COMPONENTS_INPUT_CSV)
    except FileNotFoundError as e:
        print(f"Critical Error: Could not load one of the input CSV files. {e}")
        return
    except Exception as e:
        print(f"Critical Error: Error loading input CSV files. {e}")
        return
    print("Data loaded.")

    print(f"Columns in BPE Surprisals file ('{os.path.basename(BPE_SURPRISALS_CSV)}'): {bpe_df.columns.tolist()}")
    print(f"Columns in Word Region Mapping file ('{os.path.basename(WORD_REGION_MAPPING_CSV)}'): {word_map_df.columns.tolist()}")
    print(f"Columns in Sentence Components file ('{os.path.basename(SENTENCE_COMPONENTS_INPUT_CSV)}'): {components_df.columns.tolist()}")

    if 'source_doc_name' not in bpe_df.columns:
        print(f"Critical Error: 'source_doc_name' column missing from BPE surprisals file: {BPE_SURPRISALS_CSV}")
        return

    # Standardize 'source_doc_name' in word_map_df if 'test' column exists
    if 'test' in word_map_df.columns and 'source_doc_name' not in word_map_df.columns:
        word_map_df.rename(columns={'test': 'source_doc_name'}, inplace=True)
        print(f"Renamed 'test' to 'source_doc_name' in {WORD_REGION_MAPPING_CSV_BASENAME}")
    elif 'test' not in word_map_df.columns and 'source_doc_name' not in word_map_df.columns:
        print(f"Critical Error: Missing 'source_doc_name' or 'test' column in {WORD_REGION_MAPPING_CSV_BASENAME}. Cannot proceed with merging.")
        return

    # Ensure join keys are strings and handle potential missing columns
    for df, df_name in [(bpe_df, "BPE_DF"), (word_map_df, "WordMap_DF"), (components_df, "Components_DF")]:
        for col in ['item', 'condition', 'source_doc_name']:
            if col in df.columns:
                df[col] = df[col].astype(str)
            elif col == 'source_doc_name' and df_name == "Components_DF":
                pass # Components_DF does not require source_doc_name for its main lookup
            else:
                print(f"Warning: Column '{col}' not found in {df_name}. This might affect grouping/merging.")


    print("\n--- Source Document Name Diagnostics ---")
    bpe_unique_sources = bpe_df['source_doc_name'].unique()
    word_map_unique_sources = word_map_df['source_doc_name'].unique() if 'source_doc_name' in word_map_df.columns else ["Column Missing"]
    print(f"Unique source_doc_name in BPE surprisals file (content from 1st col of '{PROCESSED_STIMULI_FILE_BASENAME_FOR_BPE}'): {bpe_unique_sources}")
    print(f"Unique source_doc_name in word mapping file: {word_map_unique_sources}")
    print("Ensure these align for successful merging (e.g., word mapping should contain entries for sources like 'test_set.csv' or 'test_set').")
    print("-------------------------------------\n")

    aggregated_results = []
    surprisal_col_name = f"surprisal_bits_{MODEL_NAME_FOR_FILES}"
    
    if surprisal_col_name not in bpe_df.columns:
        print(f"Critical Error: Surprisal column '{surprisal_col_name}' not found in {BPE_SURPRISALS_CSV}.")
        print(f"Available columns: {bpe_df.columns.tolist()}")
        return
    
    if 'bpe_token_str' not in bpe_df.columns:
        print(f"Critical Error: Column 'bpe_token_str' not found in {BPE_SURPRISALS_CSV}. Needed for word reconstruction.")
        return

    grouped_bpe = bpe_df.groupby(['source_doc_name', 'item', 'condition'])
    grouped_word_map = word_map_df.groupby(['source_doc_name', 'item', 'condition'])

    print(f"Processing {len(grouped_bpe)} unique sentence groups from BPE file...")

    for name_bpe, bpe_group in grouped_bpe:
        source_doc_bpe, item_id, condition_val = name_bpe # These are now strings
        
        word_map_group = None
        # name_bpe[0] is the source_doc_name string from the BPE file (e.g., "test_set.csv")
        keys_to_try = [name_bpe] # Try ('test_set.csv', '1', 'what_nogap')
        if source_doc_bpe.endswith('.csv'):
            # Try ('test_set', '1', 'what_nogap')
            keys_to_try.append((source_doc_bpe[:-4], item_id, condition_val))
        
        for key_try in keys_to_try:
            try:
                word_map_group = grouped_word_map.get_group(key_try)
                break 
            except KeyError:
                continue
            
        if word_map_group is None:
            print(f"  Warning: No word mapping found for BPE group {name_bpe} (tried keys: {keys_to_try}). Skipping.")
            continue

        current_bpes = bpe_group.sort_values(by="bpe_token_index").to_dict('records')
        target_words_info = word_map_group.to_dict('records') # Contains 'token' and 'region'

        sentence_word_surprisals = []
        bpe_cursor = 0
        
        for word_info in target_words_info:
            target_word_text = str(word_info['token']).strip()
            current_word_bpe_strings = []
            current_word_total_surprisal = 0
            reconstructed_word = ""
            
            start_bpe_cursor = bpe_cursor

            bpes_for_this_word_count = 0
            while bpe_cursor < len(current_bpes):
                bpe_data = current_bpes[bpe_cursor]
                current_word_bpe_strings.append(bpe_data['bpe_token_str'])
                
                # Ensure surprisal is numeric before adding
                try:
                    surprisal_value = float(bpe_data[surprisal_col_name])
                    if not math.isnan(surprisal_value):
                         current_word_total_surprisal += surprisal_value
                    else: # if a BPE has NaN, the word surprisal is NaN
                        current_word_total_surprisal = math.nan
                        break 
                except (ValueError, TypeError):
                    print(f"  Warning: Non-numeric surprisal '{bpe_data[surprisal_col_name]}' for BPE '{bpe_data['bpe_token_str']}' in group {name_bpe}. Word surprisal will be NaN.")
                    current_word_total_surprisal = math.nan
                    break # Mark word surprisal as NaN

                try:
                    reconstructed_word = tokenizer.convert_tokens_to_string(current_word_bpe_strings).strip()
                except Exception as e:
                    print(f"    Error during tokenizer.convert_tokens_to_string for BPEs {current_word_bpe_strings}: {e}")
                    reconstructed_word = "TOKENIZER_ERROR"
                
                bpe_cursor += 1
                bpes_for_this_word_count +=1

                if reconstructed_word == target_word_text:
                    break
                if len(reconstructed_word) > len(target_word_text) + 5 and \
                   not target_word_text.startswith(reconstructed_word) and \
                   not reconstructed_word.startswith(target_word_text) and \
                   bpes_for_this_word_count > 0 : # Ensure we don't pop if only one BPE was tried
                    bpe_cursor -= 1 # Backtrack
                    bpes_for_this_word_count -=1
                    popped_bpe_str = current_word_bpe_strings.pop()
                    # Subtract the last BPE's surprisal if it was valid
                    if not math.isnan(current_word_total_surprisal) and not math.isnan(float(bpe_data[surprisal_col_name])): # Check bpe_data again
                        current_word_total_surprisal -= float(bpe_data[surprisal_col_name])
                    reconstructed_word = tokenizer.convert_tokens_to_string(current_word_bpe_strings).strip() if current_word_bpe_strings else ""
                    break
            
            if reconstructed_word == target_word_text:
                sentence_word_surprisals.append({
                    'word': target_word_text,
                    'region_name': word_info['region'],
                    'surprisal': current_word_total_surprisal
                })
            else:
                print(f"  Warning: Word alignment failed for target '{target_word_text}' (reconstructed: '{reconstructed_word}') in group {name_bpe}. BPEs tried: {current_word_bpe_strings}. Surprisal set to NaN.")
                bpe_cursor = start_bpe_cursor 
                sentence_word_surprisals.append({
                    'word': target_word_text,
                    'region_name': word_info['region'],
                    'surprisal': math.nan 
                })
        
        # --- Component Lookup (uses item_id and condition_val which are strings) ---
        component_row_series = components_df[
             (components_df['item'] == item_id) & 
             (components_df['condition'] == condition_val)
        ]
        
        aggregated_critical_surprisal = math.nan
        final_critical_region_text = "N/A"

        if component_row_series.empty:
             print(f"  Warning: No component data found for item '{item_id}', condition '{condition_val}' in {os.path.basename(SENTENCE_COMPONENTS_INPUT_CSV)}. Skipping critical region aggregation for this group.")
        else:
            component_row = component_row_series.iloc[0]
            critical_region_text_parts = []
            target_region_names = []
            
            if condition_val.endswith("_gap"):
                target_region_names = ["prep"] 
                prep_text = str(component_row.get('prep', "") or "").strip()
                if prep_text: critical_region_text_parts.append(prep_text)
            elif condition_val.endswith("_nogap"):
                target_region_names = ["np2"] 
                np2_text = str(component_row.get('np2', "") or "").strip()
                if np2_text: critical_region_text_parts.append(np2_text)
            else:
                print(f"  Warning: Unknown condition format '{condition_val}' for {name_bpe}. Cannot determine critical region names.")
            
            final_critical_region_text = " ".join(critical_region_text_parts).strip()
            if not final_critical_region_text and target_region_names:
                 final_critical_region_text = "N/A (empty component text for region)"
            elif not target_region_names:
                 final_critical_region_text = "N/A (no target region names defined)"

            if target_region_names:
                current_sum_critical_surprisal = 0.0 
                found_critical_words_count = 0
                expected_critical_words_count = sum(1 for w_info in target_words_info if w_info['region'] in target_region_names)
                
                for word_sur_info in sentence_word_surprisals:
                    if word_sur_info['region_name'] in target_region_names:
                        if not math.isnan(word_sur_info['surprisal']):
                            current_sum_critical_surprisal += word_sur_info['surprisal']
                            found_critical_words_count += 1
                        else: 
                            current_sum_critical_surprisal = math.nan 
                            break 
                
                if not math.isnan(current_sum_critical_surprisal):
                    if found_critical_words_count == 0 and expected_critical_words_count > 0:
                        print(f"  Warning: Critical region '{final_critical_region_text}' (regions: {target_region_names}) for {name_bpe} had {expected_critical_words_count} expected words but none were found/aggregated. Surprisal set to NaN.")
                        aggregated_critical_surprisal = math.nan
                    elif found_critical_words_count < expected_critical_words_count and expected_critical_words_count > 0:
                        print(f"  Warning: Not all expected words found for critical region '{final_critical_region_text}' (regions: {target_region_names}) for {name_bpe}. Found {found_critical_words_count}/{expected_critical_words_count}. Aggregated from found words.")
                        aggregated_critical_surprisal = current_sum_critical_surprisal
                    elif found_critical_words_count == expected_critical_words_count and expected_critical_words_count > 0:
                         aggregated_critical_surprisal = current_sum_critical_surprisal
                    elif expected_critical_words_count == 0: # No words defined for this region in map
                        print(f"  Info: No words defined for critical region '{final_critical_region_text}' (regions: {target_region_names}) in word mapping for {name_bpe}. Surprisal is 0 or NaN if prior issues.")
                        aggregated_critical_surprisal = 0.0 if not math.isnan(current_sum_critical_surprisal) else math.nan # Or NaN
                else: # if current_sum_critical_surprisal is NaN
                    aggregated_critical_surprisal = math.nan

        aggregated_results.append({
            "source_doc_name": source_doc_bpe,
            "item": item_id,
            "condition": condition_val,
            "critical_region_text": final_critical_region_text,
            "aggregated_surprisal_bits": aggregated_critical_surprisal
        })

    if aggregated_results:
        output_df = pd.DataFrame(aggregated_results)
        output_df.to_csv(AGGREGATED_OUTPUT_CSV, index=False, float_format='%.8f')
        print(f"\nSuccessfully aggregated surprisals to {AGGREGATED_OUTPUT_CSV}")
    else:
        print("\nNo results were successfully aggregated.")

if __name__ == "__main__":
    main()