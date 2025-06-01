# llm-poverty-of-stimulus/llm_pos/wilcox/aggregate_surprisals.py
import pandas as pd
import os
import math
from transformers import AutoTokenizer # Ensure 'transformers' is installed

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input Paths (relative to CURRENT_SCRIPT_DIR)
# This is the output from your modified tim_extract_surprisals.py
BPE_SURPRISALS_FILE = os.path.join("tims_results", "master_stimuli_list_surprisals_gpt2.csv") 
# This is your token-per-row file with region labels (previously word_region_mapping.csv)
TEST_ITEMS_FILE = os.path.join("data", "processed", "test_items.csv") 

# Output Path
AGGREGATED_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "aggregated")
MODEL_NAME = "gpt2" # Used for surprisal column name and output file naming

# --- Critical Region Definitions ---
# Defines how to find critical regions based on sentence_type and parsed condition features.
# 'target_region_labels' are the 'region' values from test_items.csv.
CRITICAL_REGION_DEFINITIONS = {
    "basic_object": [
        {
            "cr_name": "basic_object_post_gap_prepositional_phrase",
            "conditions_to_match": {"gap_status": "gap"},
            "target_region_labels": ["prep"]
        },
        {
            "cr_name": "basic_object_filled_object_NP",
            "conditions_to_match": {"gap_status": "nogap"},
            "target_region_labels": ["np2"]
        }
    ],
    "basic_subject": [
        {
            "cr_name": "basic_subject_post_gap_verb",
            "conditions_to_match": {"gap_status": "gap"},
            "target_region_labels": ["verb"]
        },
        {
            "cr_name": "basic_subject_filled_subject_NP1",
            "conditions_to_match": {"gap_status": "nogap"},
            "target_region_labels": ["np1"]
        }
    ],
    "basic_pp": [
        {
            "cr_name": "basic_pp_post_gap_final_material",
            "conditions_to_match": {"gap_status": "gap"},
            "target_region_labels": ["end"]
        },
        {
            "cr_name": "basic_pp_filled_object_of_prep_NP3",
            "conditions_to_match": {"gap_status": "nogap"},
            "target_region_labels": ["np3"]
        }
    ]
    # Add other sentence type definitions here later (e.g., for islands)
}

# --- Initialize Tokenizer ---
# This tokenizer should match the one used for generating BPE surprisals.
HF_MODEL_NAME_TOKENIZER = "gpt2" 
tokenizer = None
try:
    print(f"Loading tokenizer: {HF_MODEL_NAME_TOKENIZER}")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME_TOKENIZER)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer: Setting pad_token to eos_token ('{tokenizer.eos_token}')")
except Exception as e:
    print(f"Critical Error: Could not load tokenizer '{HF_MODEL_NAME_TOKENIZER}'. Exception: {e}")
    # Consider exiting if tokenizer is essential: exit()

# --- Helper Functions ---
def parse_condition_string(condition_str, sentence_type):
    """
    Parses a condition string (e.g., 'what_gap', 'that_nogap_obj_island')
    into a dictionary of features.
    This will need to be expanded for more complex sentence types.
    """
    parts = condition_str.split('_')
    parsed = {}
    if len(parts) >= 1:
        parsed['filler_status'] = parts[0] 
    if len(parts) >= 2:
        parsed['gap_status'] = parts[1].replace('-', '') # Normalize 'no-gap' to 'nogap'
    
    # Add more sophisticated parsing based on sentence_type for island features etc. later
    # For basic types, filler_status and gap_status are often enough.
    # Example for hierarchy (if we were doing it now):
    # if sentence_type == "hierarchy":
    #     if len(parts) >= 3: parsed['hierarchy_subtype'] = parts[2] 
    
    return parsed

def get_word_level_surprisals_and_regions(
    current_bpe_df_group: pd.DataFrame, 
    current_word_map_df_group: pd.DataFrame, 
    tokenizer_instance, 
    bpe_surprisal_col: str
) -> list:
    """
    Aligns BPE tokens to words, sums BPE surprisals to get word surprisals,
    and associates words with their region labels.
    Adapted from your original aggregate_surprisals.py.
    """
    if tokenizer_instance is None:
        print("      ERROR: Tokenizer not available for BPE alignment.")
        return []

    sentence_word_data_list = []
    
    bpes_for_sentence = current_bpe_df_group.to_dict('records') # Assumes current_bpe_df_group is already sorted by bpe_token_index
    target_words_with_regions = current_word_map_df_group[['token', 'region']].to_dict('records') # Assumes current_word_map_df_group is sorted

    bpe_cursor = 0 

    for word_info in target_words_with_regions:
        target_word_text = str(word_info['token']).strip()
        target_word_region_label = word_info['region']
        
        current_word_bpe_strings = []
        current_word_total_surprisal = 0.0 # Initialize as float
        reconstructed_word = ""
        
        # Store the BPE cursor state before trying to match the current word
        start_bpe_cursor_for_this_word_attempt = bpe_cursor
        
        bpes_consumed_for_this_word_in_attempt = 0

        while bpe_cursor < len(bpes_for_sentence):
            bpe_data = bpes_for_sentence[bpe_cursor]
            bpe_text_token = bpe_data['bpe_token_str']
            bpe_surprisal_value = bpe_data[bpe_surprisal_col]

            current_word_bpe_strings.append(bpe_text_token)
            
            if pd.isna(bpe_surprisal_value):
                current_word_total_surprisal = math.nan # If any BPE is NaN, word surprisal is NaN
                # We'll break from inner while to mark word as NaN, but BPE cursor has advanced
                break 
            elif not math.isnan(current_word_total_surprisal): # Only add if sum isn't already NaN
                current_word_total_surprisal += bpe_surprisal_value

            try:
                reconstructed_word = tokenizer_instance.convert_tokens_to_string(current_word_bpe_strings).strip()
            except Exception as e:
                print(f"      Warning: Tokenizer error for BPEs {current_word_bpe_strings}: {e}")
                reconstructed_word = "TOKENIZER_ERROR" # Avoid crash
            
            bpe_cursor += 1 # Advance global BPE cursor
            bpes_consumed_for_this_word_in_attempt += 1
            
            if reconstructed_word == target_word_text:
                break # Word matched
            
            # Backtracking heuristic (from your original script)
            if len(reconstructed_word) > len(target_word_text) + 5 and \
               not target_word_text.startswith(reconstructed_word) and \
               not reconstructed_word.startswith(target_word_text) and \
               bpes_consumed_for_this_word_in_attempt > 0: # Check if any BPEs were actually consumed in this attempt
                
                bpe_cursor -= 1 # Backtrack the global BPE cursor
                # Remove the last BPE from current attempt
                popped_bpe_text = current_word_bpe_strings.pop() 
                
                # Subtract its surprisal if current sum is not NaN and popped BPE surprisal was not NaN
                last_bpe_surprisal = bpes_for_sentence[bpe_cursor][bpe_surprisal_col] # Get the surprisal of the popped BPE
                if not math.isnan(current_word_total_surprisal) and not pd.isna(last_bpe_surprisal):
                    current_word_total_surprisal -= last_bpe_surprisal
                
                reconstructed_word = tokenizer_instance.convert_tokens_to_string(current_word_bpe_strings).strip() if current_word_bpe_strings else ""
                break # Exit inner BPE loop, this attempt to match target_word_text failed with current BPEs

        if reconstructed_word == target_word_text:
            sentence_word_data_list.append({
                'word': target_word_text,
                'surprisal': current_word_total_surprisal,
                'region_label': target_word_region_label
            })
        else:
            # Word alignment failed for this target_word_text
            # print(f"      Warning: Word alignment failed for target '{target_word_text}' (reconstructed: '{reconstructed_word}'). Item: {current_bpe_group['item'].iloc[0]}, Cond: {current_bpe_group['condition'].iloc[0]}. Surprisal set to NaN.")
            sentence_word_data_list.append({
                'word': target_word_text,
                'surprisal': math.nan,
                'region_label': target_word_region_label
            })
            # Reset BPE cursor to where it was before attempting this failed word,
            # so the *next target word* starts consuming from that point.
            # This was the logic in your original script.
            bpe_cursor = start_bpe_cursor_for_this_word_attempt
            
    if len(sentence_word_data_list) != len(target_words_with_regions):
        print(f"      Warning: Alignment resulted in {len(sentence_word_data_list)} words, but expected {len(target_words_with_regions)} for this sentence.")
        # This might indicate a more serious issue with alignment or data.
    return sentence_word_data_list


def main():
    if tokenizer is None:
        print("Tokenizer not loaded. Exiting.")
        return

    if not os.path.exists(AGGREGATED_OUTPUT_DIR):
        os.makedirs(AGGREGATED_OUTPUT_DIR)
        print(f"Created output directory: {AGGREGATED_OUTPUT_DIR}")

    output_csv_file = os.path.join(AGGREGATED_OUTPUT_DIR, f"{MODEL_NAME}_critical_regions_aggregated.csv")

    print(f"Loading BPE surprisals from: {BPE_SURPRISALS_FILE}")
    try:
        bpe_df = pd.read_csv(BPE_SURPRISALS_FILE)
    except FileNotFoundError:
        print(f"Error: BPE surprisal file not found at '{os.path.abspath(BPE_SURPRISALS_FILE)}'")
        return
    except Exception as e:
        print(f"Error loading BPE surprisals: {e}")
        return
    
    print(f"Loading word-region mappings (test_items.csv) from: {TEST_ITEMS_FILE}")
    try:
        word_map_df = pd.read_csv(TEST_ITEMS_FILE)
        if 'test' in word_map_df.columns: # Ensure consistent column name for sentence type
            word_map_df.rename(columns={'test': 'sentence_type'}, inplace=True)
        if 'sentence_type' not in word_map_df.columns:
             print(f"Error: 'sentence_type' (or 'test') column not found in {TEST_ITEMS_FILE}")
             return
    except FileNotFoundError:
        print(f"Error: Word-region mapping file (test_items.csv) not found at '{os.path.abspath(TEST_ITEMS_FILE)}'")
        return
    except Exception as e:
        print(f"Error loading word-region mappings: {e}")
        return

    bpe_surprisal_col_name = f"surprisal_bits_{MODEL_NAME}"
    if bpe_surprisal_col_name not in bpe_df.columns:
        print(f"Critical Error: Surprisal column '{bpe_surprisal_col_name}' not found in {BPE_SURPRISALS_FILE}.")
        return
    if 'bpe_token_str' not in bpe_df.columns:
        print(f"Critical Error: Column 'bpe_token_str' not found in {BPE_SURPRISALS_FILE}.")
        return

    aggregated_results = []
    
    # Ensure grouping keys are of string type for robustness
    bpe_df['source_doc_name'] = bpe_df['source_doc_name'].astype(str)
    bpe_df['item'] = bpe_df['item'].astype(str)
    bpe_df['condition'] = bpe_df['condition'].astype(str)
    
    word_map_df['sentence_type'] = word_map_df['sentence_type'].astype(str)
    word_map_df['item'] = word_map_df['item'].astype(str)
    word_map_df['condition'] = word_map_df['condition'].astype(str)

    # Group BPE data by sentence identifiers
    # 'source_doc_name' from BPE file is our 'sentence_type'
    grouped_bpe_data = bpe_df.groupby(['source_doc_name', 'item', 'condition'])
    
    print(f"Processing {len(grouped_bpe_data)} unique sentences from BPE file...")

    for sentence_key_tuple, current_bpe_sentence_group in grouped_bpe_data:
        sentence_type, item_id, condition_str = sentence_key_tuple
        
        # print(f"  Processing: Type='{sentence_type}', Item='{item_id}', Condition='{condition_str}'")

        if sentence_type not in CRITICAL_REGION_DEFINITIONS:
            # print(f"    Info: No critical region definition rule found for sentence_type '{sentence_type}'. Skipping CR aggregation for this type.")
            continue

        parsed_condition = parse_condition_string(condition_str, sentence_type)
        
        matched_rule = None
        for rule in CRITICAL_REGION_DEFINITIONS[sentence_type]:
            is_match = True
            for feature, value in rule["conditions_to_match"].items():
                if parsed_condition.get(feature) != value:
                    is_match = False
                    break
            if is_match:
                matched_rule = rule
                break
        
        if not matched_rule:
            # print(f"    Info: No specific CR rule matched for Type='{sentence_type}', Cond='{condition_str}' with parsed features {parsed_condition}. Skipping CR aggregation.")
            continue

        target_region_labels = matched_rule["target_region_labels"]
        cr_name_output = matched_rule["cr_name"]

        # Get corresponding words and their region labels from test_items.csv for this specific sentence
        current_word_map_sentence_group = word_map_df[
            (word_map_df['sentence_type'] == sentence_type) &
            (word_map_df['item'] == item_id) &
            (word_map_df['condition'] == condition_str)
        ]

        if current_word_map_sentence_group.empty:
            print(f"    Warning: No word/region mapping found in test_items.csv for Type='{sentence_type}', Item='{item_id}', Cond='{condition_str}'. Skipping CR aggregation.")
            continue
        
        # Ensure data is sorted for alignment (BPEs by index, words by their order in test_items.csv)
        # It's assumed test_items.csv is already ordered by token sequence for each sentence.
        sorted_bpe_group = current_bpe_sentence_group.sort_values(by="bpe_token_index")
        
        sentence_word_level_data = get_word_level_surprisals_and_regions(
            sorted_bpe_group,
            current_word_map_sentence_group, # Assumed to be correctly ordered
            tokenizer,
            bpe_surprisal_col_name
        )

        if not sentence_word_level_data:
            print(f"    Warning: Word-level surprisal processing failed for Type='{sentence_type}', Item='{item_id}', Cond='{condition_str}'. Skipping CR.")
            continue

        cr_surprisal_sum = 0.0
        cr_text_parts = []
        found_any_cr_word = False

        for word_data in sentence_word_level_data:
            if word_data['region_label'] in target_region_labels:
                found_any_cr_word = True
                if math.isnan(word_data['surprisal']):
                    cr_surprisal_sum = math.nan # If any word in CR is NaN, sum is NaN
                    break 
                cr_surprisal_sum += word_data['surprisal']
                cr_text_parts.append(word_data['word'])
        
        if not found_any_cr_word and target_region_labels:
             # print(f"    Warning: No words found for target critical regions {target_region_labels} for Type='{sentence_type}', Item='{item_id}', Cond='{condition_str}'. CR surprisal is NaN.")
             cr_surprisal_sum = math.nan # If no words even matched the CR labels

        original_full_sentence = sorted_bpe_group['full_sentence_text'].iloc[0] if not sorted_bpe_group.empty else "N/A"

        aggregated_results.append({
            "sentence_type": sentence_type,
            "item_id": item_id,
            "condition": condition_str,
            "critical_region_name": cr_name_output,
            "critical_region_text": " ".join(cr_text_parts) if cr_text_parts else "N/A",
            "aggregated_surprisal_bits": cr_surprisal_sum,
            "original_full_sentence": original_full_sentence
        })

    if aggregated_results:
        output_df = pd.DataFrame(aggregated_results)
        output_df.to_csv(output_csv_file, index=False, float_format='%.8f')
        print(f"\nSuccessfully aggregated surprisals for {len(output_df)} sentences to {os.path.abspath(output_csv_file)}")
    else:
        print("\nNo results were successfully aggregated. Output file might be empty or only contain headers if created.")

if __name__ == "__main__":
    main()