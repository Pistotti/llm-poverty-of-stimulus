# llm_pos/extract_surprisals.py
import lib
import csv
import os
import re
import time # Optional: for timing

# --- Configuration ---
DATASET_FILE = "tims_data/extract_test.txt"  # Path to your text file with sentences (one per line)
                                  

MODEL_TO_USE = "gpt2"             # Specify the model you want to use (e.g., "gpt2", "grnn")

OUTPUT_DIRECTORY = "tims_results" # Subdirectory for results
OUTPUT_CSV_FILENAME = f"{MODEL_TO_USE}_surprisals_for_{os.path.basename(DATASET_FILE).replace('.', '_')}.csv"
# --- End Configuration ---

def load_sentences_from_file(filepath):
    """Loads sentences from a text file, one sentence per line."""
    sentences = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Clean up sentence (remove leading/trailing whitespace, multiple spaces)
                cleaned_line = re.sub(r'\s+', ' ', line).strip()
                if cleaned_line: # Add if not empty
                    sentences.append(cleaned_line)
        if not sentences:
            print(f"Warning: No sentences found or loaded from {filepath}")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        print("Please ensure the DATASET_FILE path is correct.")
        return None
    return tuple(sentences)

def main():
    print(f"Starting surprisal extraction for dataset: {DATASET_FILE}")
    print(f"Using model: {MODEL_TO_USE}")

    # Load sentences from the dataset file
    my_sentences = load_sentences_from_file(DATASET_FILE)
    if not my_sentences:
        return # Exit if no sentences loaded

    print(f"Loaded {len(my_sentences)} sentences.")
    # For brevity, optionally print only a few:
    # for i, s in enumerate(my_sentences[:3]):
    #     print(f"  S{i+1}: {s}")
    # if len(my_sentences) > 3:
    #     print("  ...")

    script_start_time = time.time()

    # --- Get Surprisal Data ---
    print("Fetching surprisal data...")
    # lib.get_surprisals_per_model expects a tuple of model names
    surprisal_data_per_model = lib.get_surprisals_per_model(
        sentences=my_sentences,
        models=(MODEL_TO_USE,) # Pass the single model in a tuple
    )
    print("Surprisal data fetched.")

    # --- Save Surprisal Data to CSV ---
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {os.path.abspath(OUTPUT_DIRECTORY)}")
    
    csv_filepath = os.path.join(OUTPUT_DIRECTORY, OUTPUT_CSV_FILENAME)
    print(f"Attempting to save surprisal data to CSV: {os.path.abspath(csv_filepath)}")

    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([
                "model_name",
                "sentence_idx_in_dataset",
                "original_sentence_text",
                "token_sequence_idx",
                "token_text",
                "token_original_word_idx",
                "surprisal_value"
            ])

            # surprisal_data_per_model will have one entry for MODEL_TO_USE
            if MODEL_TO_USE in surprisal_data_per_model:
                sentence_surprisals_tuple = surprisal_data_per_model[MODEL_TO_USE]
                for sentence_idx, sentence_surprisal_obj in enumerate(sentence_surprisals_tuple):
                    original_sentence_str = " ".join(sentence_surprisal_obj.original_sentence.original_token_strings)
                    for token_seq_idx, token_obj in enumerate(sentence_surprisal_obj.tokens):
                        csv_writer.writerow([
                            MODEL_TO_USE,
                            sentence_idx,
                            original_sentence_str,
                            token_seq_idx,
                            token_obj.text,
                            token_obj.idx,
                            f"{token_obj.surprisal:.4f}" if token_obj.surprisal is not None else "N/A"
                        ])
            else:
                print(f"Warning: No surprisal data found for model {MODEL_TO_USE} in the results.")
        
        print(f"Surprisal data saved to CSV: {os.path.abspath(csv_filepath)}")
    except IOError as e:
        print(f"Error writing CSV file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")


    script_end_time = time.time()
    print(f"Script finished in {script_end_time - script_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()