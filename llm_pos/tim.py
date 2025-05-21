# tim.py
import matplotlib
matplotlib.use('Agg')

import viz
import lib
import csv
import time
import os
import re # For cleaning up double spaces

# --- Configuration ---
output_directory = "tims_results"
my_models = ("gpt2", "grnn") # Or any other models you have set up

# Define your sentence quads
# The order of sentences within each "sentences_raw" tuple is important for pairing later:
# 1. +what, +obj
# 2. +that, +obj
# 3. +what, -obj
# 4. +that, -obj
sentence_quads_definitions = [
    {
        "name": "grabbed_construction",
        "sentences_raw": (
            "I know what with gusto our uncle grabbed the food in front of the guests at the holiday party",
            "I know that with gusto our uncle grabbed the food in front of the guests at the holiday party",
            "I know what with gusto our uncle grabbed in front of the guests at the holiday party",
            "I know that with gusto our uncle grabbed in front of the guests at the holiday party"
        ),
        "condition_labels": ("+what,+obj", "+that,+obj", "+what,-obj", "+that,-obj") # For CSV and plot titles/legends
    },
    {
        "name": "caught_construction",
        "sentences_raw": (
            "My neighbor told me what with skill the dog caught the mouse for his neighbors yesterday",
            "My neighbor told me that with skill the dog caught the mouse for his neighbors yesterday",
            "My neighbor told me what with skill the dog caught for his neighbors yesterday",
            "My neighbor told me that with skill the dog caught for his neighbors yesterday"
        ),
        "condition_labels": ("+what,+obj", "+that,+obj", "+what,-obj", "+that,-obj")
    }
]

# --- Main Processing Loop ---
print(f"Processing with models: {my_models}")
print("Note: First-time model/container startup can take a moment.")

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created base directory: {os.path.abspath(output_directory)}")

overall_start_time = time.time()

for quad_def in sentence_quads_definitions:
    quad_name = quad_def["name"]
    
    # Clean sentences
    current_quad_sentences_cleaned = tuple(re.sub(r'\s+', ' ', s).strip() for s in quad_def["sentences_raw"])
    current_quad_condition_labels = quad_def["condition_labels"]

    # Assign sentences to variables based on their assumed condition for clarity in pairing
    s_what_obj = current_quad_sentences_cleaned[0]
    s_that_obj = current_quad_sentences_cleaned[1]
    s_what_no_obj = current_quad_sentences_cleaned[2]
    s_that_no_obj = current_quad_sentences_cleaned[3]

    l_what_obj = current_quad_condition_labels[0]
    l_that_obj = current_quad_condition_labels[1]
    l_what_no_obj = current_quad_condition_labels[2]
    l_that_no_obj = current_quad_condition_labels[3]

    print(f"\n--- Processing Quad: {quad_name} ---")
    quad_start_time = time.time()

    # --- Get Surprisal Data for the entire current quad (all 4 sentences) ---
    print("Fetching surprisal data for all 4 sentences in the quad...")
    # We fetch for all 4 at once for the CSV, viz.plot_surprisals will re-fetch for pairs
    surprisal_data_per_model_for_quad = lib.get_surprisals_per_model(sentences=current_quad_sentences_cleaned, models=my_models)
    print("Surprisal data fetched for the quad.")

    # --- Save Surprisal Data to ONE CSV for the current quad ---
    csv_filename = f"{quad_name}_all_surprisals.csv"
    csv_filepath = os.path.join(output_directory, csv_filename)
    print(f"Attempting to save all surprisal data for {quad_name} to CSV: {os.path.abspath(csv_filepath)}")
    with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([
            "model_name", "condition_label", "original_sentence_text",
            "token_sequence_idx", "token_text", "token_original_word_idx", "surprisal_value"
        ])
        for model_name, sentence_surprisals_tuple in surprisal_data_per_model_for_quad.items():
            for sentence_idx, sentence_surprisal_obj in enumerate(sentence_surprisals_tuple):
                # The sentence_surprisal_obj corresponds to the order in current_quad_sentences_cleaned
                original_sentence_str = " ".join(sentence_surprisal_obj.original_sentence.original_token_strings)
                condition_label = current_quad_condition_labels[sentence_idx]
                for token_seq_idx, token_obj in enumerate(sentence_surprisal_obj.tokens):
                    csv_writer.writerow([
                        model_name, condition_label, original_sentence_str,
                        token_seq_idx, token_obj.text, token_obj.idx,
                        f"{token_obj.surprisal:.4f}" if token_obj.surprisal is not None else "N/A"
                    ])
    print(f"All surprisal data for {quad_name} saved to CSV: {os.path.abspath(csv_filepath)}")

    # --- Define pairs for plotting ---
    plot_pairs = [
        {
            "name": f"{quad_name}_what_obj-vs-no_obj",
            "sentences": (s_what_obj, s_what_no_obj),
            "labels": (l_what_obj, l_what_no_obj)
        },
        {
            "name": f"{quad_name}_that_obj-vs-no_obj",
            "sentences": (s_that_obj, s_that_no_obj),
            "labels": (l_that_obj, l_that_no_obj)
        },
        {
            "name": f"{quad_name}_obj_what-vs-that",
            "sentences": (s_what_obj, s_that_obj),
            "labels": (l_what_obj, l_that_obj)
        },
        {
            "name": f"{quad_name}_no-obj_what-vs-that",
            "sentences": (s_what_no_obj, s_that_no_obj),
            "labels": (l_what_no_obj, l_that_no_obj)
        }
    ]

    # --- Generate and Save Individual Plots for each pair ---
    for plot_info in plot_pairs:
        plot_filename = f"{plot_info['name']}_plot.png"
        plot_filepath = os.path.join(output_directory, plot_filename)
        print(f"Attempting to save plot to: {os.path.abspath(plot_filepath)} for sentences: {plot_info['labels']}")

        viz.plot_surprisals(
            sentences=plot_info["sentences"],
            models=my_models,
            labels=plot_info["labels"],
            save_to=plot_filepath
        )
        print(f"Plot saved as {os.path.abspath(plot_filepath)}")
    
    quad_end_time = time.time()
    print(f"Quad {quad_name} (CSV and 4 plots) finished in {quad_end_time - quad_start_time:.2f} seconds.")

overall_end_time = time.time()
print(f"\n--- All quads processed. Total script time: {overall_end_time - overall_start_time:.2f} seconds. ---")