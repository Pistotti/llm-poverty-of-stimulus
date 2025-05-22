import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Configuration ---
# Assumes this script is in 'phase2/scripts/'
# CURRENT_SCRIPT_DIR will be '.../phase2/scripts'
# BASE_DIR will be '.../phase2'
# RESULTS_DIR will be '.../phase2/results'

try:
    CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)
except NameError: # Fallback for environments like notebooks
    print("Warning: __file__ not defined. Assuming current working directory is 'phase2/scripts' or 'phase2'.")
    CURRENT_SCRIPT_DIR = os.getcwd()
    if os.path.basename(CURRENT_SCRIPT_DIR) == "scripts":
        BASE_DIR = os.path.dirname(CURRENT_SCRIPT_DIR)
    elif os.path.basename(CURRENT_SCRIPT_DIR) == "phase2":
        BASE_DIR = CURRENT_SCRIPT_DIR
    else:
        # If structure is unknown, assume results dir is in CWD's parent's results
        # This might need manual adjustment by the user if it's not right
        print(f"Warning: Unknown directory structure. Current CWD: {CURRENT_SCRIPT_DIR}")
        print("RESULTS_DIR will be set relative to CWD. Please verify.")
        # Defaulting to a common case: script is in a 'scripts' dir, results in parallel 'results' dir
        BASE_DIR = os.path.dirname(CURRENT_SCRIPT_DIR) # Assuming CWD is 'scripts', BASE_DIR is its parent


RESULTS_DIR = os.path.join(BASE_DIR, "results")
INPUT_CSV_PATH = os.path.join(RESULTS_DIR, "gpt2-large_critical_region_surprisals.csv")
OUTPUT_WH_EFFECTS_CSV_PATH = os.path.join(RESULTS_DIR, "gpt2-large_wh_effects.csv")
OUTPUT_VISUALIZATION_PATH = os.path.join(RESULTS_DIR, "gpt2-large_average_wh_effects_visualization.png")

def calculate_wh_effects(df):
    """
    Calculates wh-effects from the aggregated surprisal data.
    """
    wh_effects_data = []
    df['item'] = df['item'].astype(str) # Ensure item is string for consistent grouping
    grouped = df.groupby(['source_doc_name', 'item'])

    for name, group in grouped:
        source_doc, item_id = name
        
        conditions_data = {}
        critical_region_texts = {} 

        for _, row in group.iterrows():
            conditions_data[row['condition']] = row['aggregated_surprisal_bits']
            critical_region_texts[row['condition']] = row['critical_region_text']

        required_nogap_conditions = ['that_nogap', 'what_nogap']
        required_gap_conditions = ['that_gap', 'what_gap']

        wh_effect_nogap = np.nan
        wh_effect_gap = np.nan
        
        critical_region_nogap = critical_region_texts.get('that_nogap', "") 
        if not critical_region_nogap and 'what_nogap' in critical_region_texts:
             critical_region_nogap = critical_region_texts.get('what_nogap', "")

        critical_region_gap = critical_region_texts.get('that_gap', "")
        if not critical_region_gap and 'what_gap' in critical_region_texts:
            critical_region_gap = critical_region_texts.get('what_gap', "")

        if all(c in conditions_data for c in required_nogap_conditions):
            s_what_nogap = conditions_data['what_nogap']
            s_that_nogap = conditions_data['that_nogap']
            if pd.notna(s_what_nogap) and pd.notna(s_that_nogap):
                wh_effect_nogap = s_what_nogap - s_that_nogap

        if all(c in conditions_data for c in required_gap_conditions):
            s_what_gap = conditions_data['what_gap']
            s_that_gap = conditions_data['that_gap']
            if pd.notna(s_what_gap) and pd.notna(s_that_gap):
                wh_effect_gap = s_what_gap - s_that_gap
        
        if pd.notna(wh_effect_nogap) or pd.notna(wh_effect_gap):
            wh_effects_data.append({
                'source_doc_name': source_doc,
                'item': item_id,
                'wh_effect_nogap_bits': wh_effect_nogap,
                'critical_region_nogap': critical_region_nogap,
                'wh_effect_gap_bits': wh_effect_gap,
                'critical_region_gap': critical_region_gap
            })
        else:
            print(f"Info: Could not calculate wh-effects for item {item_id} in {source_doc} (missing conditions or NaN surprisals).")
            # print(f"  Available conditions data for this item: {conditions_data}") # Uncomment for debugging

    return pd.DataFrame(wh_effects_data)

def visualize_average_wh_effects(wh_effects_df, output_path):
    """
    Creates and saves a bar chart of average wh-effects.
    """
    if wh_effects_df.empty:
        print("Wh-effects DataFrame is empty. Cannot generate visualization.")
        return

    avg_wh_effect_nogap = wh_effects_df['wh_effect_nogap_bits'].mean(skipna=True)
    avg_wh_effect_gap = wh_effects_df['wh_effect_gap_bits'].mean(skipna=True)

    labels = ['-gap (No Gap Context)', '+gap (Gap Context)']
    averages = [avg_wh_effect_nogap, avg_wh_effect_gap]

    valid_labels = []
    valid_averages = []
    if pd.notna(avg_wh_effect_nogap):
        valid_labels.append(labels[0])
        valid_averages.append(avg_wh_effect_nogap)
    if pd.notna(avg_wh_effect_gap):
        valid_labels.append(labels[1])
        valid_averages.append(avg_wh_effect_gap)

    if not valid_averages: 
        print("All average wh-effects are NaN. Cannot generate visualization.")
        return

    x = np.arange(len(valid_labels))
    width = 0.35

    plt.switch_backend('Agg') # Use a non-interactive backend for headless server
    fig, ax = plt.subplots(figsize=(10, 7))
    rects = ax.bar(x, valid_averages, width, label='Average WH-Effect', color=['skyblue', 'salmon'])

    ax.set_ylabel('Average WH-Effect (bits)')
    ax.set_title('Average WH-Effects by Condition Type (GPT-2 Large)')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, rotation=0, ha="center")
    ax.legend()
    ax.axhline(0, color='grey', lw=0.8)

    for i, rect in enumerate(rects):
        height = valid_averages[i]
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top')

    fig.tight_layout()
    try:
        # Ensure the RESULTS_DIR exists before saving
        if not os.path.exists(os.path.dirname(output_path)):
            try:
                os.makedirs(os.path.dirname(output_path))
                print(f"Created directory for visualization: {os.path.dirname(output_path)}")
            except OSError as e_dir:
                print(f"Error: Could not create directory {os.path.dirname(output_path)} for visualization: {e_dir}")
                print("Please check permissions or manually create the directory.")
                plt.close(fig)
                return

        plt.savefig(output_path)
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    plt.close(fig)

def main():
    # Ensure results directory exists or can be created (for output files)
    # This is mainly for the CSV output; visualization has its own check.
    if not os.path.exists(RESULTS_DIR):
        try:
            os.makedirs(RESULTS_DIR)
            print(f"Created results directory: {RESULTS_DIR}")
        except OSError as e:
            print(f"Critical Error: Could not create results directory {RESULTS_DIR}: {e}")
            print(f"Please ensure you have write permissions for the path or manually create the directory: {os.path.abspath(RESULTS_DIR)}")
            return

    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: Input CSV file not found at {INPUT_CSV_PATH}")
        print(f"Please ensure '{os.path.basename(INPUT_CSV_PATH)}' exists in '{os.path.abspath(RESULTS_DIR)}'.")
        return

    print(f"Loading aggregated surprisals from {INPUT_CSV_PATH}...")
    try:
        aggregated_df = pd.read_csv(INPUT_CSV_PATH)
    except Exception as e:
        print(f"Error reading input CSV '{INPUT_CSV_PATH}': {e}")
        return

    if aggregated_df.empty:
        print("Input CSV is empty. No wh-effects to calculate.")
        return
    
    required_cols = ['source_doc_name', 'item', 'condition', 'aggregated_surprisal_bits', 'critical_region_text']
    if not all(col in aggregated_df.columns for col in required_cols):
        print(f"Error: Input CSV is missing one or more required columns. Needed: {required_cols}")
        print(f"Available columns: {aggregated_df.columns.tolist()}")
        return

    print("Calculating wh-effects...")
    wh_effects_df = calculate_wh_effects(aggregated_df)

    if not wh_effects_df.empty:
        try:
            wh_effects_df.to_csv(OUTPUT_WH_EFFECTS_CSV_PATH, index=False, float_format='%.8f') 
            print(f"WH-effect calculations saved to {OUTPUT_WH_EFFECTS_CSV_PATH}")
        except Exception as e:
            print(f"Error saving wh-effects CSV: {e}")
        
        print("Generating visualization...")
        visualize_average_wh_effects(wh_effects_df, OUTPUT_VISUALIZATION_PATH)
    else:
        print("No wh-effects were calculated (e.g. due to missing conditions for all items or all surprisals being NaN).")
        print("Output CSV and visualization will not be generated.")

    print("Script finished.")

if __name__ == "__main__":
    main()