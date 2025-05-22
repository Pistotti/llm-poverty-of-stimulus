# llm-poverty-of-stimulus/llm_pos/wilcox/analyze_and_visualize_wh_effects.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
# Assume this script is in llm_pos/wilcox/
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Input directory for aggregated surprisals
RESULTS_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results")
# Output directory for plots and the new CSV with wh-effects
ANALYSIS_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "analysis") # Subdirectory for clarity

# Basename of the sentence components file, used to construct the input aggregated file name
# This should match SENTENCE_COMPONENTS_INPUT_CSV_BASENAME from aggregate_surprisals.py
SENTENCE_COMPONENTS_BASENAME = "test_set.csv"
MODEL_NAME = "gpt2" # Should match the model used

# Input aggregated surprisal CSV
AGGREGATED_SURPRISALS_CSV_BASENAME = f"{SENTENCE_COMPONENTS_BASENAME.split('.')[0]}_{MODEL_NAME}_critical_region_surprisals_aggregated.csv"
AGGREGATED_SURPRISALS_CSV = os.path.join(RESULTS_INPUT_DIR, AGGREGATED_SURPRISALS_CSV_BASENAME)

# Output CSV for wh-effects
WH_EFFECTS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{SENTENCE_COMPONENTS_BASENAME.split('.')[0]}_{MODEL_NAME}_wh_effects.csv")

# Output plot file
WH_EFFECTS_PLOT_PNG = os.path.join(ANALYSIS_OUTPUT_DIR, f"{SENTENCE_COMPONENTS_BASENAME.split('.')[0]}_{MODEL_NAME}_wh_effects_plot.png")
# --- End Configuration ---

def calculate_wh_effects(df):
    """
    Calculates wh-effects from the aggregated surprisal data.
    Expects a DataFrame with columns: 'item', 'condition', 'aggregated_surprisal_bits'.
    """
    effects = []
    
    # Ensure 'item' is treated consistently (e.g., as string if it can be numeric)
    df['item'] = df['item'].astype(str)

    for item, group in df.groupby('item'):
        try:
            surprisal_what_gap = group.loc[group['condition'] == 'what_gap', 'aggregated_surprisal_bits'].iloc[0]
            surprisal_that_gap = group.loc[group['condition'] == 'that_gap', 'aggregated_surprisal_bits'].iloc[0]
            surprisal_what_nogap = group.loc[group['condition'] == 'what_nogap', 'aggregated_surprisal_bits'].iloc[0]
            surprisal_that_nogap = group.loc[group['condition'] == 'that_nogap', 'aggregated_surprisal_bits'].iloc[0]

            # Check for NaN before calculation (can happen if a critical region word alignment failed)
            if pd.isna(surprisal_what_gap) or pd.isna(surprisal_that_gap) or \
               pd.isna(surprisal_what_nogap) or pd.isna(surprisal_that_nogap):
                print(f"Warning: Item {item} has NaN surprisal values for one or more conditions. Wh-effects will be NaN.")
                wh_effect_plus_gap = np.nan
                wh_effect_minus_gap = np.nan
            else:
                wh_effect_plus_gap = surprisal_what_gap - surprisal_that_gap
                wh_effect_minus_gap = surprisal_what_nogap - surprisal_that_nogap
            
            effects.append({
                'item': item,
                'wh_effect_plus_gap': wh_effect_plus_gap,
                'wh_effect_minus_gap': wh_effect_minus_gap
            })
        except IndexError:
            print(f"Warning: Item {item} is missing one or more required conditions (what_gap, that_gap, what_nogap, that_nogap). Skipping this item for wh-effect calculation.")
            effects.append({
                'item': item,
                'wh_effect_plus_gap': np.nan, # Use np.nan for missing data
                'wh_effect_minus_gap': np.nan
            })
        except Exception as e:
            print(f"Error processing item {item}: {e}")
            effects.append({
                'item': item,
                'wh_effect_plus_gap': np.nan,
                'wh_effect_minus_gap': np.nan
            })
            
    return pd.DataFrame(effects)

def plot_wh_effects(effects_df, output_path):
    """
    Generates and saves a grouped bar plot of wh-effects.
    """
    if effects_df.empty or effects_df[['wh_effect_plus_gap', 'wh_effect_minus_gap']].isnull().all().all():
        print("No valid data to plot. Skipping plot generation.")
        return

    # Melt the DataFrame for easy plotting with seaborn
    plot_df = effects_df.melt(id_vars=['item'], 
                              value_vars=['wh_effect_plus_gap', 'wh_effect_minus_gap'],
                              var_name='effect_type', 
                              value_name='wh_effect_value')
    
    # Clean up effect type names for legend
    plot_df['effect_type'] = plot_df['effect_type'].replace({
        'wh_effect_plus_gap': 'Wh-Effect (+gap)',
        'wh_effect_minus_gap': 'Wh-Effect (-gap)'
    })

    plt.figure(figsize=(12, 7)) # Adjusted figure size
    sns.barplot(x='item', y='wh_effect_value', hue='effect_type', data=plot_df, palette="viridis")
    
    plt.title(f'Wh-Effects by Item ({MODEL_NAME} - {SENTENCE_COMPONENTS_BASENAME.split(".")[0]})', fontsize=16)
    plt.xlabel('Item ID', fontsize=14)
    plt.ylabel('Wh-Effect (Surprisal Difference in Bits)', fontsize=14)
    plt.axhline(0, color='grey', lw=1, linestyle='--') # Add a zero line for reference
    plt.legend(title='Effect Type', fontsize=12, title_fontsize=13)
    plt.xticks(rotation=45, ha='right') # Rotate x-axis labels if many items
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    
    try:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving plot to {output_path}: {e}")
    plt.close()

def main():
    # Create output directory if it doesn't exist
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
        print(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")

    # Check if input file exists
    if not os.path.exists(AGGREGATED_SURPRISALS_CSV):
        print(f"Critical Error: Aggregated surprisals file not found at '{AGGREGATED_SURPRISALS_CSV}'")
        print("Please ensure 'aggregate_surprisals.py' has been run successfully and the paths are correct.")
        return

    print(f"Loading aggregated surprisals from {AGGREGATED_SURPRISALS_CSV}...")
    try:
        aggregated_df = pd.read_csv(AGGREGATED_SURPRISALS_CSV)
    except Exception as e:
        print(f"Error loading {AGGREGATED_SURPRISALS_CSV}: {e}")
        return
        
    if aggregated_df.empty:
        print("Aggregated surprisals file is empty. Cannot calculate wh-effects.")
        return

    print("Calculating wh-effects...")
    wh_effects_df = calculate_wh_effects(aggregated_df)

    if not wh_effects_df.empty:
        try:
            wh_effects_df.to_csv(WH_EFFECTS_CSV, index=False, float_format='%.8f')
            print(f"Calculated wh-effects saved to {WH_EFFECTS_CSV}")
        except Exception as e:
            print(f"Error saving wh-effects CSV to {WH_EFFECTS_CSV}: {e}")
        
        print("Generating plot...")
        plot_wh_effects(wh_effects_df, WH_EFFECTS_PLOT_PNG)
    else:
        print("No wh-effects were calculated. Skipping CSV saving and plot generation.")
        
    print("Script finished.")

if __name__ == "__main__":
    main()