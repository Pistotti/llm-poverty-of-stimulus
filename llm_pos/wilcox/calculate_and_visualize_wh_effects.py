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
SENTENCE_COMPONENTS_BASENAME = "test_set.csv"
MODEL_NAME = "gpt2" # Should match the model used

# Input aggregated surprisal CSV
AGGREGATED_SURPRISALS_CSV_BASENAME = f"{SENTENCE_COMPONENTS_BASENAME.split('.')[0]}_{MODEL_NAME}_critical_region_surprisals_aggregated.csv"
AGGREGATED_SURPRISALS_CSV = os.path.join(RESULTS_INPUT_DIR, AGGREGATED_SURPRISALS_CSV_BASENAME)

# Output CSV for wh-effects
WH_EFFECTS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{SENTENCE_COMPONENTS_BASENAME.split('.')[0]}_{MODEL_NAME}_wh_effects.csv")

# Output plot file for individual item wh-effects
WH_EFFECTS_PLOT_PNG = os.path.join(ANALYSIS_OUTPUT_DIR, f"{SENTENCE_COMPONENTS_BASENAME.split('.')[0]}_{MODEL_NAME}_wh_effects_plot.png")
# Output plot file for average wh-effects
AVERAGE_WH_EFFECTS_PLOT_PNG = os.path.join(ANALYSIS_OUTPUT_DIR, f"{SENTENCE_COMPONENTS_BASENAME.split('.')[0]}_{MODEL_NAME}_average_wh_effects_plot.png")
# --- End Configuration ---

def calculate_wh_effects(df):
    """
    Calculates wh-effects from the aggregated surprisal data.
    Expects a DataFrame with columns: 'item', 'condition', 'aggregated_surprisal_bits'.
    """
    effects = []
    df['item'] = df['item'].astype(str)

    for item, group in df.groupby('item'):
        try:
            surprisal_what_gap = group.loc[group['condition'] == 'what_gap', 'aggregated_surprisal_bits'].iloc[0]
            surprisal_that_gap = group.loc[group['condition'] == 'that_gap', 'aggregated_surprisal_bits'].iloc[0]
            surprisal_what_nogap = group.loc[group['condition'] == 'what_nogap', 'aggregated_surprisal_bits'].iloc[0]
            surprisal_that_nogap = group.loc[group['condition'] == 'that_nogap', 'aggregated_surprisal_bits'].iloc[0]

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
                'wh_effect_plus_gap': np.nan,
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
    Generates and saves a grouped bar plot of wh-effects per item.
    """
    if effects_df.empty or effects_df[['wh_effect_plus_gap', 'wh_effect_minus_gap']].isnull().all().all():
        print("No valid individual item data to plot. Skipping individual effects plot generation.")
        return

    plot_df = effects_df.melt(id_vars=['item'], 
                              value_vars=['wh_effect_plus_gap', 'wh_effect_minus_gap'],
                              var_name='effect_type', 
                              value_name='wh_effect_value')
    
    plot_df['effect_type'] = plot_df['effect_type'].replace({
        'wh_effect_plus_gap': 'Wh-Effect (+gap)',
        'wh_effect_minus_gap': 'Wh-Effect (-gap)'
    })

    plt.figure(figsize=(12, 7))
    sns.barplot(x='item', y='wh_effect_value', hue='effect_type', data=plot_df, palette="viridis")
    
    plt.title(f'Wh-Effects by Item ({MODEL_NAME} - {SENTENCE_COMPONENTS_BASENAME.split(".")[0]})', fontsize=16)
    plt.xlabel('Item ID', fontsize=14)
    plt.ylabel('Wh-Effect (Surprisal Difference in Bits)', fontsize=14)
    plt.axhline(0, color='grey', lw=1, linestyle='--')
    plt.legend(title='Effect Type', fontsize=12, title_fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    try:
        plt.savefig(output_path)
        print(f"Individual item effects plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving individual item effects plot to {output_path}: {e}")
    plt.close()

def plot_average_wh_effects(avg_plus_gap, avg_minus_gap, output_path):
    """
    Generates and saves a bar plot of average wh-effects.
    """
    if pd.isna(avg_plus_gap) and pd.isna(avg_minus_gap):
        print("Average wh-effects are both NaN. Skipping average effects plot generation.")
        return

    effect_types = ['Average Wh-Effect (+gap)', 'Average Wh-Effect (-gap)']
    average_values = [avg_plus_gap, avg_minus_gap]

    plt.figure(figsize=(8, 6))
    colors = [sns.color_palette("viridis")[0], sns.color_palette("viridis")[1]] # Match individual plot colors if possible
    
    bars = plt.bar(effect_types, average_values, color=colors)
    
    plt.title(f'Average Wh-Effects ({MODEL_NAME} - {SENTENCE_COMPONENTS_BASENAME.split(".")[0]})', fontsize=16)
    plt.ylabel('Average Wh-Effect (Surprisal Difference in Bits)', fontsize=14)
    plt.axhline(0, color='grey', lw=1, linestyle='--')
    
    # Adding the values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        if not pd.isna(yval):
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom' if yval < 0 else 'top', ha='center')
        else:
            plt.text(bar.get_x() + bar.get_width()/2.0, 0, 'NaN', va='bottom', ha='center')


    plt.tight_layout()
    
    try:
        plt.savefig(output_path)
        print(f"Average effects plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving average effects plot to {output_path}: {e}")
    plt.close()

def main():
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
        print(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")

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

    print("Calculating wh-effects for individual items...")
    wh_effects_df = calculate_wh_effects(aggregated_df)

    if not wh_effects_df.empty:
        try:
            wh_effects_df.to_csv(WH_EFFECTS_CSV, index=False, float_format='%.8f')
            print(f"Calculated individual wh-effects saved to {WH_EFFECTS_CSV}")
        except Exception as e:
            print(f"Error saving wh-effects CSV to {WH_EFFECTS_CSV}: {e}")
        
        print("Generating plot for individual item wh-effects...")
        plot_wh_effects(wh_effects_df, WH_EFFECTS_PLOT_PNG)

        # Calculate and print average wh-effects
        # .mean() automatically skips NaN values
        avg_plus_gap = wh_effects_df['wh_effect_plus_gap'].mean()
        avg_minus_gap = wh_effects_df['wh_effect_minus_gap'].mean()

        print("\n--- Average Wh-Effects ---")
        print(f"Average Wh-Effect (+gap): {avg_plus_gap:.4f}" if not pd.isna(avg_plus_gap) else "Average Wh-Effect (+gap): NaN")
        print(f"Average Wh-Effect (-gap): {avg_minus_gap:.4f}" if not pd.isna(avg_minus_gap) else "Average Wh-Effect (-gap): NaN")
        
        print("\nGenerating plot for average wh-effects...")
        plot_average_wh_effects(avg_plus_gap, avg_minus_gap, AVERAGE_WH_EFFECTS_PLOT_PNG)

    else:
        print("No wh-effects were calculated. Skipping CSV saving and plot generation.")
        
    print("\nScript finished.")

if __name__ == "__main__":
    main()