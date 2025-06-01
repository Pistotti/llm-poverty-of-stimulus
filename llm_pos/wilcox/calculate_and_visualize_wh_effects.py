# llm-poverty-of-stimulus/llm_pos/wilcox/analyze_and_visualize_wh_effects.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "gpt2" # Should match the model used

# Input: Output from the new aggregate_surprisals.py
# This file should contain 'sentence_type', 'item_id', 'condition', 'aggregated_surprisal_bits'
AGGREGATED_INPUT_CSV = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "aggregated", f"{MODEL_NAME}_critical_regions_aggregated.csv")

# Output directory for analysis results
ANALYSIS_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "analysis")
# Output CSV for wh-effects (will include sentence_type)
WH_EFFECTS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_wh_effects_basic_types.csv")

# Output plot file for average wh-effects for basic types
AVERAGE_WH_EFFECTS_PLOT_PNG = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_average_wh_effects_basic_types_plot.png")
# Output directory for per-item plots for basic types
PER_ITEM_PLOT_DIR = os.path.join(ANALYSIS_OUTPUT_DIR, "per_item_plots_basic")

# Sentence types to focus on for this version
TARGET_SENTENCE_TYPES = ["basic_object", "basic_pp", "basic_subject"]
# --- End Configuration ---

def calculate_wh_effects_for_selected_types(df, sentence_types_to_process):
    """
    Calculates wh-effects from aggregated surprisal data for specified sentence types.
    Expects df columns: 'sentence_type', 'item_id', 'condition', 'aggregated_surprisal_bits'.
    """
    effects_data = []
    
    # Filter for target sentence types
    df_filtered = df[df['sentence_type'].isin(sentence_types_to_process)].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    if df_filtered.empty:
        print(f"Warning: No data found for target sentence types: {sentence_types_to_process}")
        return pd.DataFrame(effects_data)

    df_filtered['item_id'] = df_filtered['item_id'].astype(str)
    df_filtered['sentence_type'] = df_filtered['sentence_type'].astype(str)

    # Group by sentence_type and then by item_id
    for (sentence_type, item_id), group in df_filtered.groupby(['sentence_type', 'item_id']):
        
        # For basic types, condition names are expected to be exactly these
        # (e.g., "what_gap", "that_gap", etc., without additional suffixes)
        # If your actual condition names have suffixes even for basic types, this part needs adjustment.
        cond_wg = "what_gap"
        cond_tg = "that_gap"
        cond_wn = "what_nogap"
        cond_tn = "that_nogap"
        
        # Check if this item has these exact 4 conditions.
        # This strict matching is suitable for basic types where condition names are simple.
        s_wg_series = group.loc[group['condition'] == cond_wg, 'aggregated_surprisal_bits']
        s_tg_series = group.loc[group['condition'] == cond_tg, 'aggregated_surprisal_bits']
        s_wn_series = group.loc[group['condition'] == cond_wn, 'aggregated_surprisal_bits']
        s_tn_series = group.loc[group['condition'] == cond_tn, 'aggregated_surprisal_bits']

        if s_wg_series.empty or s_tg_series.empty or s_wn_series.empty or s_tn_series.empty:
            # print(f"  Debug: Item {item_id} (Type: {sentence_type}) missing one or more standard conditions ({cond_wg}, {cond_tg}, {cond_wn}, {cond_tn}). Skipping wh-effect for this item.")
            continue # Skip this item if the 4 conditions aren't present

        try:
            s_wg = s_wg_series.iloc[0]
            s_tg = s_tg_series.iloc[0]
            s_wn = s_wn_series.iloc[0]
            s_tn = s_tn_series.iloc[0]

            if pd.isna(s_wg) or pd.isna(s_tg) or pd.isna(s_wn) or pd.isna(s_tn):
                # print(f"Warning: Item {item_id} (Type: {sentence_type}) has NaN surprisal values. Wh-effects will be NaN.")
                wh_plus_gap = np.nan
                wh_minus_gap = np.nan
            else:
                wh_plus_gap = s_wg - s_tg
                wh_minus_gap = s_wn - s_tn
            
            effects_data.append({
                'sentence_type': sentence_type,
                'item_id': item_id,
                # 'condition_stem': '', # Not strictly needed if conditions are simple for basic types
                'wh_effect_plus_gap': wh_plus_gap,
                'wh_effect_minus_gap': wh_minus_gap
            })
        except Exception as e: # General exception if .iloc[0] fails for unexpected reasons
            print(f"Error processing item {item_id} (Type: {sentence_type}): {e}")
            effects_data.append({
                'sentence_type': sentence_type,
                'item_id': item_id,
                # 'condition_stem': '',
                'wh_effect_plus_gap': np.nan,
                'wh_effect_minus_gap': np.nan
            })
            
    if not effects_data:
        print(f"Warning: No items found with the required four conditions for wh-effect calculation among selected types: {sentence_types_to_process}.")
    
    return pd.DataFrame(effects_data)


def plot_average_wh_effects_by_type(effects_df, output_path, title_suffix=""):
    if effects_df.empty or effects_df[['wh_effect_plus_gap', 'wh_effect_minus_gap']].isnull().all().all():
        print("No valid data to plot for average wh-effects by type. Skipping plot generation.")
        return

    avg_effects = effects_df.groupby('sentence_type')[['wh_effect_plus_gap', 'wh_effect_minus_gap']].mean().reset_index()

    plot_df = avg_effects.melt(id_vars=['sentence_type'],
                               value_vars=['wh_effect_plus_gap', 'wh_effect_minus_gap'],
                               var_name='effect_type',
                               value_name='average_wh_effect_value')

    plot_df['effect_type'] = plot_df['effect_type'].replace({
        'wh_effect_plus_gap': 'Avg Wh-Effect (+gap)',
        'wh_effect_minus_gap': 'Avg Wh-Effect (-gap)'
    })

    plt.figure(figsize=(max(10, len(avg_effects['sentence_type'].unique()) * 2.5), 7)) # Dynamic width
    sns.barplot(x='sentence_type', y='average_wh_effect_value', hue='effect_type', data=plot_df, palette="viridis")
    
    plt.title(f'Average Wh-Effects by Basic Sentence Type ({MODEL_NAME}){title_suffix}', fontsize=16)
    plt.xlabel('Sentence Type', fontsize=14)
    plt.ylabel('Average Wh-Effect (Surprisal Bits)', fontsize=14)
    plt.axhline(0, color='grey', lw=1, linestyle='--')
    plt.legend(title='Effect Type', fontsize=12, title_fontsize=13)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    try:
        plt.savefig(output_path)
        print(f"Average wh-effects plot saved to {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving average wh-effects plot: {e}")
    plt.close()


def plot_item_effects_by_type(effects_df, output_dir_base, title_suffix=""):
    if effects_df.empty or effects_df[['wh_effect_plus_gap', 'wh_effect_minus_gap']].isnull().all().all():
        print("No valid individual item data to plot. Skipping per-item plot generation.")
        return

    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
        print(f"Created directory for per-item plots: {output_dir_base}")

    for sentence_type, group_df in effects_df.groupby('sentence_type'):
        if group_df.empty or group_df[['wh_effect_plus_gap', 'wh_effect_minus_gap']].isnull().all().all():
            print(f"No data or only NaN data to plot for items in sentence type '{sentence_type}'. Skipping this plot.")
            continue
            
        plot_df = group_df.melt(id_vars=['item_id'], 
                                  value_vars=['wh_effect_plus_gap', 'wh_effect_minus_gap'],
                                  var_name='effect_type', 
                                  value_name='wh_effect_value')
        
        plot_df['effect_type'] = plot_df['effect_type'].replace({
            'wh_effect_plus_gap': 'Wh-Effect (+gap)',
            'wh_effect_minus_gap': 'Wh-Effect (-gap)'
        })

        num_items = len(group_df['item_id'].unique())
        fig_width = max(10, num_items * 0.6) 
        
        plt.figure(figsize=(fig_width, 7))
        sns.barplot(x='item_id', y='wh_effect_value', hue='effect_type', data=plot_df, palette="viridis")
        
        plt.title(f'Wh-Effects by Item for {sentence_type} ({MODEL_NAME}){title_suffix}', fontsize=16)
        plt.xlabel('Item ID', fontsize=14)
        plt.ylabel('Wh-Effect (Surprisal Bits)', fontsize=14)
        plt.axhline(0, color='grey', lw=1, linestyle='--')
        plt.legend(title='Effect Type', fontsize=12, title_fontsize=13)
        plt.xticks(rotation=45, ha='right' if num_items > 15 else 'center') 
        plt.tight_layout()
        
        plot_filename = os.path.join(output_dir_base, f"{sentence_type}_{MODEL_NAME}_item_wh_effects.png")
        try:
            plt.savefig(plot_filename)
            print(f"Per-item effects plot for {sentence_type} saved to {os.path.abspath(plot_filename)}")
        except Exception as e:
            print(f"Error saving per-item plot for {sentence_type}: {e}")
        plt.close()

def main():
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
        print(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")

    if not os.path.exists(AGGREGATED_INPUT_CSV):
        print(f"Critical Error: Aggregated surprisals file not found at '{os.path.abspath(AGGREGATED_INPUT_CSV)}'")
        return

    print(f"Loading aggregated surprisals from {AGGREGATED_INPUT_CSV}...")
    try:
        aggregated_df_full = pd.read_csv(AGGREGATED_INPUT_CSV)
    except Exception as e:
        print(f"Error loading {AGGREGATED_INPUT_CSV}: {e}")
        return
        
    if aggregated_df_full.empty:
        print("Aggregated surprisals file is empty.")
        return
    
    required_cols = ['sentence_type', 'item_id', 'condition', 'aggregated_surprisal_bits']
    missing_cols = [col for col in required_cols if col not in aggregated_df_full.columns]
    if missing_cols:
        print(f"Error: Aggregated input CSV is missing required columns: {', '.join(missing_cols)}")
        print(f"Found columns: {aggregated_df_full.columns.tolist()}")
        return

    print(f"Filtering for basic sentence types: {TARGET_SENTENCE_TYPES}")
    aggregated_df_basic = aggregated_df_full[aggregated_df_full['sentence_type'].isin(TARGET_SENTENCE_TYPES)]

    if aggregated_df_basic.empty:
        print(f"No data found for the specified basic sentence types: {TARGET_SENTENCE_TYPES}. Exiting.")
        return

    print("Calculating wh-effects for basic sentence types and their items...")
    wh_effects_df_basic = calculate_wh_effects_for_selected_types(aggregated_df_basic, TARGET_SENTENCE_TYPES)

    if not wh_effects_df_basic.empty:
        try:
            wh_effects_df_basic.to_csv(WH_EFFECTS_CSV, index=False, float_format='%.8f')
            print(f"Calculated wh-effects for basic types saved to {os.path.abspath(WH_EFFECTS_CSV)}")
        except Exception as e:
            print(f"Error saving wh-effects CSV: {e}")
        
        print("\nGenerating plot for average wh-effects by basic sentence type...")
        plot_average_wh_effects_by_type(wh_effects_df_basic, AVERAGE_WH_EFFECTS_PLOT_PNG)

        print("\nGenerating plots for per-item wh-effects for basic sentence types...")
        plot_item_effects_by_type(wh_effects_df_basic, PER_ITEM_PLOT_DIR)
    else:
        print("No wh-effects were calculated for basic types (e.g., items might not have had the full 2x2 set of conditions).")
        
    print("\nScript finished.")

if __name__ == "__main__":
    main()