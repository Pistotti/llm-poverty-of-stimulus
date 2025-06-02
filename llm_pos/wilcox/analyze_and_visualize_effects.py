# llm-poverty-of-stimulus/llm_pos/wilcox/analyze_and_visualize_effects.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re 
from scipy import stats # For statistical tests

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "gpt2" 

# CHOOSE ONE: "WILCOX", "LAN", "NOVEL_SUBJECT_PG"
DATASET_TYPE = "NOVEL_SUBJECT_PG" 

MAX_ITEMS_FOR_PER_ITEM_PLOT = 200 

# Configuration flags for filtering original LAN data (not applicable to NOVEL_SUBJECT_PG)
# These will only take effect if DATASET_TYPE == "LAN"
FILTER_LAN_GENITIVE_GERUNDS = True 
FILTER_LAN_INTENT_TO = True        

# --- Conditional Configuration based on DATASET_TYPE ---
if DATASET_TYPE == "WILCOX":
    # Inputs
    AGGREGATED_INPUT_FILENAME = f"{MODEL_NAME}_critical_regions_aggregated.csv"
    AGGREGATED_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "aggregated")
    # Targets
    TARGET_SENTENCE_TYPES = ["basic_object", "basic_pp", "basic_subject"]
    # Effect Naming
    EFFECT_TYPE_LABEL = "Wh-Effect" 
    EFFECT1_NAME = 'wh_effect_plus_gap' 
    EFFECT2_NAME = 'wh_effect_minus_gap'
    EFFECT1_PLOT_LABEL = 'Avg Wh-Effect (+gap)'
    EFFECT2_PLOT_LABEL = 'Avg Wh-Effect (-gap)'
    ITEM_EFFECT1_PLOT_LABEL = 'Wh-Effect (+gap)'
    ITEM_EFFECT2_PLOT_LABEL = 'Wh-Effect (-gap)'
    # Output Suffixes and Dirs
    ANALYSIS_SUFFIX = "wilcox_basic_types"
    OUTPUT_SUBDIR = DATASET_TYPE.lower()

elif DATASET_TYPE == "LAN":
    # Inputs
    AGGREGATED_INPUT_FILENAME = f"lan_extracted_critical_surprisals_{MODEL_NAME}.csv" # Output from script like extract_lan_critical_surprisals.py
    AGGREGATED_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "aggregated") # Assuming it's placed here
    # Targets
    TARGET_SENTENCE_TYPES = ["lan_parasitic_gap"] # Or other LAN types like "lan_atb_movement"
    # Effect Naming (using Lan's Delta metrics as primary "effects" for plotting)
    EFFECT_TYPE_LABEL = "Lan Metric" 
    EFFECT1_NAME = 'delta_plus_filler'  
    EFFECT2_NAME = 'did_effect' 
    EFFECT1_PLOT_LABEL = 'Avg Delta_Plus_Filler' 
    EFFECT2_PLOT_LABEL = 'Avg DiD Effect' 
    ITEM_EFFECT1_PLOT_LABEL = 'Delta_Plus_Filler'
    ITEM_EFFECT2_PLOT_LABEL = 'DiD Effect'
    # Output Suffixes and Dirs
    ANALYSIS_SUFFIX = "lan_pg_types" # Be specific
    OUTPUT_SUBDIR = DATASET_TYPE.lower()
    
elif DATASET_TYPE == "NOVEL_SUBJECT_PG":
    # Inputs
    AGGREGATED_INPUT_FILENAME = f"novel_data_extracted_critical_surprisals_{MODEL_NAME}.csv" # Your new file
    AGGREGATED_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "aggregated")
    # Targets
    TARGET_SENTENCE_TYPES = ["subject_gap"] # The sentence_type in your novel_data.csv
    # Effect Naming (using Lan's Delta metrics as primary "effects" for plotting)
    EFFECT_TYPE_LABEL = "Lan Metric (Novel Data)" 
    EFFECT1_NAME = 'delta_plus_filler'  
    EFFECT2_NAME = 'did_effect' 
    EFFECT1_PLOT_LABEL = 'Avg Delta_Plus_Filler' 
    EFFECT2_PLOT_LABEL = 'Avg DiD Effect' 
    ITEM_EFFECT1_PLOT_LABEL = 'Delta_Plus_Filler'
    ITEM_EFFECT2_PLOT_LABEL = 'DiD Effect'
    # Output Suffixes and Dirs
    ANALYSIS_SUFFIX = "novel_subject_pg"
    OUTPUT_SUBDIR = "novel_data_findings" # New dedicated output folder

else:
    raise ValueError(f"Unsupported DATASET_TYPE: {DATASET_TYPE}")

AGGREGATED_INPUT_CSV = os.path.join(AGGREGATED_INPUT_DIR, AGGREGATED_INPUT_FILENAME)

# --- Determine Filter Suffix for Output Filenames (Only for LAN type) ---
FILTER_SUFFIX = ""
active_filter_descriptions = []
if DATASET_TYPE == "LAN":
    if FILTER_LAN_GENITIVE_GERUNDS:
        active_filter_descriptions.append("no_gen_gerunds") # More descriptive
    if FILTER_LAN_INTENT_TO:
        active_filter_descriptions.append("no_intent_to")
    if active_filter_descriptions:
        FILTER_SUFFIX = "_filtered_" + "_".join(active_filter_descriptions)

# --- Define Output Paths ---
ANALYSIS_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", OUTPUT_SUBDIR) # Main analysis subdir
CALCULATED_EFFECTS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_effects{FILTER_SUFFIX}.csv") 
AVERAGE_EFFECTS_PLOT_PNG = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_average_{ANALYSIS_SUFFIX}_effects_plot{FILTER_SUFFIX}.png")
PER_ITEM_PLOT_DIR = os.path.join(ANALYSIS_OUTPUT_DIR, f"per_item_plots_{ANALYSIS_SUFFIX}{FILTER_SUFFIX}")

# Specific to LAN-style analysis (and adaptable for NOVEL_SUBJECT_PG)
LAN_STYLE_ANALYSIS_APPLIES = (DATASET_TYPE == "LAN" or DATASET_TYPE == "NOVEL_SUBJECT_PG")
if LAN_STYLE_ANALYSIS_APPLIES:
    LAN_ITEM_CONDITION_SURPRISALS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_item_condition_surprisals{FILTER_SUFFIX}.csv")
    LAN_CONDITION_SUMMARY_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_condition_surprisal_summary{FILTER_SUFFIX}.csv")
    LAN_PAPER_METRICS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_lan_paper_metrics{FILTER_SUFFIX}.csv")
    LAN_PAPER_ACCURACY_SUMMARY_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_lan_paper_accuracy_summary{FILTER_SUFFIX}.csv")
    STAT_RESULTS_OUTPUT_TXT = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_stat_summary{FILTER_SUFFIX}.txt")

# --- End Configuration ---

# --- Helper Functions --- (calculate_effects, plot_average_effects, plot_item_effects)
# These functions will now use EFFECT1_NAME, EFFECT2_NAME etc. from config

def calculate_wilcox_effects(df, sentence_types_to_process):
    # This function is specific to Wilcox style conditions "what_gap", "that_gap", etc.
    effects_data = []
    df_filtered = df[df['sentence_type'].isin(sentence_types_to_process)].copy()
    if df_filtered.empty: return pd.DataFrame(effects_data)

    df_filtered['item_id'] = df_filtered['item_id'].astype(str)

    for (sentence_type, item_id), group in df_filtered.groupby(['sentence_type', 'item_id']):
        s_wg = group.loc[group['condition'] == 'what_gap', 'aggregated_surprisal_bits']
        s_tg = group.loc[group['condition'] == 'that_gap', 'aggregated_surprisal_bits']
        s_wn = group.loc[group['condition'] == 'what_nogap', 'aggregated_surprisal_bits']
        s_tn = group.loc[group['condition'] == 'that_nogap', 'aggregated_surprisal_bits']

        effect1, effect2 = np.nan, np.nan
        if not (s_wg.empty or s_tg.empty or s_wn.empty or s_tn.empty):
            s_wg_val, s_tg_val, s_wn_val, s_tn_val = s_wg.iloc[0], s_tg.iloc[0], s_wn.iloc[0], s_tn.iloc[0]
            if not pd.isna([s_wg_val, s_tg_val, s_wn_val, s_tn_val]).any():
                effect1 = s_wg_val - s_tg_val  # wh_effect_plus_gap
                effect2 = s_wn_val - s_tn_val  # wh_effect_minus_gap
        
        effects_data.append({'sentence_type': sentence_type, 'item_id': item_id, 
                             EFFECT1_NAME: effect1, EFFECT2_NAME: effect2})
    return pd.DataFrame(effects_data)

def summarize_lan_style_condition_surprisals(df, target_sentence_types, item_output_path, summary_output_path):
    df_filtered = df[df['sentence_type'].isin(target_sentence_types)].copy()
    if df_filtered.empty: 
        print(f"Warning: No data for {target_sentence_types} in summarize_lan_style_condition_surprisals."); return None
    
    item_level_surprisals = None
    try:
        item_level_surprisals = df_filtered.pivot_table(
            index=['sentence_type', 'item_id'], 
            columns='condition', 
            values='aggregated_surprisal_bits'
        ).reset_index()
    except Exception as e: 
        print(f"Error pivoting data for LAN/Novel summary: {e}"); return None
    
    # --- START FIX for column names ---
    # Define the mapping from short condition names (from pivot) to long expected names
    condition_rename_map = {
        "PFPG": "PLUS_FILLER_PLUS_GAP",
        "MFPG": "MINUS_FILLER_PLUS_GAP",
        "PFMG": "PLUS_FILLER_MINUS_GAP",
        "MFMG": "MINUS_FILLER_MINUS_GAP"
    }
    
    # Rename columns if they exist with short names
    cols_to_rename_actually_present = {k: v for k, v in condition_rename_map.items() if k in item_level_surprisals.columns}
    if cols_to_rename_actually_present:
        item_level_surprisals.rename(columns=cols_to_rename_actually_present, inplace=True)
        print(f"Renamed pivoted condition columns to: {cols_to_rename_actually_present.values()}")
    # --- END FIX ---
        
    lan_conditions_expected = ["PLUS_FILLER_PLUS_GAP", "MINUS_FILLER_PLUS_GAP", 
                               "PLUS_FILLER_MINUS_GAP", "MINUS_FILLER_MINUS_GAP"]
    
    # Ensure all 4 expected condition columns are present after potential rename, add as NaN if not
    for cond in lan_conditions_expected: 
        if cond not in item_level_surprisals.columns: 
            item_level_surprisals[cond] = np.nan
            print(f"Warning: Condition column '{cond}' was missing after pivot and rename, added as NaN.")

    cols_to_select = ['sentence_type', 'item_id'] + [c for c in lan_conditions_expected if c in item_level_surprisals.columns]
    item_level_surprisals = item_level_surprisals[cols_to_select]
    
    if not item_level_surprisals.empty:
        item_level_surprisals.to_csv(item_output_path, index=False, float_format='%.8f')
        print(f"Item-level condition surprisals saved to {os.path.abspath(item_output_path)}")
    else: 
        print("No item-level condition surprisals to save."); return None
    
    # ... (rest of the function for summary_df remains the same) ...
    numeric_cols_for_agg = [c for c in lan_conditions_expected if c in item_level_surprisals.columns and pd.api.types.is_numeric_dtype(item_level_surprisals[c])]
    if not numeric_cols_for_agg: 
        print("Warning: No numeric LAN condition columns found for summary aggregation."); return item_level_surprisals 
    
    summary_list = []
    for stype, group in item_level_surprisals.groupby('sentence_type'):
        summary_row = {'sentence_type': stype}
        for cond_col in lan_conditions_expected: # Use expected long names for summary
            if cond_col in group.columns and pd.api.types.is_numeric_dtype(group[cond_col]):
                summary_row[f'mean_surprisal_{cond_col}'] = group[cond_col].mean()
                summary_row[f'se_surprisal_{cond_col}'] = group[cond_col].sem()
                summary_row[f'n_items_{cond_col}'] = group[cond_col].count() 
            else: 
                summary_row[f'mean_surprisal_{cond_col}'] = np.nan
                summary_row[f'se_surprisal_{cond_col}'] = np.nan
                summary_row[f'n_items_{cond_col}'] = 0
        summary_list.append(summary_row)
        
    summary_df = pd.DataFrame(summary_list)
    if not summary_df.empty:
        summary_df.to_csv(summary_output_path, index=False, float_format='%.8f')
        print(f"Condition summary saved to {os.path.abspath(summary_output_path)}\nSummary:\n{summary_df}")
    else: 
        print("No condition summary to save.")
        
    return item_level_surprisals

def calculate_lan_style_metrics_and_accuracy(item_level_df, metrics_output_path, accuracy_summary_output_path):
    # Calculates Delta_plus_filler, Delta_minus_filler, DiD, and success flags
    if item_level_df is None or item_level_df.empty: 
        print("Error: Input item_level_df for Lan style metrics is empty. Skipping."); return None

    # Define column names based on Lan's paradigm
    s_pfpg = "PLUS_FILLER_PLUS_GAP"    # Grammatical PG (+F, +G1, +G2) -> Gapped G2 continuation
    s_mfpg = "MINUS_FILLER_PLUS_GAP"   # Ungrammatical (-F, G1 filled, +G2) -> Gapped G2 continuation
    s_pfmg = "PLUS_FILLER_MINUS_GAP"  # Ungrammatical (+F, +G1, -G2 filled) -> Ungapped G2 continuation
    s_mfmg = "MINUS_FILLER_MINUS_GAP" # Grammatical (-F, G1 filled, -G2 filled) -> Ungapped G2 continuation

    required_s_cols = [s_pfpg, s_mfpg, s_pfmg, s_mfmg]
    missing_s_cols = [col for col in required_s_cols if col not in item_level_df.columns]
    if missing_s_cols: 
        print(f"Error: Item-level data missing required surprisal columns for Lan style metrics: {missing_s_cols}. Skipping."); return None
        
    metrics_df = item_level_df.copy()
    # Delta = Surprisal(ungapped G2 continuation) - Surprisal(gapped G2 continuation)
    metrics_df['delta_plus_filler']  = metrics_df[s_pfmg] - metrics_df[s_pfpg]
    metrics_df['delta_minus_filler'] = metrics_df[s_mfmg] - metrics_df[s_mfpg]
    metrics_df['did_effect']         = metrics_df['delta_plus_filler'] - metrics_df['delta_minus_filler']
    
    metrics_df['success_delta_plus_filler'] = metrics_df['delta_plus_filler'] > 0 
    metrics_df['success_did']               = metrics_df['did_effect'] > 0 

    output_cols = ['sentence_type', 'item_id', s_pfpg, s_mfpg, s_pfmg, s_mfmg, 
                   'delta_plus_filler', 'delta_minus_filler', 'did_effect',
                   'success_delta_plus_filler', 'success_did']
    final_metrics_df = metrics_df[[col for col in output_cols if col in metrics_df.columns]].copy() # Ensure only existing columns
    
    if not final_metrics_df.empty:
        final_metrics_df.to_csv(metrics_output_path, index=False, float_format='%.8f')
        print(f"Lan style metrics saved to {os.path.abspath(metrics_output_path)}")
            
        print("\n--- Lan et al. (2024) Style Accuracy Scores (Console Output) ---")
        accuracy_summary_data = [] 
        for stype, type_group in final_metrics_df.groupby('sentence_type'):
            print(f"Sentence Type: {stype}")
            for success_col, delta_col_name, desc_str in [
                ('success_delta_plus_filler', 'delta_plus_filler', "Delta_Plus_Filler > 0 (Preference for gapped G2 with +Filler)"),
                ('success_did', 'did_effect', "Difference-in-Differences (Delta_Plus_Filler > Delta_Minus_Filler)")]:
                if success_col in type_group.columns:
                    valid_items = type_group.dropna(subset=[delta_col_name]) 
                    if not valid_items.empty: 
                        accuracy = valid_items[success_col].mean() * 100
                        n_valid = len(valid_items)
                        print(f"  Accuracy for ({desc_str}): {accuracy:.2f}% (N={n_valid})")
                        accuracy_summary_data.append({
                            'sentence_type': stype,
                            'metric_description': desc_str,
                            'accuracy_percent': accuracy,
                            'n_valid_items': n_valid
                        })
                    else: 
                        print(f"  No valid items for Accuracy ({desc_str}).")
                        accuracy_summary_data.append({'sentence_type': stype, 'metric_description': desc_str, 'accuracy_percent': np.nan, 'n_valid_items': 0})
            print("-" * 30)
        print("-------------------------------------------------")

        if accuracy_summary_data and accuracy_summary_output_path:
            acc_summary_df = pd.DataFrame(accuracy_summary_data)
            acc_summary_df.to_csv(accuracy_summary_output_path, index=False, float_format='%.2f')
            print(f"Lan style accuracy summary saved to {os.path.abspath(accuracy_summary_output_path)}")
        return final_metrics_df # Return the df with calculated metrics
    else: 
        print("No Lan style metrics calculated.")
        return None

def perform_statistical_analysis(metrics_df, stat_output_path):
    """Performs t-tests on Lan-style delta metrics and appends to a text file."""
    if metrics_df is None or metrics_df.empty:
        print("No metrics data to perform statistical analysis on. Skipping.")
        return

    all_results_text = [f"Statistical Summary for: Input derived from {DATASET_TYPE} data\n",
                        f"Model: {MODEL_NAME}\n",
                        f"Analysis Suffix: {ANALYSIS_SUFFIX}{FILTER_SUFFIX}\n",
                        "="*60 + "\n"]

    cols_to_test = {
        'delta_plus_filler': {'alternative': 'greater', 'desc': "Lan: Delta_Plus_Filler (Ungapped G2 - Gapped G2, with +Filler)"},
        'delta_minus_filler':{'alternative': 'two-sided', 'desc': "Lan: Delta_Minus_Filler (Ungapped G2 - Gapped G2, with -Filler)"}, # Often expected negative or near zero
        'did_effect':        {'alternative': 'greater', 'desc': "Lan: Difference-in-Differences"}
    }
                        
    for stype, type_group in metrics_df.groupby('sentence_type'):
        all_results_text.append(f"\n--- Statistics for Sentence Type: {stype} ---\n")
        print(f"\n--- Performing t-tests for Sentence Type: {stype} ---")
        for col, params in cols_to_test.items():
            if col in type_group.columns:
                data_series = type_group[col].dropna()
                if len(data_series) >= 2: # Need at least 2 data points for t-test
                    mean_val = data_series.mean()
                    std_val = data_series.std()
                    n_val = len(data_series)
                    t_statistic, p_value = stats.ttest_1samp(data_series, 0, alternative=params['alternative']) # Test against 0
                    
                    try:
                        conf_int = stats.t.interval(0.95, n_val-1, loc=mean_val, scale=stats.sem(data_series) if n_val > 1 else 0)
                        conf_int_str = f"[{conf_int[0]:.4f}, {conf_int[1]:.4f}]" if n_val > 1 else "N/A (N<2 for SEM)"
                    except Exception: conf_int_str = "N/A"

                    p_val_print = f"{p_value:.4f}" if p_value >= 0.0001 else "< .0001"
                    
                    result_str = (
                        f"  One-Sample T-test for '{params['desc']}' (H0: mean = 0, HA: mean {'>' if params['alternative']=='greater' else '!='} 0)\n"
                        f"    N (after NaN removal): {n_val}\n"
                        f"    Mean: {mean_val:.4f}\n"
                        f"    Std Dev: {std_val:.4f}\n"
                        f"    95% CI for Mean: {conf_int_str}\n"
                        f"    T-statistic: {t_statistic:.4f}\n"
                        f"    P-value: {p_val_print} "
                    )
                    sig_level = ""
                    if p_value < 0.001: sig_level = "(sig. at p < .001)"
                    elif p_value < 0.01: sig_level = "(sig. at p < .01)"
                    elif p_value < 0.05: sig_level = "(sig. at p < .05)"
                    else: sig_level = "(not sig. at p < .05)"
                    result_str += sig_level + "\n"
                    print(result_str)
                    all_results_text.append(result_str)
                else:
                    msg = f"  T-test for '{params['desc']}': Not enough data (N={len(data_series)} after NaN removal).\n"
                    print(msg); all_results_text.append(msg)
            else:
                msg = f"  Warning: Column '{col}' for t-test not found in type '{stype}'. Skipping.\n"
                print(msg); all_results_text.append(msg)
    
    if stat_output_path:
        try:
            output_dir_for_stat = os.path.dirname(stat_output_path)
            if output_dir_for_stat and not os.path.exists(output_dir_for_stat):
                os.makedirs(output_dir_for_stat); print(f"Created dir for stat summary: {output_dir_for_stat}")
            with open(stat_output_path, 'w') as f:
                for text_block in all_results_text: f.write(text_block + "\n")
            print(f"\nFull statistical summary saved to: {os.path.abspath(stat_output_path)}")
        except Exception as e: print(f"Error saving statistical summary: {e}")

# Plotting functions (plot_average_effects, plot_item_effects)
# These are mostly unchanged but will use the new config variables for labels/names
# Assume plot_average_effects and plot_item_effects from your provided script are here.
# For brevity, they are not repeated but should be included in your actual script.
def plot_average_effects(effects_df, effect1_col_name, effect2_col_name, 
                         effect1_plot_label, effect2_plot_label, 
                         plot_title_main_label, output_path_png, dataset_name_suffix=""):
    if effects_df.empty or not all(col in effects_df.columns for col in [effect1_col_name, effect2_col_name]):
        print(f"Warning: Missing one or both effect columns ('{effect1_col_name}', '{effect2_col_name}') "
              f"or empty dataframe for average plot '{plot_title_main_label}'. Skipping.")
        return

    # Drop rows where *both* effects are NaN for averaging, but keep if at least one is present
    effects_df_cleaned = effects_df.dropna(subset=[effect1_col_name, effect2_col_name], how='all')
    if effects_df_cleaned.empty :
        print(f"No valid data (all NaNs for effects) for avg {plot_title_main_label.lower()}s plot. Skipping."); return

    avg_effects = effects_df_cleaned.groupby('sentence_type')[[effect1_col_name, effect2_col_name]].mean(numeric_only=True).reset_index()
    
    if avg_effects.empty:
        print(f"No data after grouping by sentence_type for avg {plot_title_main_label.lower()}s plot. Skipping."); return

    plot_df = avg_effects.melt(id_vars=['sentence_type'], 
                               value_vars=[effect1_col_name, effect2_col_name], 
                               var_name='effect_type_melted', 
                               value_name='average_effect_value')
    
    plot_df['effect_type_plot_label'] = plot_df['effect_type_melted'].map({
        effect1_col_name: effect1_plot_label, 
        effect2_col_name: effect2_plot_label
    })
    
    # Handle cases where one effect might be all NaN after grouping and doesn't appear in melt
    unique_plot_labels_needed = {effect1_plot_label, effect2_plot_label}
    if not unique_plot_labels_needed.issubset(set(plot_df['effect_type_plot_label'].unique())):
        print(f"Warning: Some effect types might be all NaN and not appear in the plot for '{plot_title_main_label}'.")


    plt.figure(figsize=(max(10, len(avg_effects['sentence_type'].unique()) * 3), 7)) 
    sns.barplot(x='sentence_type', y='average_effect_value', hue='effect_type_plot_label', data=plot_df, palette="viridis")
    
    title = f'Average {plot_title_main_label} by Sentence Type ({MODEL_NAME}){dataset_name_suffix}'
    plt.title(title, fontsize=16)
    plt.xlabel('Sentence Type', fontsize=14)
    plt.ylabel(f'Average {plot_title_main_label} (Surprisal Bits)', fontsize=14)
    plt.axhline(0, color='grey', lw=1, linestyle='--')
    plt.legend(title='Effect Type', fontsize=12, title_fontsize=13, loc='best')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    try: 
        plt.savefig(output_path_png)
        print(f"Average {plot_title_main_label.lower()}s plot saved to {os.path.abspath(output_path_png)}")
    except Exception as e: 
        print(f"Error saving average {plot_title_main_label.lower()}s plot: {e}")
    plt.close()

def plot_item_effects(effects_df, effect1_col_name, effect2_col_name, 
                      item_effect1_plot_label, item_effect2_plot_label, 
                      plot_title_main_label, output_plot_dir, dataset_name_suffix=""):

    if effects_df.empty or not all(col in effects_df.columns for col in [effect1_col_name, effect2_col_name]):
        print(f"Warning: Missing one or both effect columns ('{effect1_col_name}', '{effect2_col_name}') "
              f"or empty dataframe for item plot '{plot_title_main_label}'. Skipping.")
        return
        
    if not os.path.exists(output_plot_dir): 
        os.makedirs(output_plot_dir)
        print(f"Created per-item plot directory: {output_plot_dir}")
        
    for sentence_type, group_df in effects_df.groupby('sentence_type'):
        # Drop rows where *both* effects are NaN for this specific sentence_type for plotting
        group_df_cleaned = group_df.dropna(subset=[effect1_col_name, effect2_col_name], how='all')

        if group_df_cleaned.empty: 
            print(f"No valid item data (all NaNs for effects) to plot for sentence type '{sentence_type}'. Skipping this plot.")
            continue
            
        num_items = len(group_df_cleaned['item_id'].unique())

        if num_items > MAX_ITEMS_FOR_PER_ITEM_PLOT: 
            print(f"Skipping per-item plot for sentence type '{sentence_type}' because number of items ({num_items}) "
                  f"exceeds threshold ({MAX_ITEMS_FOR_PER_ITEM_PLOT}).")
            continue 
            
        plot_df = group_df_cleaned.melt(id_vars=['item_id'], 
                                        value_vars=[effect1_col_name, effect2_col_name], 
                                        var_name='effect_type_melted', 
                                        value_name='effect_value')
        plot_df['effect_type_plot_label'] = plot_df['effect_type_melted'].map({
            effect1_col_name: item_effect1_plot_label, 
            effect2_col_name: item_effect2_plot_label
        })
        
        fig_width = max(10, num_items * 0.6) # Adjusted multiplier
        
        plt.figure(figsize=(fig_width, 7)) # Adjusted height
        sns.barplot(x='item_id', y='effect_value', hue='effect_type_plot_label', data=plot_df, palette="viridis")
        
        title = f'{plot_title_main_label} by Item for {sentence_type} ({MODEL_NAME}){dataset_name_suffix}'
        plt.title(title, fontsize=16)
        plt.xlabel('Item ID', fontsize=14)
        plt.ylabel(f'{plot_title_main_label} (Surprisal Bits)', fontsize=14)
        plt.axhline(0, color='grey', lw=1, linestyle='--')
        plt.legend(title='Effect Type', fontsize=12, title_fontsize=13, loc='best')
        plt.xticks(rotation=60 if num_items > 10 else 0, ha='right' if num_items > 10 else 'center', fontsize=min(10, 300/num_items if num_items > 0 else 10) ) # Dynamic font size for x-ticks
        plt.tight_layout() 
        
        plot_filename = os.path.join(output_plot_dir, f"{sentence_type}_{MODEL_NAME}_item_effects{dataset_name_suffix}.png")
        try: 
            plt.savefig(plot_filename)
            print(f"Per-item {plot_title_main_label.lower()}s plot for {sentence_type} saved to {os.path.abspath(plot_filename)}")
        except Exception as e: 
            print(f"Error saving per-item plot for {sentence_type}: {e}")
        plt.close()


def main():
    print(f"--- Running Analysis for DATASET_TYPE: {DATASET_TYPE} ---")
    if not os.path.exists(ANALYSIS_OUTPUT_DIR): 
        os.makedirs(ANALYSIS_OUTPUT_DIR); print(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")
        
    if not os.path.exists(AGGREGATED_INPUT_CSV): 
        print(f"Critical Error: Input file not found for {DATASET_TYPE} at '{os.path.abspath(AGGREGATED_INPUT_CSV)}'"); return
        
    print(f"Loading aggregated surprisals from {AGGREGATED_INPUT_CSV}...")
    try: 
        aggregated_df_full = pd.read_csv(AGGREGATED_INPUT_CSV)
    except Exception as e: 
        print(f"Error loading {AGGREGATED_INPUT_CSV}: {e}"); return
        
    if aggregated_df_full.empty: 
        print("Aggregated surprisals file is empty."); return
    
    # --- Apply Column Renaming and Filtering ---
    # This renaming is more general now, applying if the target names aren't present
    rename_map = {}
    if 'source_doc_name' in aggregated_df_full.columns and 'sentence_type' not in aggregated_df_full.columns:
        rename_map['source_doc_name'] = 'sentence_type'
    if 'item' in aggregated_df_full.columns and 'item_id' not in aggregated_df_full.columns:
        rename_map['item'] = 'item_id'
    if rename_map:
        print(f"Renaming columns: {rename_map}")
        aggregated_df_full.rename(columns=rename_map, inplace=True)
        # print(f"Columns after renaming: {aggregated_df_full.columns.tolist()}")

    required_cols = ['sentence_type', 'item_id', 'condition', 'aggregated_surprisal_bits'] 
    needs_orig_sentence_col = (DATASET_TYPE == "LAN" and (FILTER_LAN_GENITIVE_GERUNDS or FILTER_LAN_INTENT_TO))
    
    if needs_orig_sentence_col :
        required_cols.append('original_full_sentence')
        
    missing = [col for col in required_cols if col not in aggregated_df_full.columns]
    if missing: 
        print(f"Error: Input CSV '{AGGREGATED_INPUT_CSV}' missing required columns: {', '.join(missing)}. Found: {aggregated_df_full.columns.tolist()}"); return

    current_df_for_analysis = aggregated_df_full.copy()

    if DATASET_TYPE == "LAN":
        if FILTER_LAN_GENITIVE_GERUNDS:
            print("Filtering LAN data to exclude sentences with possessive gerund constructions (e.g., N's V-ing)...")
            regex_pattern = r"\b\w+'s\s+\w+ing\b" 
            current_df_for_analysis['original_full_sentence'] = current_df_for_analysis['original_full_sentence'].astype(str)
            items_to_exclude_gg = current_df_for_analysis[
                current_df_for_analysis['original_full_sentence'].str.contains(regex_pattern, regex=True, case=False, na=False)
            ]['item_id'].unique()
            if len(items_to_exclude_gg) > 0:
                print(f"Found {len(items_to_exclude_gg)} item_ids with genitive gerunds. Excluding them.")
                original_rows = len(current_df_for_analysis)
                current_df_for_analysis = current_df_for_analysis[~current_df_for_analysis['item_id'].isin(items_to_exclude_gg)]
                print(f"Removed {original_rows - len(current_df_for_analysis)} rows. Remaining: {len(current_df_for_analysis)}")
            else: print("No items found with genitive gerund pattern.")

        if FILTER_LAN_INTENT_TO:
            if current_df_for_analysis.empty: print("Dataframe empty before 'intent to' filter. Skipping.")
            else:
                print("Filtering LAN data to exclude sentences with the phrase 'intent to'...")
                phrase_to_exclude = "intent to"
                current_df_for_analysis['original_full_sentence'] = current_df_for_analysis['original_full_sentence'].astype(str)
                items_to_exclude_intent = current_df_for_analysis[
                    current_df_for_analysis['original_full_sentence'].str.contains(phrase_to_exclude, case=False, na=False, regex=False)
                ]['item_id'].unique()
                if len(items_to_exclude_intent) > 0:
                    print(f"Found {len(items_to_exclude_intent)} item_ids with '{phrase_to_exclude}'. Excluding them.")
                    original_rows = len(current_df_for_analysis)
                    current_df_for_analysis = current_df_for_analysis[~current_df_for_analysis['item_id'].isin(items_to_exclude_intent)]
                    print(f"Removed {original_rows - len(current_df_for_analysis)} rows. Remaining: {len(current_df_for_analysis)}")
                else: print(f"No items found with '{phrase_to_exclude}'.")
    
    if current_df_for_analysis.empty: print(f"Dataframe empty after all filtering for {DATASET_TYPE}. Exiting."); return

    current_df_for_analysis['sentence_type'] = current_df_for_analysis['sentence_type'].astype(str) 
    df_processed = current_df_for_analysis[current_df_for_analysis['sentence_type'].isin(TARGET_SENTENCE_TYPES)]
    if df_processed.empty: 
        print(f"No data for target sentence types: {TARGET_SENTENCE_TYPES} after all filtering. Exiting."); return

    # --- Analysis Flow ---
    if DATASET_TYPE == "WILCOX":
        print(f"\nCalculating '{EFFECT_TYPE_LABEL}' for {DATASET_TYPE} data (on {len(df_processed)} rows)...")
        effects_df = calculate_wilcox_effects(df_processed, TARGET_SENTENCE_TYPES)
        if not effects_df.empty:
            effects_df.to_csv(CALCULATED_EFFECTS_CSV, index=False, float_format='%.8f')
            print(f"Calculated '{EFFECT_TYPE_LABEL}' saved to {os.path.abspath(CALCULATED_EFFECTS_CSV)}")
            
            plot_average_effects(effects_df, EFFECT1_NAME, EFFECT2_NAME, 
                                   EFFECT1_PLOT_LABEL, EFFECT2_PLOT_LABEL, 
                                   EFFECT_TYPE_LABEL, AVERAGE_EFFECTS_PLOT_PNG, dataset_name_suffix=FILTER_SUFFIX)
            plot_item_effects(effects_df, EFFECT1_NAME, EFFECT2_NAME, 
                                ITEM_EFFECT1_PLOT_LABEL, ITEM_EFFECT2_PLOT_LABEL, 
                                EFFECT_TYPE_LABEL, PER_ITEM_PLOT_DIR, title_suffix=FILTER_SUFFIX)
        else: print(f"No '{EFFECT_TYPE_LABEL}' calculated for {DATASET_TYPE}.")

    elif LAN_STYLE_ANALYSIS_APPLIES: # For LAN and NOVEL_SUBJECT_PG
        print(f"\nSummarizing raw condition surprisals for {DATASET_TYPE} data (on {len(df_processed)} rows)...")
        item_level_surprisals = summarize_lan_style_condition_surprisals(
            df_processed, 
            TARGET_SENTENCE_TYPES,  
            LAN_ITEM_CONDITION_SURPRISALS_CSV, 
            LAN_CONDITION_SUMMARY_CSV      
        )
        
        metrics_df = None
        if item_level_surprisals is not None and not item_level_surprisals.empty:
            print(f"\nCalculating Lan et al. (2024) style metrics for {DATASET_TYPE} data...")
            metrics_df = calculate_lan_style_metrics_and_accuracy(
                item_level_surprisals, 
                LAN_PAPER_METRICS_CSV,
                LAN_PAPER_ACCURACY_SUMMARY_CSV 
            )
        else:
            print(f"Skipping Lan style metrics & stats due to missing item-level surprisal data for {DATASET_TYPE}.")

        if metrics_df is not None and not metrics_df.empty:
            print(f"\nPerforming statistical analysis for {DATASET_TYPE} data...")
            perform_statistical_analysis(metrics_df, STAT_RESULTS_OUTPUT_TXT)
            
            print(f"\nGenerating plots for average {EFFECT_TYPE_LABEL.lower()}s for {DATASET_TYPE}...")
            plot_average_effects(metrics_df, EFFECT1_NAME, EFFECT2_NAME, 
                                   EFFECT1_PLOT_LABEL, EFFECT2_PLOT_LABEL, 
                                   EFFECT_TYPE_LABEL, AVERAGE_EFFECTS_PLOT_PNG, dataset_name_suffix=FILTER_SUFFIX)
            
            print(f"\nGenerating plots for per-item {EFFECT_TYPE_LABEL.lower()}s for {DATASET_TYPE}...")
            plot_item_effects(metrics_df, EFFECT1_NAME, EFFECT2_NAME, 
                                ITEM_EFFECT1_PLOT_LABEL, ITEM_EFFECT2_PLOT_LABEL, 
                                EFFECT_TYPE_LABEL, PER_ITEM_PLOT_DIR, title_suffix=FILTER_SUFFIX)
        else:
            print(f"No Lan-style metrics calculated for {DATASET_TYPE}, skipping related plots and stats.")
            
    print("\nScript finished.")

if __name__ == "__main__":
    main()