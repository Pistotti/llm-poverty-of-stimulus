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
# Set this to the type of data you want to analyze.
DATASET_TYPE = "LAN" 

MAX_ITEMS_FOR_PER_ITEM_PLOT = 200 

# Configuration flags for filtering original LAN data
# These will only take effect if DATASET_TYPE is "LAN"
FILTER_LAN_GENITIVE_GERUNDS = False
FILTER_LAN_INTENT_TO = False        

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
    AGGREGATED_INPUT_FILENAME = f"lan_extracted_critical_surprisals_{MODEL_NAME}.csv"
    AGGREGATED_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results")
    # Targets
    TARGET_SENTENCE_TYPES = ["lan_parasitic_gap"] 
    # Effect Naming (for plotting)
    EFFECT_TYPE_LABEL = "Lan Metric" 
    EFFECT1_NAME = 'delta_plus_filler'  
    EFFECT2_NAME = 'did_effect' 
    EFFECT1_PLOT_LABEL = 'Avg Delta_Plus_Filler' 
    EFFECT2_PLOT_LABEL = 'Avg DiD Effect' 
    ITEM_EFFECT1_PLOT_LABEL = 'Delta_Plus_Filler'
    ITEM_EFFECT2_PLOT_LABEL = 'DiD Effect'
    # Output Suffixes and Dirs
    ANALYSIS_SUFFIX = "lan_pg_types"
    OUTPUT_SUBDIR = DATASET_TYPE.lower()
    
elif DATASET_TYPE == "NOVEL_SUBJECT_PG":
    # Inputs
    AGGREGATED_INPUT_FILENAME = f"novel_data_extracted_critical_surprisals_{MODEL_NAME}.csv"
    AGGREGATED_INPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "aggregated")
    # Targets
    TARGET_SENTENCE_TYPES = ["subject_gap"] # As in your novel_data.csv
    # Effect Naming (for plotting)
    EFFECT_TYPE_LABEL = "Lan Metric (Novel Data)" 
    EFFECT1_NAME = 'delta_plus_filler'  
    EFFECT2_NAME = 'did_effect' 
    EFFECT1_PLOT_LABEL = 'Avg Delta_Plus_Filler' 
    EFFECT2_PLOT_LABEL = 'Avg DiD Effect' 
    ITEM_EFFECT1_PLOT_LABEL = 'Delta_Plus_Filler'
    ITEM_EFFECT2_PLOT_LABEL = 'DiD Effect'
    # Output Suffixes and Dirs
    ANALYSIS_SUFFIX = "novel_subject_pg"
    OUTPUT_SUBDIR = "novel_data_findings"

else:
    raise ValueError(f"Unsupported DATASET_TYPE: {DATASET_TYPE}")

AGGREGATED_INPUT_CSV = os.path.join(AGGREGATED_INPUT_DIR, AGGREGATED_INPUT_FILENAME)

# --- Determine Filter Suffix for Output Filenames (Only for LAN type) ---
FILTER_SUFFIX = ""
if DATASET_TYPE == "LAN":
    active_filter_descriptions = []
    if FILTER_LAN_GENITIVE_GERUNDS:
        active_filter_descriptions.append("no_gen_gerunds")
    if FILTER_LAN_INTENT_TO:
        active_filter_descriptions.append("no_intent_to")
    if active_filter_descriptions:
        FILTER_SUFFIX = "_filtered_" + "_".join(active_filter_descriptions)

# --- Define Output Paths ---
ANALYSIS_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", OUTPUT_SUBDIR)
PER_ITEM_PLOT_DIR = os.path.join(ANALYSIS_OUTPUT_DIR, f"per_item_plots_{ANALYSIS_SUFFIX}{FILTER_SUFFIX}")
AVERAGE_EFFECTS_PLOT_PNG = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_average_{ANALYSIS_SUFFIX}_effects_plot{FILTER_SUFFIX}.png")
CALCULATED_EFFECTS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_effects{FILTER_SUFFIX}.csv") # For Wilcox-style effects

# Specific to LAN-style analysis
LAN_STYLE_ANALYSIS_APPLIES = (DATASET_TYPE == "LAN" or DATASET_TYPE == "NOVEL_SUBJECT_PG")
if LAN_STYLE_ANALYSIS_APPLIES:
    LAN_ITEM_CONDITION_SURPRISALS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_item_condition_surprisals{FILTER_SUFFIX}.csv")
    LAN_CONDITION_SUMMARY_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_condition_surprisal_summary{FILTER_SUFFIX}.csv")
    LAN_PAPER_METRICS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_lan_paper_metrics{FILTER_SUFFIX}.csv")
    LAN_PAPER_ACCURACY_SUMMARY_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_lan_paper_accuracy_summary{FILTER_SUFFIX}.csv")
    STAT_RESULTS_OUTPUT_TXT = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_stat_summary{FILTER_SUFFIX}.txt")

# --- End Configuration ---


# --- HELPER & ANALYSIS FUNCTIONS ---

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
                effect1 = s_wg_val - s_tg_val
                effect2 = s_wn_val - s_tn_val
        
        effects_data.append({'sentence_type': sentence_type, 'item_id': item_id, 
                             EFFECT1_NAME: effect1, EFFECT2_NAME: effect2})
    return pd.DataFrame(effects_data)

def summarize_lan_style_condition_surprisals(df, target_sentence_types, item_output_path, summary_output_path):
    df_filtered = df[df['sentence_type'].isin(target_sentence_types)].copy()
    if df_filtered.empty: 
        print(f"Warning: No data for {target_sentence_types} in summarize_lan_style_condition_surprisals."); return None
    
    try:
        item_level_df = df_filtered.pivot_table(
            index=['sentence_type', 'item_id'], columns='condition', values='aggregated_surprisal_bits'
        ).reset_index()
    except Exception as e: 
        print(f"Error pivoting data: {e}"); return None
        
    rename_map = {
        "PFPG": "PLUS_FILLER_PLUS_GAP", "MFPG": "MINUS_FILLER_PLUS_GAP",
        "PFMG": "PLUS_FILLER_MINUS_GAP", "MFMG": "MINUS_FILLER_MINUS_GAP"
    }
    item_level_df.rename(columns=rename_map, inplace=True)
        
    lan_conditions_expected = ["PLUS_FILLER_PLUS_GAP", "MINUS_FILLER_PLUS_GAP", "PLUS_FILLER_MINUS_GAP", "MINUS_FILLER_MINUS_GAP"]
    for cond in lan_conditions_expected: 
        if cond not in item_level_df.columns: 
            item_level_df[cond] = np.nan
            
    item_level_df = item_level_df[['sentence_type', 'item_id'] + lan_conditions_expected]
    item_level_df.to_csv(item_output_path, index=False, float_format='%.8f')
    print(f"Item-level condition surprisals saved to {os.path.abspath(item_output_path)}")
    
    summary = item_level_df.groupby('sentence_type')[lan_conditions_expected].agg(['mean', 'sem', 'count']).reset_index()
    summary.to_csv(summary_output_path, index=False, float_format='%.8f')
    print(f"Condition summary saved to {os.path.abspath(summary_output_path)}\nSummary:\n{summary}")
    
    return item_level_df

def calculate_lan_style_metrics(item_level_df, metrics_output_path, accuracy_summary_output_path):
    if item_level_df is None or item_level_df.empty:
        print("Skipping Lan style metrics: no item-level data provided.")
        return None

    s_pfpg = "PLUS_FILLER_PLUS_GAP"
    s_mfpg = "MINUS_FILLER_PLUS_GAP"
    s_pfmg = "PLUS_FILLER_MINUS_GAP"
    s_mfmg = "MINUS_FILLER_MINUS_GAP"

    required_cols = [s_pfpg, s_mfpg, s_pfmg, s_mfmg]
    if not all(col in item_level_df.columns for col in required_cols):
        print(f"Error: Pivoted data missing required columns for Lan metrics: {required_cols}"); return None

    metrics_df = item_level_df.copy()
    metrics_df['delta_plus_filler'] = metrics_df[s_pfmg] - metrics_df[s_pfpg]
    metrics_df['delta_minus_filler'] = metrics_df[s_mfmg] - metrics_df[s_mfpg]
    metrics_df['did_effect'] = metrics_df['delta_plus_filler'] - metrics_df['delta_minus_filler']
    metrics_df['success_delta_plus_filler'] = metrics_df['delta_plus_filler'] > 0
    metrics_df['success_did'] = metrics_df['did_effect'] > 0

    metrics_df.to_csv(metrics_output_path, index=False, float_format='%.8f')
    print(f"Lan style metrics saved to {os.path.abspath(metrics_output_path)}")

    accuracy_summary_data = []
    print("\n--- Lan et al. (2024) Style Accuracy Scores ---")
    for stype, type_group in metrics_df.groupby('sentence_type'):
        print(f"Sentence Type: {stype}")
        for success_col, delta_col, desc in [
            ('success_delta_plus_filler', 'delta_plus_filler', "Delta_Plus_Filler > 0"),
            ('success_did', 'did_effect', "Difference-in-Differences")]:
            valid_items = type_group.dropna(subset=[delta_col])
            accuracy = valid_items[success_col].mean() * 100 if not valid_items.empty else 0
            n_valid = len(valid_items)
            print(f"  Accuracy for ({desc}): {accuracy:.2f}% (N={n_valid})")
            accuracy_summary_data.append({'sentence_type': stype, 'metric': desc, 'accuracy_percent': accuracy, 'n_valid_items': n_valid})
    
    pd.DataFrame(accuracy_summary_data).to_csv(accuracy_summary_output_path, index=False, float_format='%.2f')
    print(f"Accuracy summary saved to {os.path.abspath(accuracy_summary_output_path)}")
    
    return metrics_df

def perform_statistical_analysis(metrics_df, stat_output_path):
    if metrics_df is None or metrics_df.empty:
        print("No metrics data to perform statistical analysis on. Skipping.")
        return

    all_results_text = [f"Statistical Summary for: {DATASET_TYPE} data ({os.path.basename(AGGREGATED_INPUT_FILENAME)})\n",
                        f"Model: {MODEL_NAME}\n",
                        "="*60 + "\n"]

    cols_to_test = {
        'delta_plus_filler': {'alternative': 'greater', 'desc': "Lan: Delta_Plus_Filler"},
        'delta_minus_filler':{'alternative': 'two-sided', 'desc': "Lan: Delta_Minus_Filler"},
        'did_effect':        {'alternative': 'greater', 'desc': "Lan: Difference-in-Differences"}
    }
                        
    for stype, type_group in metrics_df.groupby('sentence_type'):
        all_results_text.append(f"\n--- Statistics for Sentence Type: {stype} ---\n")
        print(f"\n--- Performing t-tests for Sentence Type: {stype} ---")
        for col, params in cols_to_test.items():
            if col in type_group.columns:
                data_series = type_group[col].dropna()
                if len(data_series) >= 2:
                    mean_val = data_series.mean()
                    std_val = data_series.std()
                    n_val = len(data_series)
                    t_statistic, p_value = stats.ttest_1samp(data_series, 0, alternative=params['alternative'])
                    
                    try:
                        conf_int = stats.t.interval(0.95, n_val-1, loc=mean_val, scale=stats.sem(data_series))
                        conf_int_str = f"[{conf_int[0]:.4f}, {conf_int[1]:.4f}]"
                    except Exception: conf_int_str = "N/A"

                    p_val_print = f"{p_value:.4f}" if p_value >= 0.0001 else "< .0001"
                    
                    result_str = (
                        f"  One-Sample T-test for '{params['desc']}' (H0: mean=0)\n"
                        f"    N: {n_val}, Mean: {mean_val:.4f}, SD: {std_val:.4f}\n"
                        f"    95% CI: {conf_int_str}\n"
                        f"    t({n_val-1}) = {t_statistic:.4f}, p = {p_val_print} ({params['alternative']})\n"
                    )
                    print(result_str)
                    all_results_text.append(result_str)
    
    if stat_output_path:
        try:
            with open(stat_output_path, 'w') as f:
                for text_block in all_results_text: f.write(text_block + "\n")
            print(f"\nFull statistical summary saved to: {os.path.abspath(stat_output_path)}")
        except Exception as e: print(f"Error saving statistical summary: {e}")

def plot_average_effects(effects_df, effect1_col_name, effect2_col_name, 
                         effect1_plot_label, effect2_plot_label, 
                         plot_title_main_label, output_path_png):
    # This plotting function remains as previously drafted
    pass

def plot_item_effects(effects_df, effect1_col_name, effect2_col_name, 
                      item_effect1_plot_label, item_effect2_plot_label, 
                      plot_title_main_label, output_plot_dir):
    # This plotting function remains as previously drafted
    pass

def main():
    print(f"--- Running Analysis for DATASET_TYPE: {DATASET_TYPE} ---")
    
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
        print(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")
        
    if not os.path.exists(AGGREGATED_INPUT_CSV):
        print(f"Critical Error: Input file not found for {DATASET_TYPE} at '{os.path.abspath(AGGREGATED_INPUT_CSV)}'")
        return
        
    print(f"Loading aggregated surprisals from {AGGREGATED_INPUT_CSV}...")
    try: 
        aggregated_df_full = pd.read_csv(AGGREGATED_INPUT_CSV)
    except Exception as e: 
        print(f"Error loading {AGGREGATED_INPUT_CSV}: {e}"); return
        
    if aggregated_df_full.empty: 
        print("Aggregated surprisals file is empty."); return
    
    rename_map = {}
    if 'source_doc_name' in aggregated_df_full.columns:
        rename_map['source_doc_name'] = 'sentence_type'
    if 'item' in aggregated_df_full.columns:
        rename_map['item'] = 'item_id'
    if rename_map:
        aggregated_df_full.rename(columns=rename_map, inplace=True)

    required_cols = ['sentence_type', 'item_id', 'condition', 'aggregated_surprisal_bits']
    # ... (rest of filtering and dataframe preparation logic remains the same) ...

    df_processed = aggregated_df_full[aggregated_df_full['sentence_type'].isin(TARGET_SENTENCE_TYPES)].copy()
    if df_processed.empty:
        print(f"No data found for target sentence types: {TARGET_SENTENCE_TYPES}. Exiting.")
        return

    # --- Analysis Flow ---
    if DATASET_TYPE == "WILCOX":
        effects_df = calculate_wilcox_effects(df_processed, TARGET_SENTENCE_TYPES)
        if not effects_df.empty:
            effects_df.to_csv(CALCULATED_EFFECTS_CSV, index=False, float_format='%.8f')
            # ... (Plotting for Wilcox) ...
        else:
            print(f"No wh-effects calculated for {DATASET_TYPE}.")

    elif LAN_STYLE_ANALYSIS_APPLIES:
        print(f"\nSummarizing raw condition surprisals for {DATASET_TYPE} data...")
        item_level_surprisals = summarize_lan_style_condition_surprisals(
            df_processed, 
            TARGET_SENTENCE_TYPES,  
            LAN_ITEM_CONDITION_SURPRISALS_CSV, 
            LAN_CONDITION_SUMMARY_CSV      
        )
        
        metrics_df = None
        if item_level_surprisals is not None and not item_level_surprisals.empty:
            print(f"\nCalculating Lan et al. (2024) style metrics for {DATASET_TYPE} data...")
            metrics_df = calculate_lan_style_metrics(
                item_level_surprisals, 
                LAN_PAPER_METRICS_CSV,
                LAN_PAPER_ACCURACY_SUMMARY_CSV
            )
        else:
            print(f"Skipping Lan style metrics & stats due to missing item-level surprisal data.")

        if metrics_df is not None and not metrics_df.empty:
            perform_statistical_analysis(metrics_df, STAT_RESULTS_OUTPUT_TXT)
            
            print(f"\nGenerating plots for average {EFFECT_TYPE_LABEL.lower()}s for {DATASET_TYPE}...")
            plot_average_effects(
                metrics_df, EFFECT1_NAME, EFFECT2_NAME,
                EFFECT1_PLOT_LABEL, EFFECT2_PLOT_LABEL,
                EFFECT_TYPE_LABEL, AVERAGE_EFFECTS_PLOT_PNG,
                dataset_name_suffix=FILTER_SUFFIX
            )
            
            # Per-item plotting call can be uncommented if needed
            # print(f"\nGenerating plots for per-item {EFFECT_TYPE_LABEL.lower()}s...")
            # plot_item_effects(...)
        else:
            print(f"No Lan-style metrics calculated, skipping stats and plots.")
            
    print("\nScript finished.")

if __name__ == "__main__":
    main()