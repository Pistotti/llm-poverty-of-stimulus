# llm-poverty-of-stimulus/llm_pos/wilcox/analyze_and_visualize_effects.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "gpt2" # Should match the model used

# *** NEW: Dataset type switch ***
# Change this to "WILCOX" to run for Wilcox data or "LAN" for LAN data
DATASET_TYPE = "LAN" # Or "WILCOX"

# --- Conditional Configuration based on DATASET_TYPE ---
if DATASET_TYPE == "WILCOX":
    AGGREGATED_INPUT_CSV = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "aggregated", f"{MODEL_NAME}_critical_regions_aggregated.csv")
    TARGET_SENTENCE_TYPES = ["basic_object", "basic_pp", "basic_subject"]
    EFFECT_TYPE_LABEL = "Wh-Effect" 
    EFFECT1_NAME = 'wh_effect_plus_gap' # Actual column name in output CSV
    EFFECT2_NAME = 'wh_effect_minus_gap'# Actual column name in output CSV
    EFFECT1_PLOT_LABEL = 'Avg Wh-Effect (+gap)'
    EFFECT2_PLOT_LABEL = 'Avg Wh-Effect (-gap)'
    ITEM_EFFECT1_PLOT_LABEL = 'Wh-Effect (+gap)'
    ITEM_EFFECT2_PLOT_LABEL = 'Wh-Effect (-gap)'
    ANALYSIS_SUFFIX = "wilcox_basic_types"
elif DATASET_TYPE == "LAN":
    AGGREGATED_INPUT_CSV = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", f"lan_extracted_critical_surprisals_{MODEL_NAME}.csv")
    TARGET_SENTENCE_TYPES = ["lan_parasitic_gap"] 
    # *** CHANGED Labels and Effect Names for LAN to use "Wh-Effect" terminology ***
    EFFECT_TYPE_LABEL = "Wh-Effect" # Changed from "Filler-Effect"
    EFFECT1_NAME = 'wh_effect_plus_gap'  # Changed from 'filler_effect_plus_gap'
    EFFECT2_NAME = 'wh_effect_minus_gap' # Changed from 'filler_effect_minus_gap'
    EFFECT1_PLOT_LABEL = 'Avg Wh-Effect (+gap)' # (Derived from +Filler vs -Filler in +Gap context)
    EFFECT2_PLOT_LABEL = 'Avg Wh-Effect (-gap)' # (Derived from +Filler vs -Filler in -Gap context)
    ITEM_EFFECT1_PLOT_LABEL = 'Wh-Effect (+gap)'
    ITEM_EFFECT2_PLOT_LABEL = 'Wh-Effect (-gap)'
    ANALYSIS_SUFFIX = "lan_types"
    
    LAN_ITEM_CONDITION_SURPRISALS_CSV = os.path.join(
        CURRENT_SCRIPT_DIR, "tims_results", "analysis", DATASET_TYPE.lower(),
        f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_item_condition_surprisals.csv"
    )
    LAN_CONDITION_SUMMARY_CSV = os.path.join(
        CURRENT_SCRIPT_DIR, "tims_results", "analysis", DATASET_TYPE.lower(), 
        f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_condition_surprisal_summary.csv"
    )
    LAN_PAPER_METRICS_CSV = os.path.join(
        CURRENT_SCRIPT_DIR, "tims_results", "analysis", DATASET_TYPE.lower(),
        f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_lan_paper_metrics.csv"
    )
else:
    raise ValueError(f"Unsupported DATASET_TYPE: {DATASET_TYPE}")

ANALYSIS_OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results", "analysis", DATASET_TYPE.lower())
CALCULATED_EFFECTS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_effects.csv") # This will now contain 'wh_effect...' for LAN too
AVERAGE_EFFECTS_PLOT_PNG = os.path.join(ANALYSIS_OUTPUT_DIR, f"{MODEL_NAME}_average_{ANALYSIS_SUFFIX}_effects_plot.png")
PER_ITEM_PLOT_DIR = os.path.join(ANALYSIS_OUTPUT_DIR, f"per_item_plots_{ANALYSIS_SUFFIX}")
# --- End Configuration ---

def calculate_effects(df, sentence_types_to_process, dataset_type_mode):
    effects_data = []
    df_filtered = df[df['sentence_type'].isin(sentence_types_to_process)].copy()
    if df_filtered.empty: return pd.DataFrame(effects_data)
    df_filtered['item_id'] = df_filtered['item_id'].astype(str)
    df_filtered['sentence_type'] = df_filtered['sentence_type'].astype(str)
    output_effect1_col, output_effect2_col = 'effect1_value', 'effect2_value'

    for (sentence_type, item_id), group in df_filtered.groupby(['sentence_type', 'item_id']):
        s_values = {}; effect1, effect2 = np.nan, np.nan
        conditions_map = {}
        if dataset_type_mode == "WILCOX":
            conditions_map = {"s_wg": "what_gap", "s_tg": "that_gap", "s_wn": "what_nogap", "s_tn": "that_nogap"}
        elif dataset_type_mode == "LAN": 
            # Conditions for LAN Wh-Effect (comparing +Filler vs -Filler)
            conditions_map = {"s_pfp": "PLUS_FILLER_PLUS_GAP", "s_mfp": "MINUS_FILLER_PLUS_GAP",
                              "s_pfm": "PLUS_FILLER_MINUS_GAP", "s_mfm": "MINUS_FILLER_MINUS_GAP"}
        all_present = True
        for key, cond_name in conditions_map.items():
            series = group.loc[group['condition'] == cond_name, 'aggregated_surprisal_bits']
            if series.empty or pd.isna(series.iloc[0]): all_present = False; break
            s_values[key] = series.iloc[0]
        if all_present:
            try:
                if dataset_type_mode == "WILCOX": 
                    effect1, effect2 = s_values["s_wg"] - s_values["s_tg"], s_values["s_wn"] - s_values["s_tn"]
                elif dataset_type_mode == "LAN": 
                    # Calculation for LAN: (+Filler) - (-Filler)
                    effect1 = s_values["s_pfp"] - s_values["s_mfp"] # For +Gap context
                    effect2 = s_values["s_pfm"] - s_values["s_mfm"] # For -Gap context
            except Exception as e: print(f"Error calc effects for {item_id} ({sentence_type}): {e}"); effect1, effect2 = np.nan, np.nan
        effects_data.append({'sentence_type': sentence_type, 'item_id': item_id, output_effect1_col: effect1, output_effect2_col: effect2})
    
    final_df = pd.DataFrame(effects_data)
    if not final_df.empty: 
        final_df = final_df.rename(columns={
            output_effect1_col: EFFECT1_NAME, 
            output_effect2_col: EFFECT2_NAME  
        })
    return final_df

def summarize_lan_condition_surprisals(df, target_sentence_types, item_output_path, summary_output_path):
    df_lan_filtered = df[df['sentence_type'].isin(target_sentence_types)].copy()
    if df_lan_filtered.empty: print(f"Warning: No LAN data for {target_sentence_types} in summarize_lan_condition_surprisals."); return None
    item_level_surprisals = None
    try:
        item_level_surprisals = df_lan_filtered.pivot_table(index=['sentence_type', 'item_id'], columns='condition', values='aggregated_surprisal_bits').reset_index()
    except Exception as e: print(f"Error pivoting data for LAN summary: {e}"); return None
    lan_conditions = ["PLUS_FILLER_PLUS_GAP", "MINUS_FILLER_PLUS_GAP", "PLUS_FILLER_MINUS_GAP", "MINUS_FILLER_MINUS_GAP"]
    for cond in lan_conditions:
        if cond not in item_level_surprisals.columns: item_level_surprisals[cond] = np.nan
    item_level_surprisals = item_level_surprisals[['sentence_type', 'item_id'] + lan_conditions] # Ensure order
    if not item_level_surprisals.empty:
        try: item_level_surprisals.to_csv(item_output_path, index=False, float_format='%.8f'); print(f"LAN item-level surprisals saved to {os.path.abspath(item_output_path)}")
        except Exception as e: print(f"Error saving LAN item-level surprisals: {e}")
    else: print("No item-level LAN surprisals to save."); return None # Return None if empty
    
    numeric_cols = [c for c in lan_conditions if c in item_level_surprisals.columns and pd.api.types.is_numeric_dtype(item_level_surprisals[c])]
    if not numeric_cols: print("Warning: No numeric LAN condition columns for summary."); return item_level_surprisals # Return the item_level data even if summary fails
    summary_list = []
    for stype, group in item_level_surprisals.groupby('sentence_type'):
        s_row = {'sentence_type': stype}
        for cond in numeric_cols: # Iterate over actual numeric columns present
            s_row[f'mean_surprisal_{cond}'] = group[cond].mean(); s_row[f'se_surprisal_{cond}'] = group[cond].sem(); s_row[f'n_items_{cond}'] = group[cond].count()
        summary_list.append(s_row)
    lan_summary_df = pd.DataFrame(summary_list)
    if not lan_summary_df.empty:
        try: lan_summary_df.to_csv(summary_output_path, index=False, float_format='%.8f'); print(f"LAN condition summary saved to {os.path.abspath(summary_output_path)}\nSummary:\n{lan_summary_df}")
        except Exception as e: print(f"Error saving LAN condition summary: {e}")
    else: print("No LAN summary to save.")
    return item_level_surprisals 

def calculate_lan_paper_metrics_and_accuracy(item_level_df, output_path):
    if item_level_df is None or item_level_df.empty: print("Error: Input item_level_df for Lan paper metrics is empty. Skipping."); return
    s_pfpg_col, s_mfpg_col, s_pfmg_col, s_mfmg_col = "PLUS_FILLER_PLUS_GAP", "MINUS_FILLER_PLUS_GAP", "PLUS_FILLER_MINUS_GAP", "MINUS_FILLER_MINUS_GAP"
    req_s_cols = [s_pfpg_col, s_mfpg_col, s_pfmg_col, s_mfmg_col]
    missing_s_cols = [col for col in req_s_cols if col not in item_level_df.columns]
    if missing_s_cols: print(f"Error: Item-level data missing required surprisal columns for Lan paper metrics: {missing_s_cols}. Skipping."); return
    metrics_df = item_level_df.copy()
    metrics_df['delta_plus_filler'] = metrics_df[s_pfmg_col] - metrics_df[s_pfpg_col]
    metrics_df['delta_minus_filler'] = metrics_df[s_mfmg_col] - metrics_df[s_mfpg_col]
    metrics_df['did_effect'] = metrics_df['delta_plus_filler'] - metrics_df['delta_minus_filler']
    metrics_df['success_delta_plus_filler'] = metrics_df['delta_plus_filler'] > 0
    metrics_df['success_did'] = metrics_df['did_effect'] > 0 
    out_cols = ['sentence_type', 'item_id', s_pfpg_col, s_mfpg_col, s_pfmg_col, s_mfmg_col, 
                  'delta_plus_filler', 'delta_minus_filler', 'did_effect', 'success_delta_plus_filler', 'success_did']
    for col in out_cols: # Ensure all desired columns exist
        if col not in metrics_df.columns: metrics_df[col] = np.nan
    metrics_df = metrics_df[out_cols] # Reorder and select
    if not metrics_df.empty:
        try: metrics_df.to_csv(output_path, index=False, float_format='%.8f'); print(f"Lan paper style metrics saved to {os.path.abspath(output_path)}")
        except Exception as e: print(f"Error saving Lan paper style metrics CSV: {e}")
        print("\n--- Lan et al. (2024) Style Accuracy Scores ---")
        for success_col, delta_col_name, desc in [
            ('success_delta_plus_filler', 'delta_plus_filler', "Delta_Plus_Filler > 0"),
            ('success_did', 'did_effect', "Difference-in-Differences (Delta_Plus_Filler > Delta_Minus_Filler)")]:
            if success_col in metrics_df.columns:
                valid_items = metrics_df.dropna(subset=[delta_col_name]) # Calculate accuracy only on non-NaN delta values
                if not valid_items.empty: print(f"Accuracy for ({desc}): {valid_items[success_col].mean() * 100:.2f}% (N={len(valid_items)})")
                else: print(f"No valid items for Accuracy ({desc}).")
        print("-------------------------------------------------")
    else: print("No Lan paper style metrics calculated.")

def plot_average_effects(effects_df, effect1_col, effect2_col, effect1_lab, effect2_lab, title_lab, out_path, suffix=""):
    if effects_df.empty or effects_df[[effect1_col, effect2_col]].isnull().all().all(): print(f"No valid data for avg {title_lab.lower()}s plot."); return
    avg_eff = effects_df.groupby('sentence_type')[[effect1_col, effect2_col]].mean(numeric_only=True).reset_index()
    plot_df = avg_eff.melt(id_vars=['sentence_type'], value_vars=[effect1_col, effect2_col], var_name='eff_type_melt', value_name='avg_eff_val')
    plot_df['eff_type_plot'] = plot_df['eff_type_melt'].map({effect1_col: effect1_lab, effect2_col: effect2_lab})
    plt.figure(figsize=(max(10, len(avg_eff['sentence_type'].unique()) * 2.5), 7))
    sns.barplot(x='sentence_type', y='avg_eff_val', hue='eff_type_plot', data=plot_df, palette="viridis")
    plt.title(f'Avg {title_lab}s by Sentence Type ({MODEL_NAME}){suffix}', fontsize=16); plt.xlabel('Sentence Type', fontsize=14); plt.ylabel(f'Avg {title_lab} (Surprisal Bits)', fontsize=14)
    plt.axhline(0, color='grey', lw=1, linestyle='--'); plt.legend(title='Effect Type', fontsize=12, title_fontsize=13); plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    try: plt.savefig(out_path); print(f"Avg {title_lab.lower()}s plot saved to {os.path.abspath(out_path)}")
    except Exception as e: print(f"Error saving avg {title_lab.lower()}s plot: {e}")
    plt.close()

def plot_item_effects(effects_df, effect1_col, effect2_col, item_eff1_lab, item_eff2_lab, title_lab, out_dir_base, suffix=""):
    if effects_df.empty or effects_df[[effect1_col, effect2_col]].isnull().all().all(): print(f"No valid item data for {title_lab.lower()}s plot."); return
    if not os.path.exists(out_dir_base): os.makedirs(out_dir_base); print(f"Created per-item plot dir: {out_dir_base}")
    for stype, group_df in effects_df.groupby('sentence_type'):
        if group_df.empty or group_df[[effect1_col, effect2_col]].isnull().all().all(): print(f"No data/NaNs for items in {stype}. Skip plot."); continue
        plot_df = group_df.melt(id_vars=['item_id'], value_vars=[effect1_col, effect2_col], var_name='eff_type_melt', value_name='eff_val')
        plot_df['eff_type_plot'] = plot_df['eff_type_melt'].map({effect1_col: item_eff1_lab, effect2_col: item_eff2_lab})
        num_items = len(group_df['item_id'].unique()); fig_width = max(10, num_items * 0.6)
        plt.figure(figsize=(fig_width, 7))
        sns.barplot(x='item_id', y='eff_val', hue='eff_type_plot', data=plot_df, palette="viridis")
        plt.title(f'{title_lab}s by Item for {stype} ({MODEL_NAME}){suffix}', fontsize=16); plt.xlabel('Item ID', fontsize=14); plt.ylabel(f'{title_lab} (Surprisal Bits)', fontsize=14)
        plt.axhline(0, color='grey', lw=1, linestyle='--'); plt.legend(title='Effect Type', fontsize=12, title_fontsize=13); plt.xticks(rotation=45, ha='right' if num_items > 15 else 'center'); plt.tight_layout()
        plot_fname = os.path.join(out_dir_base, f"{stype}_{MODEL_NAME}_item_effects.png")
        try: plt.savefig(plot_fname); print(f"Per-item {title_lab.lower()}s plot for {stype} saved to {os.path.abspath(plot_fname)}")
        except Exception as e: print(f"Error saving per-item plot for {stype}: {e}")
        plt.close()

def main():
    print(f"--- Running Analysis for DATASET_TYPE: {DATASET_TYPE} ---")
    if not os.path.exists(ANALYSIS_OUTPUT_DIR): os.makedirs(ANALYSIS_OUTPUT_DIR); print(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")
    if not os.path.exists(AGGREGATED_INPUT_CSV): print(f"Critical Error: Input file not found for {DATASET_TYPE} at '{os.path.abspath(AGGREGATED_INPUT_CSV)}'"); return
    print(f"Loading aggregated surprisals from {AGGREGATED_INPUT_CSV}...")
    try: aggregated_df_full = pd.read_csv(AGGREGATED_INPUT_CSV)
    except Exception as e: print(f"Error loading {AGGREGATED_INPUT_CSV}: {e}"); return
    if aggregated_df_full.empty: print("Aggregated surprisals file is empty."); return
    
    req_cols = ['sentence_type', 'item_id', 'condition', 'aggregated_surprisal_bits'] # Basic columns needed from input
    missing = [col for col in req_cols if col not in aggregated_df_full.columns]
    if missing: print(f"Error: Input CSV missing: {', '.join(missing)}. Found: {aggregated_df_full.columns.tolist()}"); return
    
    aggregated_df_full['sentence_type'] = aggregated_df_full['sentence_type'].astype(str) # Ensure consistent type
    aggregated_df_filtered = aggregated_df_full[aggregated_df_full['sentence_type'].isin(TARGET_SENTENCE_TYPES)]
    if aggregated_df_filtered.empty: print(f"No data for target types: {TARGET_SENTENCE_TYPES}. Exiting."); return

    # --- Calculate and save main effects (e.g., Wh-effects for Wilcox, or Wh-effects (derived from Fillers) for LAN) ---
    print(f"\nCalculating main '{EFFECT_TYPE_LABEL}' for {DATASET_TYPE} data...")
    effects_df = calculate_effects(aggregated_df_filtered, TARGET_SENTENCE_TYPES, DATASET_TYPE)
    if not effects_df.empty:
        try: effects_df.to_csv(CALCULATED_EFFECTS_CSV, index=False, float_format='%.8f'); print(f"Calculated '{EFFECT_TYPE_LABEL}' saved to {os.path.abspath(CALCULATED_EFFECTS_CSV)}")
        except Exception as e: print(f"Error saving '{EFFECT_TYPE_LABEL}' CSV: {e}")
        print(f"\nGenerating plots for average {EFFECT_TYPE_LABEL.lower()}s...")
        plot_average_effects(effects_df, EFFECT1_NAME, EFFECT2_NAME, EFFECT1_PLOT_LABEL, EFFECT2_PLOT_LABEL, EFFECT_TYPE_LABEL, AVERAGE_EFFECTS_PLOT_PNG)
        print(f"\nGenerating plots for per-item {EFFECT_TYPE_LABEL.lower()}s...")
        plot_item_effects(effects_df, EFFECT1_NAME, EFFECT2_NAME, ITEM_EFFECT1_PLOT_LABEL, ITEM_EFFECT2_PLOT_LABEL, EFFECT_TYPE_LABEL, PER_ITEM_PLOT_DIR)
    else: print(f"No main '{EFFECT_TYPE_LABEL}' were calculated.")
        
    # --- For LAN data: also summarize raw condition surprisals AND calculate Lan et al. (2024) paper metrics ---
    if DATASET_TYPE == "LAN":
        print(f"\nSummarizing raw condition surprisals for LAN data...")
        item_level_lan_surprisals = summarize_lan_condition_surprisals(
            aggregated_df_filtered, 
            TARGET_SENTENCE_TYPES,
            LAN_ITEM_CONDITION_SURPRISALS_CSV, 
            LAN_CONDITION_SUMMARY_CSV      
        )
        
        if item_level_lan_surprisals is not None and not item_level_lan_surprisals.empty:
            print(f"\nCalculating Lan et al. (2024) style metrics for LAN data...")
            calculate_lan_paper_metrics_and_accuracy(
                item_level_lan_surprisals, 
                LAN_PAPER_METRICS_CSV
            )
        else:
            print("Skipping Lan et al. (2024) style metrics calculation due to missing item-level surprisal data.")
            
    print("\nScript finished.")

if __name__ == "__main__":
    main()