# stat_analysis.py
import pandas as pd
import os
from scipy import stats
import numpy as np

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "gpt2"

# Define a single base directory for analysis inputs and outputs for simplicity
ANALYSIS_BASE_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results")

# --- INPUT FILE ---
# Corrected path to point to the aggregated results for your novel data
INPUT_SURPRISALS_CSV = os.path.join(ANALYSIS_BASE_DIR, "aggregated", "novel_data_extracted_critical_surprisals_gpt2.csv")

# --- OUTPUT FILES ---
# Place outputs in a new, clearly named subdirectory
ANALYSIS_OUTPUT_DIR = os.path.join(ANALYSIS_BASE_DIR, "analysis", "novel_data_stats")
NEW_STATS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, "novel_data_full_metrics.csv")
STAT_SUMMARY_TXT = os.path.join(ANALYSIS_OUTPUT_DIR, "novel_data_statistical_summary.txt")

def perform_one_sample_t_test(data_series, column_name, popmean=0, alternative='two-sided'):
    """Performs a one-sample t-test on a pandas Series and returns a results string."""
    cleaned_data = data_series.dropna()
    
    if len(cleaned_data) < 2:
        return f"--- T-test for '{column_name}' ---\n  Not enough data (N={len(cleaned_data)} after NaN removal).\n"

    mean_val = cleaned_data.mean()
    std_val = cleaned_data.std()
    n_val = len(cleaned_data)
    
    t_statistic, p_value = stats.ttest_1samp(cleaned_data, popmean, alternative=alternative)
    
    try:
        conf_int = stats.t.interval(0.95, len(cleaned_data)-1, loc=mean_val, scale=stats.sem(cleaned_data))
        conf_int_str = f"[{conf_int[0]:.4f}, {conf_int[1]:.4f}]"
    except Exception:
        conf_int_str = "N/A"

    p_value_print = f"{p_value:.4f}"
    if p_value < 0.0001: p_value_print = "< .0001"

    result_str = (
        f"--- One-Sample T-test for '{column_name}' (vs {popmean}, alternative='{alternative}') ---\n"
        f"  N (after NaN removal): {n_val}\n"
        f"  Mean: {mean_val:.4f}\n"
        f"  Std Dev: {std_val:.4f}\n"
        f"  95% CI for Mean: {conf_int_str}\n"
        f"  T-statistic: {t_statistic:.4f}\n"
        f"  P-value: {p_value_print} "
    )
    if p_value < 0.001: result_str += "(sig. at p < .001)\n"
    elif p_value < 0.01: result_str += "(sig. at p < .01)\n"
    elif p_value < 0.05: result_str += "(sig. at p < .05)\n"
    else: result_str += "(not sig. at p < .05)\n"
    
    return result_str

def main():
    print("--- Statistical Analysis of Novel PG Data Critical Surprisals ---")
    
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
        print(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")

    if not os.path.exists(INPUT_SURPRISALS_CSV):
        print(f"Error: Input file not found at '{os.path.abspath(INPUT_SURPRISALS_CSV)}'"); return

    try:
        df = pd.read_csv(INPUT_SURPRISALS_CSV)
    except Exception as e:
        print(f"Error reading {INPUT_SURPRISALS_CSV}: {e}"); return

    if df.empty:
        print("Input surprisals file is empty."); return

    # --- Step 1: Reshape the data ---
    try:
        item_level_df = df.pivot_table(
            index=['sentence_type', 'item_id'],
            columns='condition',
            values='aggregated_surprisal_bits'
        ).reset_index()
    except Exception as e:
        print(f"Error: Could not pivot the data. Check if the input CSV has unique rows for each item-condition pair. Details: {e}")
        return
    
    print(f"Reshaped data to {item_level_df.shape[0]} items.")

    # --- Step 2: Calculate Lan et al. (2024) style metrics ---
    s_pfpg_col = "PFPG"    
    s_mfpg_col = "MFPG"   
    s_pfmg_col = "PFMG"  
    s_mfmg_col = "MFMG" 
    
    required_cols = [s_pfpg_col, s_mfpg_col, s_pfmg_col, s_mfmg_col]
    if not all(col in item_level_df.columns for col in required_cols):
        print(f"Error: Pivoted data is missing one or more required condition columns. Found: {item_level_df.columns.tolist()}"); return

    item_level_df['delta_plus_filler'] = item_level_df[s_pfmg_col] - item_level_df[s_pfpg_col]
    item_level_df['delta_minus_filler'] = item_level_df[s_mfmg_col] - item_level_df[s_mfpg_col]
    item_level_df['did_effect'] = item_level_df['delta_plus_filler'] - item_level_df['delta_minus_filler']
    item_level_df['success_delta_plus_filler'] = item_level_df['delta_plus_filler'] > 0
    item_level_df['success_did'] = item_level_df['did_effect'] > 0

    print("Calculated Lan et al. (2024) style delta and DiD metrics per item.")

    # --- Step 3: Calculate metrics for baseline word type comparison ---
    item_level_df['avg_surprisal_plus_gap_words'] = (item_level_df[s_pfpg_col] + item_level_df[s_mfpg_col]) / 2
    item_level_df['avg_surprisal_minus_gap_words'] = (item_level_df[s_pfmg_col] + item_level_df[s_mfmg_col]) / 2
    item_level_df['diff_gapped_vs_ungapped_words'] = item_level_df['avg_surprisal_plus_gap_words'] - item_level_df['avg_surprisal_minus_gap_words']
    
    print("Calculated average surprisal difference between +Gap and -Gap critical words.")

    # --- Step 4: Perform statistical tests ---
    all_results_text = [
        f"Statistical Summary for: {os.path.basename(INPUT_SURPRISALS_CSV)}\n",
        f"Model: {MODEL_NAME}\n",
        "="*60 + "\n"
    ]
    
    print("\nPerforming statistical tests...")
    
    lan_metrics_results = ""
    for col in ['delta_plus_filler', 'delta_minus_filler', 'did_effect']:
        alt = 'greater' if col != 'delta_minus_filler' else 'two-sided'
        lan_metrics_results += perform_one_sample_t_test(item_level_df[col], f"Lan Metric: {col}", alternative=alt)
    all_results_text.append("Part 1: Lan et al. (2024) Style Delta Metrics\n" + "-"*50 + "\n" + lan_metrics_results)
    
    word_type_diff_results = perform_one_sample_t_test(
        item_level_df['diff_gapped_vs_ungapped_words'],
        'Difference: Avg(+Gap Word Surp) - Avg(-Gap Word Surp)',
        alternative='two-sided'
    )
    all_results_text.append("\nPart 2: Comparison of Baseline Surprisal for Critical Word Types\n" + "-"*70 + "\n" + word_type_diff_results)
    
    mean_avg_plus_gap = item_level_df['avg_surprisal_plus_gap_words'].dropna().mean()
    mean_avg_minus_gap = item_level_df['avg_surprisal_minus_gap_words'].dropna().mean()
    context_msg = (f"  Context Mean Avg(+Gap Word Surp): {mean_avg_plus_gap:.4f}\n"
                   f"  Context Mean Avg(-Gap Word Surp): {mean_avg_minus_gap:.4f}\n")
    all_results_text.append(context_msg)

    # --- Step 5: Save outputs ---
    try:
        item_level_df.to_csv(NEW_STATS_CSV, index=False, float_format='%.8f')
        print(f"\nFull item-level metrics and calculations saved to: {os.path.abspath(NEW_STATS_CSV)}")
    except Exception as e:
        print(f"Error saving detailed stats CSV: {e}")
        
    try:
        with open(STAT_SUMMARY_TXT, 'w') as f:
            for text_block in all_results_text:
                f.write(text_block)
        print(f"Full statistical summary saved to: {os.path.abspath(STAT_SUMMARY_TXT)}")
    except Exception as e:
        print(f"Error saving statistical summary text file: {e}")

    print("\nStatistical analysis finished.")

if __name__ == "__main__":
    main()