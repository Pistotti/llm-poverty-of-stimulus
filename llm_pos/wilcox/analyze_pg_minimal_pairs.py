# analyze_lan_metrics_on_novel_data_full_cr.py
import pandas as pd
import os
from scipy import stats
import numpy as np

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "gpt2"

# Define a single base directory for analysis inputs and outputs
ANALYSIS_BASE_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results")
# Define the output subdirectory for this specific analysis
ANALYSIS_OUTPUT_DIR = os.path.join(ANALYSIS_BASE_DIR, "analysis", "novel_pg_full_cr_analysis")

# --- INPUT FILE ---
# This is the output from your script that aggregated surprisals for the FULL critical region
AGGREGATED_SURPRISALS_CSV = os.path.join(ANALYSIS_BASE_DIR, "aggregated", "novel_pg_full_paradigm_aggregated_surprisals.csv")

# --- OUTPUT FILES ---
# Main CSV output with all per-item calculations
FULL_CR_METRICS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, "lan_metrics_on_full_cr.csv")
# File for the summary of the statistical tests
STAT_SUMMARY_TXT = os.path.join(ANALYSIS_OUTPUT_DIR, "lan_metrics_on_full_cr_stat_summary.txt")

def perform_one_sample_t_test(data_series, column_name, popmean=0, alternative='two-sided'):
    """Performs a one-sample t-test vs 0 and returns a results string."""
    cleaned_data = data_series.dropna()
    if len(cleaned_data) < 2:
        return f"--- T-test for '{column_name}' ---\n  Not enough data (N={len(cleaned_data)}).\n"

    mean_val, std_val, n_val = cleaned_data.mean(), cleaned_data.std(), len(cleaned_data)
    t_statistic, p_value = stats.ttest_1samp(cleaned_data, 0, alternative=alternative)
    
    try:
        if hasattr(stats.t, 'confidence_interval'):
            ci_obj = stats.t.confidence_interval(0.95, df=n_val-1, loc=mean_val, scale=stats.sem(cleaned_data))
            conf_int_str = f"[{ci_obj.low:.4f}, {ci_obj.high:.4f}]"
        else:
            conf_int = stats.t.interval(0.95, len(cleaned_data)-1, loc=mean_val, scale=stats.sem(cleaned_data))
            conf_int_str = f"[{conf_int[0]:.4f}, {conf_int[1]:.4f}]"
    except:
        conf_int_str = "N/A"

    p_value_print = f"{p_value:.4f}" if p_value >= 0.0001 else "< .0001"
    
    result_str = (
        f"--- One-Sample T-test for '{column_name}' (H0: mean=0, HA: mean {'!=' if alternative=='two-sided' else '>'} 0) ---\n"
        f"  N: {n_val}, Mean: {mean_val:.4f}, SD: {std_val:.4f}\n"
        f"  95% CI: {conf_int_str}\n"
        f"  t({n_val-1}) = {t_statistic:.4f}, p = {p_value_print}\n"
    )
    return result_str

def main():
    print("--- Analyzing Lan-style Metrics on FULL Critical Regions ---")
    
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
        print(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")

    if not os.path.exists(AGGREGATED_SURPRISALS_CSV):
        print(f"Error: Input file not found at '{os.path.abspath(AGGREGATED_SURPRISALS_CSV)}'"); return

    try:
        df = pd.read_csv(AGGREGATED_SURPRISALS_CSV)
    except Exception as e:
        print(f"Error reading {AGGREGATED_SURPRISALS_CSV}: {e}"); return

    # --- Step 1: Reshape the data ---
    try:
        # Use the systematic condition names from your novel_pg_full_paradigm.csv
        item_level_df = df.pivot_table(
            index=['sentence_type', 'item_id'],
            columns='condition',
            values='aggregated_surprisal_bits'
        ).reset_index()
    except Exception as e:
        print(f"Error pivoting data. Details: {e}"); return
    
    print(f"Reshaped data to {item_level_df.shape[0]} items.")
    
    # --- Step 2: Calculate Lan et al. (2024) style metrics on FULL region surprisals ---
    s_pfpg = 'plusF_plusG1_plusG2'    # Gapped G2 (+F) -> CR is an adverb
    s_pfmg = 'plusF_plusG1_minusG2'   # Filled G2 (+F) -> CR is an NP like "the campaign"
    s_mfpg = 'minusF_minusG1_plusG2'  # Gapped G2 (-F)
    s_mfmg = 'minusF_minusG1_minusG2' # Filled G2 (-F)

    required_cols = [s_pfpg, s_pfmg, s_mfpg, s_mfmg]
    if not all(col in item_level_df.columns for col in required_cols):
        print(f"Error: Pivoted data is missing one or more required condition columns. Found: {item_level_df.columns.tolist()}"); return

    item_level_df['delta_plus_filler'] = item_level_df[s_pfmg] - item_level_df[s_pfpg]
    item_level_df['delta_minus_filler'] = item_level_df[s_mfmg] - item_level_df[s_mfpg]
    item_level_df['did_effect'] = item_level_df['delta_plus_filler'] - item_level_df['delta_minus_filler']
    item_level_df['success_delta_plus_filler'] = item_level_df['delta_plus_filler'] > 0
    item_level_df['success_did'] = item_level_df['did_effect'] > 0

    print("Calculated Lan et al. (2024) style delta and DiD metrics using full critical region surprisals.")

    # --- Step 3: Perform and Save Statistical Tests ---
    all_results_text = [
        f"Statistical Summary for Lan-Style Metrics on FULL Critical Regions\n",
        f"Dataset: {os.path.basename(AGGREGATED_SURPRISALS_CSV)}\n",
        f"Model: {MODEL_NAME}\n", "="*60 + "\n"
    ]
    
    hypotheses_to_test = {
        'delta_plus_filler': {'alt': 'greater', 'desc': 'Lan Metric: Delta_Plus_Filler'},
        'did_effect':        {'alt': 'greater', 'desc': 'Lan Metric: Difference-in-Differences'}
    }
    
    for col, params in hypotheses_to_test.items():
        results_text = perform_one_sample_t_test(item_level_df[col], params['desc'], alternative=params['alt'])
        all_results_text.append(results_text)
    
    try:
        with open(STAT_SUMMARY_TXT, 'w') as f:
            for text_block in all_results_text: f.write(text_block + "\n")
        print(f"\nFull statistical summary saved to: {os.path.abspath(STAT_SUMMARY_TXT)}")
    except Exception as e:
        print(f"Error saving statistical summary: {e}")

    # --- Step 4: Save the Full Metrics CSV ---
    item_level_df.to_csv(FULL_CR_METRICS_CSV, index=False, float_format='%.8f')
    print(f"Full item-level metrics saved to: {os.path.abspath(FULL_CR_METRICS_CSV)}")

    # --- Step 5: Print Accuracy Summary to Console ---
    print("\n--- Accuracy Scores (based on full CR surprisals) ---")
    for success_col, delta_col, desc in [
        ('success_delta_plus_filler', 'delta_plus_filler', "Delta_Plus_Filler > 0"),
        ('success_did', 'did_effect', "Difference-in-Differences > 0")]:
        valid_items = item_level_df.dropna(subset=[delta_col])
        accuracy = valid_items[success_col].mean() * 100 if not valid_items.empty else 0
        n_valid = len(valid_items)
        print(f"  Accuracy for ({desc}): {accuracy:.2f}% (N={n_valid})")

    print("\nAnalysis finished.")

if __name__ == "__main__":
    main()