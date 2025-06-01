# stat_analysis.py
import pandas as pd
import os
from scipy import stats
import numpy as np

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "gpt2" 
ANALYSIS_SUFFIX = "lan_types" 

INPUT_METRICS_CSV = os.path.join(
    CURRENT_SCRIPT_DIR, "tims_results", "analysis", "lan", 
    f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_lan_paper_metrics.csv"
)
# This ^ file contains the item-level raw surprisals needed: PLUS_FILLER_PLUS_GAP, etc.

STAT_RESULTS_OUTPUT_TXT = os.path.join(
    CURRENT_SCRIPT_DIR, "tims_results", "analysis", "lan",
    f"{MODEL_NAME}_{ANALYSIS_SUFFIX}_lan_paper_stat_summary.txt" # Will append to this
)

COLUMNS_TO_TEST_LAN_PAPER_METRICS = [ # For Lan et al. style deltas
    'delta_plus_filler',
    'delta_minus_filler',
    'did_effect'
]

# Column names for raw surprisals from ..._lan_paper_metrics.csv 
# (which originally come from ..._item_condition_surprisals.csv)
S_PFPG_COL = "PLUS_FILLER_PLUS_GAP"    # Surprisal of gapped with +Filler (e.g., "soon")
S_MFPG_COL = "MINUS_FILLER_PLUS_GAP"   # Surprisal of gapped with -Filler (e.g., "soon")
S_PFMG_COL = "PLUS_FILLER_MINUS_GAP"  # Surprisal of ungapped with +Filler (e.g., "Kim/you")
S_MFMG_COL = "MINUS_FILLER_MINUS_GAP" # Surprisal of ungapped with -Filler (e.g., "Kim/you")

# --- End Configuration ---

def perform_one_sample_t_test(data_series, column_name, popmean=0, alternative='two-sided'):
    """Performs a one-sample t-test on a pandas Series and returns a results string."""
    cleaned_data = data_series.dropna()
    
    if len(cleaned_data) < 2: 
        return f"T-test for '{column_name}': Not enough data (N={len(cleaned_data)} after NaN removal).\n"

    mean_val = cleaned_data.mean()
    std_val = cleaned_data.std()
    n_val = len(cleaned_data)
    
    # For SciPy >= 1.6.0, ttest_1samp has 'alternative' parameter
    t_statistic, p_value = stats.ttest_1samp(cleaned_data, popmean, alternative=alternative)
    
    try:
        conf_int = stats.t.interval(0.95, len(cleaned_data)-1, loc=mean_val, scale=stats.sem(cleaned_data))
        conf_int_str = f"[{conf_int[0]:.4f}, {conf_int[1]:.4f}]"
    except Exception: 
        conf_int_str = "N/A"

    p_value_print = f"{p_value:.4f}"
    if p_value < 0.0001: p_value_print = "< .0001" # Avoid 0.0000 for very small p-values

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
    print(f"--- Statistical Analysis for Lan et al. (2024) Style Metrics & Word Type Surprisals ---")
    print(f"Reading data from: {INPUT_METRICS_CSV}")

    if not os.path.exists(INPUT_METRICS_CSV):
        print(f"Error: Input file not found at {INPUT_METRICS_CSV}"); return

    try: metrics_df = pd.read_csv(INPUT_METRICS_CSV)
    except Exception as e: print(f"Error reading {INPUT_METRICS_CSV}: {e}"); return

    if metrics_df.empty: print("Input metrics file is empty."); return

    all_results_text = [f"Statistical Summary for: {INPUT_METRICS_CSV}\n",
                        f"Model: {MODEL_NAME}\n",
                        f"Analysis Suffix: {ANALYSIS_SUFFIX}\n",
                        "="*60 + "\n",
                        "Part 1: Lan et al. (2024) Style Delta Metrics\n" + "-"*50 + "\n"]
                        
    print("\nPerforming one-sample t-tests for Lan et al. (2024) style delta metrics (H0: mean = 0):")
    for col in COLUMNS_TO_TEST_LAN_PAPER_METRICS:
        if col in metrics_df.columns:
            # For delta_plus_filler and did_effect, Lan et al. predict > 0
            # For delta_minus_filler, prediction is typically also > 0 if defined as S(ungapped)-S(gapped)
            # and ungapped is less preferred (more surprising) than gapped without a filler (which is usually opposite,
            # ungapped is preferred, so S(ungapped) < S(gapped), making delta_minus_filler negative).
            # Given previous results, delta_plus_filler and delta_minus_filler were negative.
            # The DiD was positive. Let's use two-sided for deltas for now, or be specific.
            # Lan et al. success is delta_plus_filler > 0 and DiD > 0.
            # We test if mean is different from 0. The direction can be inferred from mean.
            alt_hypothesis = 'two-sided'
            if col == 'did_effect': # Expected to be positive
                alt_hypothesis = 'greater'
            # For delta_plus_filler, expected positive if model learns PG, but we found negative
            # For delta_minus_filler, expected positive if model disprefers gapped version even more without filler, 
            # or negative if model strongly prefers ungapped version (which is what we found).
            # Let's stick to testing if different from 0 for deltas, and greater for DiD.
            
            result = perform_one_sample_t_test(metrics_df[col], col, alternative=alt_hypothesis if col == 'did_effect' else 'two-sided')
            print(result)
            all_results_text.append(result)
        else:
            warning_msg = f"Warning: Column '{col}' not found. Skipping t-test.\n"
            print(warning_msg); all_results_text.append(warning_msg)
    
    # --- NEW: Test for Average +Gap Surprisal vs Average -Gap Surprisal ---
    all_results_text.append("\nPart 2: Comparison of Average Surprisal for +Gap vs. -Gap Word Types\n" + "-"*70 + "\n")
    print("\n--- Comparing Average Surprisal for +Gap vs. -Gap Word Types ---")

    required_raw_s_cols = [S_PFPG_COL, S_MFPG_COL, S_PFMG_COL, S_MFMG_COL]
    if not all(col in metrics_df.columns for col in required_raw_s_cols):
        print(f"Error: Not all required raw surprisal columns found in input file for +Gap vs -Gap comparison. Missing: "
              f"{[col for col in required_raw_s_cols if col not in metrics_df.columns]}")
    else:
        # Calculate average surprisal for +Gap conditions (words like "soon") per item
        metrics_df['avg_plus_gap_surprisal'] = (metrics_df[S_PFPG_COL] + metrics_df[S_MFPG_COL]) / 2
        
        # Calculate average surprisal for -Gap conditions (words like "Kim", "you") per item
        metrics_df['avg_minus_gap_surprisal'] = (metrics_df[S_PFMG_COL] + metrics_df[S_MFMG_COL]) / 2
        
        # Calculate the difference: (+Gap Surprisal) - (-Gap Surprisal)
        metrics_df['diff_gapped_vs_ungapped_words'] = metrics_df['avg_plus_gap_surprisal'] - metrics_df['avg_minus_gap_surprisal']
        
        # Perform a one-sample t-test on this difference.
        # H0: Mean difference is 0.
        # HA: Mean difference is > 0 (i.e., +Gap word surprisals are greater than -Gap word surprisals)
        result_gap_ungap_diff = perform_one_sample_t_test(
            metrics_df['diff_gapped_vs_ungapped_words'], 
            'Difference: Avg(+Gap Word Surp) - Avg(-Gap Word Surp)',
            alternative='greater' # One-sided test: expect +Gap > -Gap surprisal
        )
        print(result_gap_ungap_diff)
        all_results_text.append(result_gap_ungap_diff)

        # For context, also print means of the averages:
        if 'avg_plus_gap_surprisal' in metrics_df and 'avg_minus_gap_surprisal' in metrics_df:
            mean_avg_plus_gap = metrics_df['avg_plus_gap_surprisal'].dropna().mean()
            mean_avg_minus_gap = metrics_df['avg_minus_gap_surprisal'].dropna().mean()
            context_msg = (f"  Context Mean Avg(+Gap Word Surp): {mean_avg_plus_gap:.4f}\n"
                           f"  Context Mean Avg(-Gap Word Surp): {mean_avg_minus_gap:.4f}\n")
            print(context_msg)
            all_results_text.append(context_msg)

    # Save all results to a text file
    if STAT_RESULTS_OUTPUT_TXT:
        try:
            output_dir = os.path.dirname(STAT_RESULTS_OUTPUT_TXT)
            if output_dir and not os.path.exists(output_dir): # Ensure output_dir is not empty string if script is in results dir
                os.makedirs(output_dir); print(f"Created dir for stat summary: {output_dir}")
            with open(STAT_RESULTS_OUTPUT_TXT, 'w') as f:
                for text_block in all_results_text: f.write(text_block + "\n")
            print(f"\nFull statistical summary saved to: {os.path.abspath(STAT_RESULTS_OUTPUT_TXT)}")
        except Exception as e: print(f"Error saving statistical summary: {e}")

    print("\nStatistical analysis finished.")

if __name__ == "__main__":
    main()