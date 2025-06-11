# llm-poverty-of-stimulus/llm_pos/wilcox/analyze_pg_minimal_pairs.py
import pandas as pd
import os
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "gpt2"
ANALYSIS_BASE_DIR = os.path.join(CURRENT_SCRIPT_DIR, "tims_results")
ANALYSIS_OUTPUT_DIR = os.path.join(ANALYSIS_BASE_DIR, "analysis", "novel_pg_fine_grained")

# --- INPUT FILE ---
# This is the output from your script that aggregated surprisals for the FULL paradigm
AGGREGATED_SURPRISALS_CSV = os.path.join(ANALYSIS_BASE_DIR, "aggregated", "novel_pg_full_paradigm_aggregated_surprisals.csv")

# --- ITEMS TO EXCLUDE ---
# List of item_ids to exclude from the analysis due to verb-selection issues.
ITEMS_TO_EXCLUDE = ['30', '35', '38', '39']

# --- OUTPUT FILES ---
# Suffix to add to output files to indicate this is a filtered analysis
FILTER_SUFFIX = "_filtered"
FULL_METRICS_CSV = os.path.join(ANALYSIS_OUTPUT_DIR, f"pg_full_analysis_metrics{FILTER_SUFFIX}.csv")
STAT_SUMMARY_TXT = os.path.join(ANALYSIS_OUTPUT_DIR, f"pg_full_analysis_stat_summary{FILTER_SUFFIX}.txt")
P1_P4_PLOT_PNG = os.path.join(ANALYSIS_OUTPUT_DIR, f"pg_minimal_pair_effects_plot{FILTER_SUFFIX}.pdf")
LAN_ACCURACY_PLOT_PNG = os.path.join(ANALYSIS_OUTPUT_DIR, f"pg_lan_style_accuracy_plot{FILTER_SUFFIX}.pdf")


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
            conf_int = stats.t.interval(0.95, n_val-1, loc=mean_val, scale=stats.sem(cleaned_data))
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

def create_accuracy_plot(accuracy_data, output_path):
    """Generates and saves a bar chart of accuracy scores."""
    if not accuracy_data:
        print("No accuracy data to plot.")
        return

    df = pd.DataFrame(accuracy_data)
    
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(7, 5))
    
    barplot = sns.barplot(x='Metric', y='Accuracy (%)', data=df, ax=ax, palette='viridis',
                          edgecolor='black', linewidth=0.8)

    ax.axhline(50, ls='--', color='gray', lw=1, zorder=0)
    ax.text(len(df['Metric'])-0.45, 51.5, "Chance", color='gray', fontsize=9)
    
    for p in barplot.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center',
                       xytext=(0, 9), textcoords='offset points',
                       fontsize=10.5, color='black')

    ax.set_title(f'Lan et al. (2024) Style Accuracy Scores ({MODEL_NAME})', fontsize=15, weight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xticklabels([r'Success Rate for $\Delta_{+filler} > 0$', 'Success Rate for DiD > 0'], fontsize=10)
    ax.set_ylim(0, 100)
    
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Lan-style accuracy plot saved to: {os.path.abspath(output_path)}")
    plt.close(fig)

def main():
    print("--- Combined Analysis of Parasitic Gap Paradigms ---")
    
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)
        print(f"Created output directory: {ANALYSIS_OUTPUT_DIR}")

    if not os.path.exists(AGGREGATED_SURPRISALS_CSV):
        print(f"Error: Input file not found at '{os.path.abspath(AGGREGATED_SURPRISALS_CSV)}'"); return

    df = pd.read_csv(AGGREGATED_SURPRISALS_CSV)

    # --- Step 1: Filter out problematic items ---
    df['item_id'] = df['item_id'].astype(str)
    print(f"Loaded {len(df['item_id'].unique())} unique items from input file.")
    if ITEMS_TO_EXCLUDE:
        print(f"Excluding {len(ITEMS_TO_EXCLUDE)} items due to pre-identified issues: {ITEMS_TO_EXCLUDE}")
        df = df[~df['item_id'].isin(ITEMS_TO_EXCLUDE)]
        print(f"Analysis will proceed with {len(df['item_id'].unique())} items.")
    
    if df.empty:
        print("No data remaining after filtering. Exiting.")
        return

    # --- Step 2: Reshape the data ---
    try:
        item_level_df = df.pivot_table(
            index=['sentence_type', 'item_id'],
            columns='condition',
            values='aggregated_surprisal_bits'
        ).reset_index()
    except Exception as e:
        print(f"Error pivoting data. Details: {e}"); return
    
    print(f"Reshaped data to {item_level_df.shape[0]} items.")
    
    # --- Step 3: Calculate Fine-Grained Minimal Pair Effects (P1-P4) ---
    p1_ungram = "plusF_minusG1_minusG2"; p1_gram = "plusF_plusG1_plusG2"
    p2_complex = "plusF_plusG1_plusG2";   p2_simple = "plusF_minusG1_plusG2"
    p3_gram = "plusF_plusG1_plusG2";     p3_ungram = "minusF_plusG1_plusG2"
    p4_ungram = "plusF_plusG1_minusG2";  p4_gram = "plusF_plusG1_plusG2"
    
    item_level_df['P1_filler_requires_gap'] = item_level_df[p1_ungram] - item_level_df[p1_gram]
    item_level_df['P2_pg_vs_simple_extraction'] = item_level_df[p2_complex] - item_level_df[p2_simple]
    item_level_df['P3_pg_requires_filler'] = item_level_df[p3_ungram] - item_level_df[p3_gram]
    item_level_df['P4_pg_requires_host_gap'] = item_level_df[p4_ungram] - item_level_df[p4_gram]
    print("Calculated surprisal differences for P1-P4 hypotheses.")

    # --- Step 4: Calculate Lan et al. (2024) Style Metrics ---
    lan_pfpg = 'plusF_plusG1_plusG2'
    lan_pfmg = 'plusF_plusG1_minusG2'
    lan_mfpg = 'minusF_minusG1_plusG2'
    lan_mfmg = 'minusF_minusG1_minusG2'
    
    item_level_df['delta_plus_filler'] = item_level_df[lan_pfmg] - item_level_df[lan_pfpg]
    item_level_df['delta_minus_filler'] = item_level_df[lan_mfmg] - item_level_df[lan_mfpg]
    item_level_df['did_effect'] = item_level_df['delta_plus_filler'] - item_level_df['delta_minus_filler']
    item_level_df['success_delta_plus_filler'] = item_level_df['delta_plus_filler'] > 0
    item_level_df['success_did'] = item_level_df['did_effect'] > 0
    print("Calculated Lan et al. style delta, DiD, and success metrics per item.")

    # --- Step 5: Perform and Save All Statistical Tests ---
    all_results_text = [
        f"Statistical Summary for Fine-Grained PG Analysis ({os.path.basename(AGGREGATED_SURPRISALS_CSV)})\n",
        f"Model: {MODEL_NAME}\n", 
        f"NOTE: Analysis run on a filtered dataset excluding items: {ITEMS_TO_EXCLUDE}\n",
        "="*60 + "\n"
    ]
    
    all_results_text.append("\nPart 1: Fine-Grained Minimal Pair Hypotheses (P1-P4)\n" + "-"*50 + "\n")
    p_hypotheses = {
        'P1_filler_requires_gap': 'greater', 'P2_pg_vs_simple_extraction': 'greater',
        'P3_pg_requires_filler': 'greater', 'P4_pg_requires_host_gap': 'greater'
    }
    for col, alt in p_hypotheses.items():
        all_results_text.append(perform_one_sample_t_test(item_level_df[col], f"Hypothesis {col}", alternative=alt))
        
    all_results_text.append("\nPart 2: Lan et al. (2024) Style Metrics (Mean Effects)\n" + "-"*50 + "\n")
    lan_hypotheses = {
        'delta_plus_filler': 'greater', 'delta_minus_filler': 'two-sided', 'did_effect': 'greater'
    }
    for col, alt in lan_hypotheses.items():
        all_results_text.append(perform_one_sample_t_test(item_level_df[col], f"Lan Metric: {col}", alternative=alt))

    with open(STAT_SUMMARY_TXT, 'w') as f:
        for text_block in all_results_text: f.write(text_block)
    print(f"\nFull statistical summary saved to: {os.path.abspath(STAT_SUMMARY_TXT)}")

    # --- Step 6: Save the Full Metrics CSV ---
    item_level_df.to_csv(FULL_METRICS_CSV, index=False, float_format='%.8f')
    print(f"Full item-level metrics saved to: {os.path.abspath(FULL_METRICS_CSV)}")

    # --- Step 7: Print Accuracy Summary and Generate Plot ---
    accuracy_data_for_plot = []
    print("\n--- Lan et al. (2024) Style Accuracy Scores ---")
    for success_col, delta_col, desc in [
        ('success_delta_plus_filler', 'delta_plus_filler', "Delta_Plus_Filler > 0"),
        ('success_did', 'did_effect', "Difference-in-Differences > 0")]:
        valid_items = item_level_df.dropna(subset=[delta_col])
        accuracy = valid_items[success_col].mean() * 100 if not valid_items.empty else 0
        n_valid = len(valid_items)
        print(f"  Accuracy for ({desc}): {accuracy:.2f}% (N={n_valid})")
        accuracy_data_for_plot.append({'Metric': desc, 'Accuracy (%)': accuracy})
    
    create_accuracy_plot(accuracy_data_for_plot, LAN_ACCURACY_PLOT_PNG)

    # Plot for P1-P4 mean effects
    p1_p4_plot_df = item_level_df[['P1_filler_requires_gap', 'P2_pg_vs_simple_extraction', 'P3_pg_requires_filler', 'P4_pg_requires_host_gap']].melt(var_name='Hypothesis', value_name='Surprisal Difference (bits)')
    p1_p4_labels = {
        'P1_filler_requires_gap': 'P1:\nFiller requires Gap',
        'P2_pg_vs_simple_extraction': 'P2:\nPG vs. Simple',
        'P3_pg_requires_filler': 'P3:\nPG requires Filler',
        'P4_pg_requires_host_gap': 'P4:\nPG requires Host'
    }
    p1_p4_plot_df['Hypothesis'] = p1_p4_plot_df['Hypothesis'].map(p1_p4_labels)
    
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.barplot(x='Hypothesis', y='Surprisal Difference (bits)', data=p1_p4_plot_df, ax=ax1, palette='magma', capsize=.1, errorbar=('ci', 95))
    ax1.axhline(0, color='grey', lw=1, linestyle='--')
    ax1.set_title(f'Mean Surprisal Differences for PG Hypotheses ({MODEL_NAME})', fontsize=15, weight='bold')
    ax1.set_xlabel('Hypothesis Tested', fontsize=12)
    ax1.set_ylabel('Mean Surprisal Difference (bits)', fontsize=12)
    ax1.tick_params(axis='x', labelsize=10, rotation=10)
    fig1.tight_layout()
    fig1.savefig(P1_P4_PLOT_PNG, format='pdf', bbox_inches='tight')
    print(f"\nP1-P4 effects plot saved to: {os.path.abspath(P1_P4_PLOT_PNG)}")
    plt.close(fig1)
    
    print("\nAnalysis finished.")

if __name__ == "__main__":
    main()