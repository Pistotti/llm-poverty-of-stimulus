# llm-poverty-of-stimulus/llm_pos/wilcox/generate_comparison_plot.py
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# --- Data Configuration with Sample Sizes (N) ---
# N values are crucial for calculating confidence intervals.
results_data = [
    # Data for the "Direct Preference (Δ_filler > 0)" metric
    {"Dataset": "Original Lan et al.", "Metric": "Δ_filler > 0 Accuracy", "Accuracy": 5.61, "N": 8064},
    {"Dataset": "Filtered Lan et al.", "Metric": "Δ_filler > 0 Accuracy", "Accuracy": 7.01, "N": 5760},
    {"Dataset": "Refined Novel Data", "Metric": "Δ_filler > 0 Accuracy", "Accuracy": 60.0, "N": 10},
    
    # Data for the "Difference-in-Differences (DiD)" metric
    {"Dataset": "Original Lan et al.", "Metric": "DiD Accuracy", "Accuracy": 68.75, "N": 8064},
    {"Dataset": "Filtered Lan et al.", "Metric": "DiD Accuracy", "Accuracy": 72.93, "N": 5760},
    {"Dataset": "Refined Novel Data", "Metric": "DiD Accuracy", "Accuracy": 80.0, "N": 10},
]

# --- Output Configuration ---
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_SCRIPT_DIR, "figures")
OUTPUT_FILENAME = "gpt2_pg_performance_comparison_with_ci.pdf" # PDF for high quality

def calculate_wilson_score_interval(p, n, z=1.96):
    """
    Calculates the Wilson score interval for a proportion. Returns the distance
    from the proportion to the lower and upper bounds.
    """
    if n == 0:
        return (0, 0)
        
    p_obs = p / 100.0 # Convert percentage to proportion for calculation
        
    denominator = 1 + z**2 / n
    numerator_term1 = p_obs + z**2 / (2 * n)
    numerator_term2 = z * np.sqrt((p_obs * (1 - p_obs) / n) + (z**2 / (4 * n**2)))
    
    center_adjusted = numerator_term1 / denominator
    
    lower_bound = center_adjusted - (numerator_term2 / denominator)
    upper_bound = center_adjusted + (numerator_term2 / denominator)
    
    # Return the size of the error bars (distance from observed p) as a tuple
    return (p_obs - lower_bound, upper_bound - p_obs)

def create_comparison_plot_with_ci(data, output_path):
    """
    Generates and saves a bar chart with 95% confidence intervals,
    manually drawing the error bars and caps for maximum reliability.
    """
    if not data:
        print("No data provided to plot.")
        return
        
    df = pd.DataFrame(data)

    # Calculate asymmetric confidence intervals for each row
    ci_errors_asymmetric = [calculate_wilson_score_interval(row['Accuracy'], row['N']) for _, row in df.iterrows()]
    df['ci_lower_err'] = [err[0] * 100 for err in ci_errors_asymmetric]
    df['ci_upper_err'] = [err[1] * 100 for err in ci_errors_asymmetric]
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    # Step 1: Draw the bars using Seaborn to get positions and colors
    barplot = sns.barplot(
        data=df,
        x="Dataset",
        y="Accuracy",
        hue="Metric",
        palette="viridis",
        edgecolor="black",
        linewidth=0.8,
        ax=ax,
        errorbar=None 
    )

    # --- Step 2: Manually Draw Error Bars and Caps ---
    # This loop iterates through the bars seaborn has drawn
    for i, bar in enumerate(barplot.patches):
        # Get the corresponding data row. This relies on the order of patches.
        # Seaborn groups by hue, so first all bars for Metric 1, then all for Metric 2
        # A more robust mapping might be needed for complex plots, but for this 3x2, this should hold.
        if i >= len(df): continue # Safeguard if more patches than data
            
        data_row = df.iloc[i]
        
        # Bar's center x-position and height
        bar_x = bar.get_x() + bar.get_width() / 2
        bar_y = bar.get_height()
        
        # Error values for this bar
        lower_error = data_row['ci_lower_err']
        upper_error = data_row['ci_upper_err']
        
        # Define cap width as a fraction of bar width
        cap_width = bar.get_width() * 0.25
        
        # Draw the vertical error bar line
        ax.plot([bar_x, bar_x], [bar_y - lower_error, bar_y + upper_error], c='black', linewidth=1)
        
        # Draw the bottom cap
        ax.plot([bar_x - cap_width, bar_x + cap_width], [bar_y - lower_error, bar_y - lower_error], c='black', linewidth=1)
        
        # Draw the top cap
        ax.plot([bar_x - cap_width, bar_x + cap_width], [bar_y + upper_error, bar_y + upper_error], c='black', linewidth=1)

        # Add text labels on top of each bar, positioned above the error bar
        if not np.isnan(bar_y):
            text_y_position = bar_y + upper_error + 2 # Position text 2 points above the upper cap
            ax.annotate(f"{bar_y:.1f}%",
                        (bar_x, text_y_position),
                        ha='center', va='bottom',
                        fontsize=9.5,
                        color='black')

    # --- Customize Appearance ---
    ax.set_title("GPT-2 Performance on Parasitic Gaps Across Stimulus Datasets", fontsize=14, weight='bold', pad=15)
    ax.set_xlabel("Stimulus Dataset", fontsize=12, weight='bold')
    ax.set_ylabel("Accuracy (%) with 95% CI", fontsize=12, weight='bold')
    
    ax.set_ylim(0, 105)
    ax.axhline(50, ls='--', color='gray', lw=1, zorder=0)
    
    # Set x-ticks to be in the middle of each group of bars
    datasets = df['Dataset'].unique()
    index = np.arange(len(datasets))
    ax.set_xticks(index)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    
    # Re-create legend handles as barplot might not have them if called with ax
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title="Evaluation Metric", title_fontsize='11', fontsize='10', frameon=False, loc='upper left')
    
    sns.despine(ax=ax)
    fig.tight_layout()

    # --- Save Figure ---
    # ... (saving logic remains the same) ...
    output_dir_path = os.path.dirname(output_path)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    try:
        fig.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Figure successfully saved to: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Error saving figure: {e}")

# The rest of the script (data definition, __main__ block) remains the same

if __name__ == "__main__":
    create_comparison_plot_with_ci(results_data, os.path.join(OUTPUT_DIR, OUTPUT_FILENAME))