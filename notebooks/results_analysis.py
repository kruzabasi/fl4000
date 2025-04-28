import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Load Results ---
results_path = './data/results/experiments_comparison_log.csv'
df = pd.read_csv(results_path)

# --- Filter Runs (edit as needed for your experiment IDs/model types) ---
# Example: filter for baseline and initial FedProx+DP runs
relevant_models = ['Centralized', 'LocalOnly', 'FedAvg', 'FedProxDP']
df_filtered = df[df['model_type'].isin(relevant_models)]

# --- Create Comparison Table ---
# Only include columns that exist in the CSV (for backward compatibility)
expected_cols = ['experiment_id', 'model_type', 'Sharpe', 'Max Drawdown', 'VaR_95', 'CVaR_95', 'elapsed_time_sec', 'final_delta']
comparison_cols = [col for col in expected_cols if col in df_filtered.columns]
comparison_table = df_filtered[comparison_cols].drop_duplicates()

# --- Load Baseline Results ---
baseline_path = './data/results/baseline/baseline_portfolio_metrics.csv'
baseline_df = pd.read_csv(baseline_path, index_col=0)
baseline_df = baseline_df.reset_index().rename(columns={'index': 'model_type'})
baseline_df['experiment_id'] = 'Baseline'

# --- Combine Baseline and FL Results ---
# Align columns for concatenation
baseline_df = baseline_df[['experiment_id', 'model_type', 'Sharpe', 'Max Drawdown', 'VaR_95', 'CVaR_95']]
combined_table = pd.concat([comparison_table, baseline_df], ignore_index=True)

def plot_metric_bar(df, metric, model_order, title, ylabel, fname, palette=None, annotate_decimals=2):
    plt.figure(figsize=(9, 5))
    plot_df = df[['model_type', metric]].copy()
    plot_df = plot_df[plot_df['model_type'].isin(model_order)]
    plot_df['model_type'] = pd.Categorical(plot_df['model_type'], categories=model_order, ordered=True)
    plot_df = plot_df.sort_values('model_type')
    ax = sns.barplot(
        x='model_type', y=metric, data=plot_df,
        order=model_order, palette=palette, edgecolor='black', width=0.7
    )
    plt.title(title, fontsize=16, weight='bold')
    plt.ylabel(ylabel, fontsize=13)
    plt.xlabel("Model", fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for p in ax.patches:
        val = p.get_height()
        if pd.notnull(val):
            ax.annotate(f"{val:.{annotate_decimals}f}",
                        (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom' if val >= 0 else 'top',
                        fontsize=11, color='black', weight='bold',
                        xytext=(0, 5 if val >= 0 else -12),
                        textcoords='offset points')
    ylim = plot_df[metric].quantile([0.01, 0.99]).values
    if ylim[0] == ylim[1]:
        ylim = (ylim[0] - 1, ylim[1] + 1)
    plt.ylim(ylim[0] - abs(ylim[0])*0.2, ylim[1] + abs(ylim[1])*0.2)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    print(f"Saved improved bar plot: {fname}")
    plt.show()

# --- Improved Bar plots for key metrics ---
sns.set(style="whitegrid")
plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(plots_dir, exist_ok=True)

bar_metrics = [
    ('Sharpe', 'Sharpe', 'Sharpe'),
    ('Max Drawdown', 'Max Drawdown', 'Max Drawdown'),
    ('VaR_95', 'VaR_95', 'VaR 95'),
    ('CVaR_95', 'CVaR_95', 'CVaR 95'),
    ('final_delta', 'final_delta', 'Final Delta'),
]

model_order = ['FedProxDP', 'ridge', 'randomforest', 'xgboost']
palette = sns.color_palette("Set2", len(model_order))

for title, metric, ylabel in bar_metrics:
    fname = os.path.join(plots_dir, f"bar_{metric}.png")
    if metric in combined_table.columns:
        plot_metric_bar(combined_table, metric, model_order, f"{title} by Model", ylabel, fname, palette=palette, annotate_decimals=3 if 'delta' in metric else 2)

# --- Print Comparison Table Nicely ---
print("\n=== Comparison Table ===\n")
metrics_to_plot = ['Sharpe', 'Max Drawdown', 'VaR_95', 'CVaR_95']
for metric in metrics_to_plot + ['elapsed_time_sec', 'final_delta']:
    if metric in combined_table.columns:
        vals = []
        for idx, row in combined_table.iterrows():
            try:
                v = float(row[metric])
                if not (pd.isna(v) or pd.isnull(v) or v == float('inf') or v == float('-inf')):
                    vals.append((row['model_type'], v))
            except Exception:
                continue
        print(f"{metric:20}: ", end='')
        for label, v in vals:
            print(f"{label}: {v}", end=' | ')
        print()

# --- (Optional) Save combined table ---
combined_table.to_csv('./data/results/experiments_comparison_with_baselines.csv', index=False)
