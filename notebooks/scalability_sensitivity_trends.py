"""
scalability_sensitivity_trends.py

Task: Creating Visualizations (Task 5)
Goal: Visually represent the trends identified in the analysis.

- Uses: matplotlib, seaborn
- Reads consolidated experiment results.
- Generates publication-ready plots for each parameter sweep with error bars.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

# --- Config ---
RESULTS_PATH = os.path.join(os.path.dirname(__file__), '../data/results/experiments_full_log.csv')
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(RESULTS_PATH)

# --- Parameter Extraction (as in analysis) ---
def extract_sweep_param(eid):
    parts = eid.split('_')
    return parts[1] if len(parts) >= 4 else None

def extract_sweep_value(eid):
    parts = eid.split('_')
    if len(parts) < 4:
        return None
    val = parts[2]
    try:
        return float(val)
    except ValueError:
        return val

df['sweep_param'] = df['experiment_id'].apply(extract_sweep_param)
df['sweep_value'] = df['experiment_id'].apply(extract_sweep_value)

# --- Plotting Helper ---
def save_plot(fig, name):
    fig.savefig(os.path.join(PLOTS_DIR, name), bbox_inches='tight')
    plt.close(fig)

# --- Plot Definitions ---
plots = [
    # (param, metric, y-label, plot-filename, plot-title)
    ('M', 'sharpe', 'Sharpe Ratio', 'sharpe_vs_M.png', 'Sharpe Ratio vs Number of Clients (M)'),
    ('M', 'mse', 'MSE', 'mse_vs_M.png', 'MSE vs Number of Clients (M)'),
    ('alpha', 'sharpe', 'Sharpe Ratio', 'sharpe_vs_alpha.png', 'Sharpe Ratio vs Non-IID Alpha'),
    ('epsilon', 'sharpe', 'Sharpe Ratio', 'sharpe_vs_epsilon.png', 'Sharpe Ratio vs Epsilon (DP)'),
    ('epsilon', 'mse', 'MSE', 'mse_vs_epsilon.png', 'MSE vs Epsilon (DP)'),
    ('mu_prox', 'sharpe', 'Sharpe Ratio', 'sharpe_vs_mu_prox.png', 'Sharpe Ratio vs FedProx mu_prox'),
    ('C', 'sharpe', 'Sharpe Ratio', 'sharpe_vs_C.png', 'Sharpe Ratio vs Participation Ratio (C)'),
    # Add more as needed
]

for param, metric, ylab, fname, title in plots:
    df_p = df[df['sweep_param'] == param]
    if df_p.empty:
        continue
    agg = df_p.groupby('sweep_value')[metric].agg(['mean', 'std', 'count']).reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    ax.errorbar(agg['sweep_value'], agg['mean'], yerr=agg['std'], fmt='o-', capsize=4, label=metric.capitalize())
    ax.set_xlabel(param)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend()
    save_plot(fig, fname)

# Example: Plotting convergence rounds vs mu_prox if available
if 'convergence_rounds' in df.columns:
    param = 'mu_prox'
    df_p = df[df['sweep_param'] == param]
    if not df_p.empty:
        agg = df_p.groupby('sweep_value')['convergence_rounds'].agg(['mean', 'std']).reset_index()
        fig, ax = plt.subplots(figsize=(8,5))
        ax.errorbar(agg['sweep_value'], agg['mean'], yerr=agg['std'], fmt='o-', capsize=4, color='purple')
        ax.set_xlabel('FedProx mu_prox')
        ax.set_ylabel('Convergence Rounds')
        ax.set_title('Convergence Rounds vs FedProx mu_prox')
        save_plot(fig, 'convergence_vs_mu_prox.png')

# Example: Plotting total communication cost vs C if available
if 'metric_communication_total_MB_uploaded' in df.columns:
    param = 'C'
    df_p = df[df['sweep_param'] == param]
    if not df_p.empty:
        agg = df_p.groupby('sweep_value')['metric_communication_total_MB_uploaded'].agg(['mean', 'std']).reset_index()
        fig, ax = plt.subplots(figsize=(8,5))
        ax.errorbar(agg['sweep_value'], agg['mean'], yerr=agg['std'], fmt='o-', capsize=4, color='orange')
        ax.set_xlabel('Participation Ratio (C)')
        ax.set_ylabel('Total Communication Uploaded (MB)')
        ax.set_title('Total Communication Cost vs Participation Ratio (C)')
        save_plot(fig, 'communication_vs_C.png')

# Example: Plotting final epsilon vs C if available
if 'final_epsilon' in df.columns:
    param = 'C'
    df_p = df[df['sweep_param'] == param]
    if not df_p.empty:
        agg = df_p.groupby('sweep_value')['final_epsilon'].agg(['mean', 'std']).reset_index()
        fig, ax = plt.subplots(figsize=(8,5))
        ax.errorbar(agg['sweep_value'], agg['mean'], yerr=agg['std'], fmt='o-', capsize=4, color='red')
        ax.set_xlabel('Participation Ratio (C)')
        ax.set_ylabel('Final Epsilon')
        ax.set_title('Final Epsilon vs Participation Ratio (C)')
        save_plot(fig, 'final_epsilon_vs_C.png')

print('All trend visualizations generated and saved to notebooks/plots/')
