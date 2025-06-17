# %%
"""
Notebook: Scalability & Sensitivity Analysis
Goal: Identify trends and relationships between varied parameters and performance metrics.
Data: data/results/experiments_full_log.csv
"""

# %% Imports
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style='whitegrid')

# Create plots directory
plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(plots_dir, exist_ok=True)

# %% Load full log results
# Adjust the path if needed based on notebook location
df = pd.read_csv('./data/results/experiments_full_log.csv')

# %% Clean data
# Drop rows missing core metrics
df = df.dropna(subset=['mse', 'r2', 'sharpe'])
# Convert numeric columns
numeric_cols = ['elapsed_time_sec', 'final_epsilon', 'final_delta', 'mse', 'r2', 'sharpe',
                'max_drawdown', 'var_95', 'var_99', 'cvar_95']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# %% Extract parameter values
# Parse dict-like strings into actual values
def extract_non_iid_alpha(x):
    if pd.isnull(x):
        return np.nan
    try:
        val = ast.literal_eval(x)
        if isinstance(val, dict):
            return val.get('alpha', np.nan)
        elif isinstance(val, list) and len(val) > 0:
            first = val[0]
            if isinstance(first, dict):
                return first.get('alpha', np.nan)
        return np.nan
    except Exception as e:
        print(f"Error parsing non_iid_params: {x}, error: {e}")
        return np.nan

def extract_fedprox_mu(x):
    if pd.isnull(x):
        return np.nan
    try:
        val = ast.literal_eval(x)
        if isinstance(val, dict):
            return val.get('mu_prox', np.nan)
        elif isinstance(val, list) and len(val) > 0:
            first = val[0]
            if isinstance(first, dict):
                return first.get('mu_prox', np.nan)
        return np.nan
    except Exception as e:
        print(f"Error parsing fl_params: {x}, error: {e}")
        return np.nan

if 'non_iid_params' in df.columns:
    df['non_iid_alpha'] = df['non_iid_params'].apply(extract_non_iid_alpha)
if 'fl_params' in df.columns:
    df['fedprox_mu'] = df['fl_params'].apply(extract_fedprox_mu)
if 'model_params' in df.columns:
    df['model_alpha'] = df['model_params'].apply(
        lambda x: ast.literal_eval(x).get('alpha') if pd.notnull(x) else np.nan)

# Ensure risk_free_rate is numeric
if 'risk_free_rate' in df.columns:
    df['risk_free_rate'] = pd.to_numeric(df['risk_free_rate'], errors='coerce')

# %% Extract sweep param & value from experiment_id

def extract_sweep_param(eid):
    parts = eid.split("_")
    # Handles cases like SENSITIVITY_mu_prox_0.01_seed1
    if len(parts) >= 4 and parts[1] == 'mu' and parts[2] == 'prox':
        return 'mu_prox'
    return parts[1] if len(parts) >= 4 else None

def extract_sweep_value(eid):
    parts = eid.split("_")
    # Handles cases like SENSITIVITY_mu_prox_0.01_seed1
    if len(parts) >= 5 and parts[1] == 'mu' and parts[2] == 'prox':
        val = parts[3]
    elif len(parts) >= 4:
        val = parts[2]
    else:
        return np.nan
    return np.inf if val == 'inf' else float(val) if val.replace('.', '', 1).isdigit() else val

df['sweep_param'] = df['experiment_id'].apply(extract_sweep_param)
df['sweep_value'] = df['experiment_id'].apply(extract_sweep_value)

# %% Filtering, aggregation, and plotting for trend analysis
trend_observations = {}
for param in df['sweep_param'].dropna().unique():
    df_p = df[df['sweep_param'] == param]
    agg = df_p.groupby('sweep_value')[['mse','r2','sharpe']].agg(['mean','std','count']).reset_index()
    print(f"\n==== Parameter: {param} ====")
    print(agg.head())
    obs = []
    if param == 'M':
        mse_trend = agg[('mse','mean')].values
        sharpe_trend = agg[('sharpe','mean')].values
        obs.append(f"MSE: min={mse_trend.min():.6f}, max={mse_trend.max():.6f}, delta={mse_trend.max()-mse_trend.min():.6g}")
        obs.append(f"Sharpe: min={sharpe_trend.min():.3f}, max={sharpe_trend.max():.3f}, delta={sharpe_trend.max()-sharpe_trend.min():.3g}")
        # Plot MSE and Sharpe vs M
        plt.figure(figsize=(8,5))
        plt.errorbar(agg['sweep_value'], agg[('mse','mean')], yerr=agg[('mse','std')], label='MSE', marker='o')
        plt.xlabel('Number of Clients (M)')
        plt.ylabel('MSE')
        plt.title('MSE vs Number of Clients (M)')
        plt.tight_layout()
        mse_path = os.path.join(plots_dir, 'mse_vs_M.png')
        plt.savefig(mse_path)
        plt.close()
        plt.figure(figsize=(8,5))
        plt.errorbar(agg['sweep_value'], agg[('sharpe','mean')], yerr=agg[('sharpe','std')], label='Sharpe', marker='o', color='g')
        plt.xlabel('Number of Clients (M)')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio vs Number of Clients (M)')
        plt.tight_layout()
        sharpe_path = os.path.join(plots_dir, 'sharpe_vs_M.png')
        plt.savefig(sharpe_path)
        plt.close()
    elif param == 'alpha':
        sharpe_trend = agg[('sharpe','mean')].values
        obs.append(f"Sharpe: min={sharpe_trend.min():.3f}, max={sharpe_trend.max():.3f}, delta={sharpe_trend.max()-sharpe_trend.min():.3g}")
        plt.figure(figsize=(8,5))
        plt.errorbar(agg['sweep_value'], agg[('sharpe','mean')], yerr=agg[('sharpe','std')], marker='o', color='g')
        plt.xlabel('Non-IID alpha')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio vs Non-IID Alpha')
        plt.tight_layout()
        sharpe_path = os.path.join(plots_dir, 'sharpe_vs_alpha.png')
        plt.savefig(sharpe_path)
        plt.close()
    elif param == 'epsilon':
        sharpe_trend = agg[('sharpe','mean')].values
        mse_trend = agg[('mse','mean')].values
        obs.append(f"Sharpe: min={sharpe_trend.min():.3f}, max={sharpe_trend.max():.3f}, delta={sharpe_trend.max()-sharpe_trend.min():.3g}")
        obs.append(f"MSE: min={mse_trend.min():.6f}, max={mse_trend.max():.6f}, delta={mse_trend.max()-mse_trend.min():.3g}")
        plt.figure(figsize=(8,5))
        plt.errorbar(agg['sweep_value'], agg[('sharpe','mean')], yerr=agg[('sharpe','std')], marker='o', color='g')
        plt.xlabel('Epsilon (DP)')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio vs Epsilon (DP)')
        plt.tight_layout()
        sharpe_path = os.path.join(plots_dir, 'sharpe_vs_epsilon.png')
        plt.savefig(sharpe_path)
        plt.close()
        plt.figure(figsize=(8,5))
        plt.errorbar(agg['sweep_value'], agg[('mse','mean')], yerr=agg[('mse','std')], marker='o')
        plt.xlabel('Epsilon (DP)')
        plt.ylabel('MSE')
        plt.title('MSE vs Epsilon (DP)')
        plt.tight_layout()
        mse_path = os.path.join(plots_dir, 'mse_vs_epsilon.png')
        plt.savefig(mse_path)
        plt.close()
    elif param == 'mu_prox':
        sharpe_trend = agg[('sharpe','mean')].values
        obs.append(f"Sharpe: min={sharpe_trend.min():.3f}, max={sharpe_trend.max():.3f}, delta={sharpe_trend.max()-sharpe_trend.min():.3g}")
        plt.figure(figsize=(8,5))
        plt.errorbar(agg['sweep_value'], agg[('sharpe','mean')], yerr=agg[('sharpe','std')], marker='o', color='g')
        plt.xlabel('FedProx mu_prox')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio vs FedProx mu_prox')
        plt.tight_layout()
        sharpe_path = os.path.join(plots_dir, 'sharpe_vs_mu_prox.png')
        plt.savefig(sharpe_path)
        plt.close()
    elif param == 'C':
        sharpe_trend = agg[('sharpe','mean')].values
        obs.append(f"Sharpe: min={sharpe_trend.min():.3f}, max={sharpe_trend.max():.3f}, delta={sharpe_trend.max()-sharpe_trend.min():.3g}")
        plt.figure(figsize=(8,5))
        plt.errorbar(agg['sweep_value'], agg[('sharpe','mean')], yerr=agg[('sharpe','std')], marker='o', color='g')
        plt.xlabel('Participation Ratio (C)')
        plt.ylabel('Sharpe Ratio')
        plt.title('Sharpe Ratio vs Participation Ratio (C)')
        plt.tight_layout()
        sharpe_path = os.path.join(plots_dir, 'sharpe_vs_C.png')
        plt.savefig(sharpe_path)
        plt.close()
    trend_observations[param] = obs

# Print all observations
print("\n=== Trend Analysis Observations ===")
for param, obs in trend_observations.items():
    print(f"\nParameter: {param}")
    for o in obs:
        print(f"- {o}")

# %% Aggregate by configuration (excluding seed)
config_cols = ['risk_free_rate', 'non_iid_alpha', 'fedprox_mu', 'model_alpha', 'sweep_param', 'sweep_value']
agg = df.groupby(config_cols).agg(
    mse_mean=('mse', 'mean'), mse_std=('mse', 'std'),
    r2_mean=('r2', 'mean'), r2_std=('r2', 'std'),
    sharpe_mean=('sharpe', 'mean'), sharpe_std=('sharpe', 'std'),
    time_mean=('elapsed_time_sec', 'mean'), time_std=('elapsed_time_sec', 'std')
).reset_index()
