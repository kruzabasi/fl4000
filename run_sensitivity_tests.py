import os
import shutil
import subprocess
import yaml
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dp_accounting.rdp.rdp_privacy_accountant import RdpAccountant
from dp_accounting.dp_event import GaussianDpEvent
import sys
import pandas as pd

# Ensure src/config.py is importable regardless of CWD
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    import config
except ImportError as e:
    raise ImportError(f"Could not import config.py from {src_path}: {e}")

# --- Experiment Parameter Ranges (from sprint7_experiment_plan.md) ---
M_RANGE = [10, 20, 40, 60, 80, 100]
ALPHA_RANGE = [10.0, 5.0, 1.0, 0.5, 0.1]
EPSILON_RANGE = [float('inf'), 10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
MU_PROX_RANGE = [0, 0.001, 0.01, 0.1, 0.5, 1.0]
C_RANGE = [0.1, 0.25, 0.5, 0.75, 1.0]
SEEDS = [1, 2, 3]

# --- Baseline/defaults ---
BASELINE = {
    'M': 40,
    'ALPHA': 0.5,
    'EPSILON': 10.0,
    'DELTA': 1e-5,
    'MU_PROX': 0.001,
    'C': 0.25,
    'LR': 0.1,
    'E': 5,
    'CLIP_NORM': 0.5,
    'ROUNDS': 100,
    'BATCH_SIZE': 64,
}

TEMPLATE_PATH = Path("configs/sensitivity/template_sensitivity.yaml")
CONFIG_OUT_DIR = Path("configs/sensitivity/")
PARTITION_SCRIPT = Path("src/partition_data.py")
HARNESS_SCRIPT = Path("src/run_experiment.py")
PYTHON = ".venv/bin/python"

# Utility to fill config template
def fill_template(param, value, seed, baseline, data_partition_path, noise_multiplier=None):
    with open(TEMPLATE_PATH) as f:
        config = yaml.safe_load(f)
    config['experiment_id'] = f"SENSITIVITY_{param}_{value}_seed{seed}"
    config['random_seed'] = seed
    config['non_iid_params']['alpha'] = baseline['ALPHA'] if param != 'alpha' else value
    config['fl_params']['clients_per_round'] = int(max(1, (baseline['C'] if param != 'C' else value) * (baseline['M'] if param != 'M' else value)))
    config['fl_params']['local_epochs'] = baseline['E']
    config['fl_params']['batch_size'] = baseline['BATCH_SIZE']
    config['fl_params']['learning_rate'] = baseline['LR']
    config['fl_params']['total_rounds'] = 2  # TEMP: limit to 2 rounds for debugging log output
    config['model_params']['mu_prox'] = baseline['MU_PROX'] if param != 'mu_prox' else value
    config['runs']['fedprox_dp_run']['dp_params']['clip_norm'] = baseline['CLIP_NORM']
    # Ensure target_delta is always a float
    config['runs']['fedprox_dp_run']['dp_params']['target_delta'] = float(baseline['DELTA'])
    if noise_multiplier is not None:
        config['runs']['fedprox_dp_run']['dp_params']['noise_multiplier'] = float(noise_multiplier)
    else:
        config['runs']['fedprox_dp_run']['dp_params']['noise_multiplier'] = 0.0 if (param == 'epsilon' and value == float('inf')) else 0.01
    if param == 'epsilon':
        # Record target epsilon for DP-budget sweep
        config['runs']['fedprox_dp_run']['dp_params']['target_epsilon'] = None if value == float('inf') else float(value)
    config['data_partition_path'] = str(data_partition_path)
    return config

# Utility to compute noise_multiplier (stub - replace with real accountant if needed)
def compute_noise_multiplier(epsilon, delta, q, rounds, clip_norm):
    # Analytical DP-accounting via RdpAccountant for target (epsilon, delta)
    if epsilon == float('inf'):
        return 0.0
    low, high = 0.1, 10.0
    for _ in range(20):
        mid = (low + high) / 2
        acct = RdpAccountant()
        for _ in range(rounds):
            acct.compose(GaussianDpEvent(mid), count=1)
        eps_spent, _ = acct.get_epsilon_and_optimal_order(delta)
        if eps_spent > epsilon:
            low = mid
        else:
            high = mid
    return high

def ensure_partitioned_data(M, alpha):
    out_dir = Path(f"data/federated_M{M}_alpha{alpha}/")
    if out_dir.exists() and any(out_dir.iterdir()):
        return out_dir
    print(f"Partitioning data for M={M}, alpha={alpha}...")
    cmd = [PYTHON, str(PARTITION_SCRIPT), "--M", str(M), "--alpha", str(alpha), "--output_dir", str(out_dir)]
    subprocess.run(cmd, check=True)
    return out_dir

def run_one(config_path):
    print(f"Running experiment: {config_path}")
    cmd = [PYTHON, str(HARNESS_SCRIPT), "--config", str(config_path)]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print(f"Experiment {config_path} failed with code {proc.returncode}")
        print(proc.stderr)
    return proc.returncode

def main():
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Precompute partitions and noise multipliers to avoid redundant work
    partitions_M = {M: ensure_partitioned_data(M, BASELINE['ALPHA']) for M in M_RANGE}
    partitions_alpha = {alpha: ensure_partitioned_data(BASELINE['M'], alpha) for alpha in ALPHA_RANGE}
    baseline_partition = partitions_M[BASELINE['M']]
    noise_multiplier_map = {eps: compute_noise_multiplier(eps, BASELINE['DELTA'], BASELINE['C'], BASELINE['ROUNDS'], BASELINE['CLIP_NORM']) for eps in EPSILON_RANGE}
    runs = []
    # 1. Sweep M (scalability)
    for M in M_RANGE:
        for seed in SEEDS:
            out_dir = partitions_M[M]
            config = fill_template('M', M, seed, BASELINE, out_dir)
            config_path = CONFIG_OUT_DIR / f"vary_M_{M}_seed{seed}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            runs.append(config_path)
    # 2. Sweep alpha (non-IID)
    for alpha in ALPHA_RANGE:
        for seed in SEEDS:
            out_dir = partitions_alpha[alpha]
            config = fill_template('alpha', alpha, seed, BASELINE, out_dir)
            config_path = CONFIG_OUT_DIR / f"vary_alpha_{alpha}_seed{seed}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            runs.append(config_path)
    # 3. Sweep epsilon (DP)
    for epsilon in EPSILON_RANGE:
        for seed in SEEDS:
            # Use baseline partition and cached noise multiplier
            out_dir = baseline_partition
            noise_multiplier = noise_multiplier_map[epsilon]
            config = fill_template('epsilon', epsilon, seed, BASELINE, out_dir, noise_multiplier=noise_multiplier)
            config_path = CONFIG_OUT_DIR / f"vary_epsilon_{epsilon}_seed{seed}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            runs.append(config_path)
    # 4. Sweep mu_prox (FedProx)
    for mu in MU_PROX_RANGE:
        for seed in SEEDS:
            out_dir = baseline_partition
            config = fill_template('mu_prox', mu, seed, BASELINE, out_dir)
            config_path = CONFIG_OUT_DIR / f"vary_mu_prox_{mu}_seed{seed}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            runs.append(config_path)
    # 5. Sweep C (client ratio)
    for C in C_RANGE:
        for seed in SEEDS:
            out_dir = baseline_partition
            config = fill_template('C', C, seed, BASELINE, out_dir)
            config_path = CONFIG_OUT_DIR / f"vary_C_{C}_seed{seed}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            runs.append(config_path)
    # --- Execute all runs ---
    max_workers = min(len(runs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, cp): cp for cp in runs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error running {futures[future]}: {e}")

    # --- Consolidate per-run CSVs into master logs (single-writer) ---
    results_dir = config[RESULTS_DIR]
    final_full = os.path.join(results_dir, 'experiments_full_log.csv')
    final_comp = os.path.join(results_dir, 'experiments_comparison_log.csv')
    full_dfs, comp_dfs = [], []
    for cp in runs:
        cfg = yaml.safe_load(open(cp))
        exp_id = cfg['experiment_id']
        run_name = next(cfg['runs'])
        fpath = os.path.join(results_dir, f"experiments_full_{exp_id}_{run_name}.csv")
        cpath = os.path.join(results_dir, f"experiments_comparison_{exp_id}_{run_name}.csv")
        if os.path.exists(fpath): full_dfs.append(pd.read_csv(fpath))
        if os.path.exists(cpath): comp_dfs.append(pd.read_csv(cpath))
    if full_dfs:
        pd.concat(full_dfs, ignore_index=True).to_csv(final_full, index=False)
    if comp_dfs:
        pd.concat(comp_dfs, ignore_index=True).to_csv(final_comp, index=False)

if __name__ == "__main__":
    main()

# time .venv/bin/python run_sensitivity_tests.py