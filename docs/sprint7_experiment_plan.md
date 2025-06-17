# Sprint 7: Scalability and Sensitivity Analysis Experiment Plan

This document details the systematic experiment design for evaluating the scalability and sensitivity of the federated learning framework, as per Methodology Section 3.3 (Phase 4).

---

## 1. Baseline (Default) Parameter Settings

Baseline values are chosen from Sprint 6 tuning (see `exp_fedprox_best.yaml`, `tune_fedprox.yaml`, and `sprint6_initial_results.md`).

| Parameter           | Symbol         | Baseline Value | Notes                                                        |
|---------------------|---------------|----------------|--------------------------------------------------------------|
| Number of Clients   | M             | 40             | As per Sprint 6, adjust if needed                            |
| Dirichlet α (skew)  | α             | 0.5            | From non_iid_params in best config                           |
| Privacy Budget      | ε             | 10.0           | Use infinity for non-private runs                            |
| DP Delta            | δ             | 1e-5           | Standard DP setting                                          |
| FedProx μ           | μ_prox        | 0.001          | From best config                                             |
| Client Ratio        | C             | 0.25           | Fraction of clients per round (clients_per_round / M)         |
| Learning Rate       | LR            | 0.1            | From best config                                             |
| Local Epochs        | E             | 5              | From best config                                             |
| DP Clip Norm        | clip_norm     | 0.5            | From best config                                             |
| Batch Size          |               | 64             | From best config                                             |
| Rounds              |               | 100            | For main runs, 5 for quick tests                             |

---

## 2. Parameter Ranges and Experiment Structure

For each experiment, vary ONE parameter across its range, holding all others at baseline values.

### 2.1 Number of Clients (M - Scalability)
- **Range:** [10, 20, 40, 60, 80, 100]
- **Fixed:** α=0.5, μ_prox=0.001, C=0.25, LR=0.1, E=5, ε=10.0, δ=1e-5, clip_norm=0.5
- **Partitioning:**
    - For each M, run: `src/partition_data.py --M <M> --alpha 0.5`
    - Data saved to: `data/federated_M<M>_alpha0.5/`
- **Harness:** Simulation must load data from correct partition path for each M.

### 2.2 Dirichlet α (Non-IID Skew)
- **Range:** [10.0, 5.0, 1.0, 0.5, 0.1]
- **Fixed:** M=40, others as baseline
- **Partitioning:**
    - For each α, run: `src/partition_data.py --M 40 --alpha <α>`
    - Data saved to: `data/federated_M40_alpha<α>/`

### 2.3 DP Privacy Budget (ε)
- **Range:** [infinity, 10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
- **Fixed:** M=40, α=0.5, others as baseline
- **Implementation:**
    - Harness/Aggregator must set ε and δ for each run.
    - If ε = infinity, set noise_multiplier = 0 (no DP noise).
    - Use privacy accountant (dp-accounting) to compute noise_multiplier as needed.

### 2.4 FedProx μ (mu_prox)
- **Range:** [0, 0.001, 0.01, 0.1, 0.5, 1.0]
- **Fixed:** M=40, α=0.5, others as baseline
- **Implementation:**
    - Pass mu_prox in training_config for each run.
    - μ_prox=0 = FedAvg (no proximal term).

### 2.5 Client Participation Ratio (C)
- **Range:** [0.1, 0.25, 0.5, 0.75, 1.0]
- **Fixed:** M=40, α=0.5, others as baseline
- **Implementation:**
    - For each C, set clients_per_round = max(1, floor(C*M)).
    - Aggregator/select_clients must use correct K per round.
    - Privacy accountant must use q=K/M for DP runs.

---

## 3. Random Seeds
- **Seeds per run:** 3 (or 5 if feasible)
- **Purpose:** Ensure robust, reproducible results and support statistical significance.

---

## 4. Key Metrics to Record
- Final test Sharpe Ratio
- Final test MSE
- Convergence round (if applicable)
- Final ε (privacy cost, for DP runs)
- Total MB uploaded (communication cost)
- VaR, Max Drawdown, R² (if available)

---

## 5. Execution Notes
- For each parameter sweep, ensure only the target parameter is varied; all others must be baseline.
- Use the correct data partition for M/α sweeps.
- DP runs must use the correct noise_multiplier (computed for each ε, δ, q, rounds, clip_norm).
- All experiment configs, logs, and results must be saved for traceability and regulatory compliance.

---

## 6. Example Naming Conventions
- Data partitions: `data/federated_M<M>_alpha<α>/`
- Experiment runs: `exp_<param>_<value>_<date>`
- Logs/results: Save with parameter and value in filename for clarity.

---

## 7. References
- Sprint 6 configs: `configs/exp_fedprox_best.yaml`, `configs/tune_fedprox.yaml`
- Initial results: `docs/sprint6_initial_results.md`
- Analysis: `notebooks/results_analysis.ipynb`

---

*Prepared: 2025-04-28*
