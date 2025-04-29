# Scalability & Sensitivity Analysis: Key Observations

This document records the key **qualitative** and **quantitative** observations for each parameter sweep from the federated learning sensitivity analysis. All results are based on the FTSE 100 portfolio optimization experiments, analyzed according to DSR methodology.

---

## 1. Scalability (Number of Clients, M)
- **Quantitative:**
    - MSE: min=0.000263, max=0.000267, delta=3.64e-06
    - Sharpe: min=1.943, max=3.276, delta=1.33
- **Qualitative:**
    - Sharpe ratio shows moderate variation with increasing clients, indicating some sensitivity but no catastrophic degradation.
    - MSE remains stable, suggesting the model scales robustly in terms of prediction error.
    - No evidence of super-linear communication cost or convergence slowdown observed (communication/convergence metrics not plotted here).
    - See plot below for detailed trend:

    ![Sharpe Ratio vs Number of Clients (M)](../notebooks/plots/sharpe_vs_M.png)
    ![MSE vs Number of Clients (M)](../notebooks/plots/mse_vs_M.png)

---

## 2. Non-IIDness (Alpha)
- **Quantitative:**
    - Sharpe: min=3.113, max=3.113, delta=0.000386
- **Qualitative:**
    - Sharpe ratio is nearly flat across all alpha values; performance is robust to non-IID data skew in this experiment.
    - No significant drop as skew increases (alpha decreases).
    - Further analysis with FedProx vs FedAvg could reveal nuanced effects.
    - See plot below for detailed trend:

    ![Sharpe Ratio vs Non-IID Alpha](../notebooks/plots/sharpe_vs_alpha.png)

---

## 3. Privacy (Epsilon, DP)
- **Quantitative:**
    - Sharpe: min=3.077, max=3.113, delta=0.0358
    - MSE: min=0.000265, max=0.000265, delta=2.41e-08
- **Qualitative:**
    - Utility (Sharpe, MSE) is stable across a wide range of epsilon values; no sharp utility cliff observed.
    - The system appears robust to privacy constraints in this setting.
    - See plots below for detailed trends:

    ![Sharpe Ratio vs Epsilon (DP)](../notebooks/plots/sharpe_vs_epsilon.png)
    ![MSE vs Epsilon (DP)](../notebooks/plots/mse_vs_epsilon.png)

---

## 4. FedProx (mu_prox)
- **Quantitative:**
    - Sharpe: min=3.112, max=3.113, delta=0.000494
- **Qualitative:**
    - Sharpe ratio is very stable across mu_prox values; no clear optimal point or instability.
    - Model is not highly sensitive to FedProx regularization in this regime.
    - See plot below for detailed trend:

    ![Sharpe Ratio vs FedProx mu_prox](../notebooks/plots/sharpe_vs_mu_prox.png)

---

## 5. Participation Ratio (C)
- **Quantitative:**
    - Sharpe: min=1.943, max=3.113, delta=1.17
- **Qualitative:**
    - Sharpe ratio shows more variation with participation ratio, indicating that reduced participation can impact performance.
    - Lower participation (smaller C) can lead to reduced Sharpe and higher variability.
    - Implications for communication and privacy amplification not directly measured here.
    - See plot below for detailed trend:

    ![Sharpe Ratio vs Participation Ratio (C)](../notebooks/plots/sharpe_vs_C.png)

---

*All observations are based on the current experiment logs and plots. For further insights, see the generated plots in `notebooks/plots/` and the quantitative summaries in the analysis notebook.*
