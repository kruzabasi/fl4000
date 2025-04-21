# Privacy in Federated Learning

## Central Differential Privacy (DP)

This project implements **Central Differential Privacy** (DP) to ensure quantifiable privacy guarantees for all participants, in line with data protection regulations and best practices in federated learning (FL).

### 1. Chosen Technique: Central DP
Central DP is applied at the aggregator/server. After collecting updates from clients, the server applies the DP mechanism before updating the global model. This approach is practical and provides strong, quantifiable privacy guarantees [(Dwork & Roth, 2014)].

### 2. Application in the Pipeline

#### a. Clipping (C_clip)
- Each client clips its model update to a maximum L2 norm `C_clip` before sending it to the server.
- This bounds the influence of any single client on the global model, a prerequisite for DP.
- See: [Chen et al., 2024; McMahan et al., 2018]

#### b. Noise Addition (σ)
- The aggregator adds **Gaussian noise** to the sum of the clipped client updates before averaging.
- The noise is sampled from `N(0, σ^2 I)` where `σ` is calibrated based on `C_clip`, the target privacy budget `(ε, δ)`, and the number of clients/rounds.
- The mechanism and calibration are implemented using the [dp-accounting](https://github.com/google/differential-privacy/tree/main/python/accounting) library.
- See: [Chen et al., 2024; Abadi et al., 2016]

#### c. Privacy Budget Accounting
- The cumulative privacy loss `(ε, δ)` is tracked across rounds using a privacy accountant (`RdpAccountant` from `dp-accounting`).
- This allows for rigorous monitoring and enforcement of the privacy budget throughout training.
- See: [Dwork & Roth, 2014]

### 3. Configuration Parameters
- `C_CLIP`: L2 norm bound for client updates.
- `target_epsilon`: Target total privacy budget (ε) for the entire training.
- `target_delta`: Target δ (probability of privacy breach).
- `total_rounds`: Number of federated learning rounds.
- `noise_multiplier`: Ratio σ/C_clip, determined using the accountant for the desired (ε, δ).

### 4. References & Methodology
- **Section 3.7 Privacy-Preserving Techniques** in project methodology.
- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*.
- Abadi, M., et al. (2016). *Deep Learning with Differential Privacy*.
- McMahan, B., et al. (2018). *Learning Differentially Private Recurrent Language Models*.
- Chen, et al. (2024). *Recent Advances in Federated Learning Privacy*.

---

*For further details, see Section 3.7 of the methodology and the referenced papers.*

---

**Completing these steps will integrate Central Differential Privacy into our FedProx framework, providing quantifiable privacy guarantees alongside the federated learning process. Remember to tune the DP parameters (`C_clip`, target ε, δ) carefully, as they directly impact both privacy and model utility.**
