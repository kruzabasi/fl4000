# Baseline Model Evaluation: Centralized FTSE 100

## Introduction
This document presents a critical evaluation of centralized baseline models for FTSE 100 stock prediction and portfolio simulation. The pipeline aims to assess both predictive and portfolio performance, serving as a benchmark for future federated learning experiments.

## Data Loading
- **Source:** Processed features are loaded from `ftse_processed_features.parquet`.
- **Shape:** 239,333 rows, 11-15 columns (8-12 features + 1 target + metadata, varies by experiment).

## Task Definition
- **Target:** T+1 log return for each stock (predicting next day log return).
- **Features (as defined in `src/baseline_trainer.py`):**
  - `volatility_20`
  - `sma_60`
  - `sma_5`
  - `log_return_lag_2`
  - `log_return_lag_3`
  - `log_return_lag_5`
  - `quarter`
  - `log_return_lag_1`
  - `rsi_14`
  - Note: This list reflects the current `FEATURE_COLS` in the script. Experimental features are tested separately in `experiments/`.
- **Covariance (Σ):** Ledoit-Wolf Shrinkage, rolling 60-day window for portfolio simulation.

## Data Splitting
- **Train:** Up to 2020-12-31 (typically ~187,467 rows)
- **Validation:** 2021-01-01 to 2022-12-31 (typically ~24,144 rows)
- **Test:** 2023-01-01 onward (typically ~27,722 rows)
- **Note:** Splitting is strictly chronological with no overlap between splits. No duplicate (symbol, timestamp) rows detected. **Row counts may vary if processed data is updated.**

## Feature Scaling
- **Scaler:** StandardScaler fit on train, applied to val/test.
- **Artifact:** The scaler is saved to `data/results/baseline_scaler.pkl`.

## Model Tuning and Validation
Validation set (`X_val, y_val`) is used for model selection. Hyperparameter tuning uses cross-validation on the training set. The model with the lowest validation MSE is selected for test evaluation and portfolio simulation. **The model is not refit on the combined train+val set after selection.** Validation and test metrics are logged in `logs/baseline_trainer.log`.

## Predictive Evaluation (Baseline Only)
- **Model Selection:** Ridge, Random Forest, and XGBoost models are tuned using TimeSeriesSplit CV on the training data. The model achieving the lowest Mean Squared Error (MSE) on the **validation set** is selected as the single best model.
- **Final Evaluation:** Only this **single best model** is evaluated on the **test set**.
- **Test Results:** Test MSE and R² for the selected best model are logged in `logs/baseline_trainer.log`. The specific results depend on which model performed best on the validation set during the run. The `baseline_predictions.csv` file contains predictions for the test set from only this best model.
- **Note:** The table previously shown here, displaying results for all three models on the test set, was inaccurate for the current script (`src/baseline_trainer.py`) which only evaluates the single best model on the test set. Results for all models might be available from separate experimental runs.

---

## Experimental Results (For Reference Only)
_The following results are from scripts in `experiments/` and are not part of the official baseline pipeline._

- **Stacking Ensemble:** Test MSE = 0.000266, R² = -0.0210
- **Cross-sectional Spearman (RF):** -0.0038 (no predictive ranking power)
- **Additional engineered features and alternative targets** have been tested, see experiment scripts and logs for details. These results are not included in the official baseline.

---

## Portfolio Simulation
- **Simulation Period:** 2023-01-03 to 2025-04-11
- **Rebalancing:** Weekly (Friday), 119 periods
- **Assets:** Up to 48 FTSE 100 stocks; asset count may vary due to improved NaN handling and pre-simulation filtering.
- **Loop:**
  - At each rebalance, use last 60 days to compute Ledoit-Wolf Shrinkage Σ (covariance matrix) and use model predictions as μ (expected returns).
  - Optimize weights using PyPortfolioOpt for the **single best model** selected via validation performance.
  - Hold if insufficient data or optimization fails.
  - Calculate realized portfolio log return for the best model's strategy.
- **Data Quality Improvements:**
  - Assets with >10% missing returns are dropped pre-simulation.
  - Missing log returns are handled by forward filling, then zero filling if any remain, reducing asset dropouts and improving rebalancing frequency.
  - Automated checks confirm no split overlap, no duplicates, and no lookahead/leakage detected in features or targets.
- **Warnings:**
  - Some periods skipped early due to insufficient history
  - Occasional optimization failures (see logs)

## Data Integrity & Leakage Checks
- Automated diagnostics confirm:
  - No duplicate (symbol, timestamp) rows
  - No overlap between train/val/test splits
  - No lookahead or data leakage in lag features
  - No missing log returns; zero log returns present (likely holidays or no price movement)
  - Target columns are created dynamically during modeling, not present in raw processed data
- **Note:** Automated data integrity and leakage checks are performed by running `experiments/data_leakage_checks.py` and are not part of the main baseline pipeline execution.

## Portfolio Evaluation
- **Metrics:** Calculated using QuantStats for the **single best model** selected via validation performance.
- **Results:** Portfolio metrics (Sharpe, Max Drawdown, VaR, CVaR) for the selected best model are logged in `logs/baseline_trainer.log` and saved to `data/results/baseline_portfolio_metrics.csv`. The specific results depend on which model was selected and its performance during the simulation.
- **Note:** The table previously shown here, displaying results for all three models, was inaccurate for the current script (`src/baseline_trainer.py`) which only simulates and evaluates the single best model. Results for all models might be available from separate experimental runs.

## Recent Changes
- Experiments moved to `experiments/`.
- Automated data integrity and leakage checks added.
- Ensemble stacking, cross-sectional ranking, and additional engineered features have been tested in experiments.
- **Note:** The baseline feature list in the official pipeline still uses `rsi_14`. Some experiments use `rsi` instead, depending on data availability.

## Conclusion
- **Predictive:** Ridge regression struggles to explain variance (R² ≈ 0), but achieves low MSE due to low volatility in daily returns. RF and XGB show improved portfolio metrics.
- **Portfolio:** Sharpe ratios and drawdowns vary by model; RF and XGB outperform Ridge on Sharpe and drawdown. Performance is consistent with modest predictive signals and improved robustness from enhanced data cleaning.
- **Artifacts:**
  - Predictions (best model only): `data/results/baseline_predictions.csv`
  - Portfolio metrics (best model only): `data/results/baseline_portfolio_metrics.csv`
  - Saved Scaler: `data/results/baseline_scaler.pkl`
  - Saved Best Model: `data/results/baseline_{model_name}.pkl` (e.g., `baseline_ridge_model.pkl`)
  - Logs: `logs/baseline_trainer.log`

**Recommendation:** Use these results as a baseline for comparing federated and more advanced models. See logs for detailed warnings, optimization diagnostics, and model-specific performance.
