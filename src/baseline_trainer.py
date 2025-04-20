import pandas as pd
import numpy as np
import os
import logging
import pickle
from pathlib import Path  # Use pathlib for cleaner path handling
from typing import List, Dict, Any

# Scikit-learn imports
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.multioutput import MultiOutputRegressor # If needed

# XGBoost import
try:
    import xgboost as xgb
except ImportError:
    logging.warning("xgboost not found. XGBoost baseline will not be available.")
    xgb = None

# Portfolio Optimization and Metrics imports
try:
    from pypfopt import EfficientFrontier, objective_functions, risk_models, expected_returns
    HAS_PYPFOpt = True
except ImportError:
    logging.warning("pypfopt not installed. Portfolio optimization simulation will be limited.")
    HAS_PYPFOpt = False

try:
    import quantstats as qs
    HAS_QUANTSTATS = True
except ImportError as e:
    logging.warning(f"quantstats not installed. Advanced portfolio metrics will not be calculated. ImportError: {e}")
    HAS_QUANTSTATS = False
except Exception as e:
    logging.warning(f"quantstats import failed with unexpected error: {e}")
    HAS_QUANTSTATS = False


# Attempt to import config
try:
    import config
except ImportError:
    logging.error("config.py not found. Using placeholder settings.")
    PROCESSED_DIR = os.path.join("..", "data", "processed")
    RESULTS_DIR = os.path.join("..", "results")
else:
    PROCESSED_DIR = config.PROCESSED_DIR
    RESULTS_DIR = os.path.join(os.path.dirname(PROCESSED_DIR), "results")

# Configure logging
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "baseline_trainer.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Configuration ---
PROCESSED_FILE = os.path.join(PROCESSED_DIR, 'ftse_processed_features.parquet')
TRAIN_END_DATE = '2020-12-31'
VAL_END_DATE = '2022-12-31' # Used for splitting data if needed for tuning complexity
COVARIANCE_LOOKBACK = 60 # Days for empirical covariance calculation
RISK_FREE_RATE = 0.01 # Annualized
REBALANCE_FREQ = 'W-FRI' # Weekly on Friday
CV_SPLITS = 5 # Number of splits for TimeSeriesSplit

# Define feature columns (update based on actual columns after Sprint 1)
# Selected based on feature importance analysis
FEATURE_COLS = [
    'volatility_20',
    'sma_60',
    'sma_5',
    'log_return_lag_2',
    'log_return_lag_3',
    'log_return_lag_5',
    'quarter',
    'log_return_lag_1',
    'rsi_14',
]


# --- Helper Functions ---

def define_task_and_split(df: pd.DataFrame, feature_cols: List[str],
                          train_end: str, val_end: str) -> tuple:
    """Defines target, selects features, and splits data."""
    logging.info("Defining prediction task (T+1 log return) and splitting data...")
    # Only sort by index name if it exists and is a column/index level
    by = ['symbol']
    if df.index.name and df.index.name in df.index.names:
        by.append(df.index.name)
    df = df.sort_values(by=by)
    df['target'] = df.groupby('symbol')['log_return'].shift(-1)

    # Select valid features available before dropping target NaNs
    valid_features = [col for col in feature_cols if col in df.columns]
    df_task = df[['symbol', 'log_return'] + valid_features + ['target']].copy()
    df_task.dropna(subset=['target'], inplace=True) # Drop last day per symbol

    logging.info(f"Using {len(valid_features)} features. Task DataFrame shape: {df_task.shape}")

    train_df = df_task[df_task.index <= train_end]
    # Validation set can be used for tuning more complex models or final model training decision
    val_df = df_task[(df_task.index > train_end) & (df_task.index <= val_end)]
    test_df = df_task[df_task.index > val_end]

    X_train = train_df[valid_features]
    y_train = train_df['target']
    X_val = val_df[valid_features]
    y_val = val_df['target']
    X_test = test_df[valid_features]
    y_test = test_df['target']

    logging.info(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test, test_df


def scale_features(X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """Scales features using StandardScaler."""
    logging.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to preserve index/columns if needed downstream
    X_train_scaled = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, index=X_val.index, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    logging.info("Features scaled.")
    # Save scaler for potential use in FL client simulation
    scaler_path = os.path.join(RESULTS_DIR, 'baseline_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {scaler_path}")
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def tune_and_train_model(model_name: str, model_instance: Any, params: Dict,
                         X_train: pd.DataFrame, y_train: pd.DataFrame,
                         cv_splits: int = 5, use_random_search: bool = False, n_iter: int = 10) -> Any:
    """Tunes hyperparameters using TimeSeriesSplit and returns the best trained model."""
    logging.info(f"--- Tuning {model_name} ---")
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    if use_random_search:
        search = RandomizedSearchCV(model_instance, params, n_iter=n_iter, cv=tscv,
                                    scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=1)
    else:
        search = GridSearchCV(model_instance, params, cv=tscv,
                              scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

    search.fit(X_train, y_train) # Assumes X_train is scaled already
    best_params = search.best_params_
    best_score = -search.best_score_ # Score is negative MSE
    logging.info(f"Best {model_name} Params: {best_params}")
    logging.info(f"Best {model_name} CV Score (MSE): {best_score:.6f}")

    # Train final model on the entire training set with best parameters
    final_model = model_instance.set_params(**best_params)
    final_model.fit(X_train, y_train)
    logging.info(f"Final {model_name} trained on full training set.")
    return final_model


def evaluate_predictions(model_name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
    """Makes predictions and evaluates MSE and R2."""
    logging.info(f"Evaluating {model_name} predictions...")
    y_pred = model.predict(X_test) # Assumes X_test is scaled
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"{model_name} - Test MSE: {mse:.6f}, Test R2: {r2:.4f}")
    return pd.Series(y_pred, index=y_test.index, name=f"{model_name}_pred")


def run_portfolio_simulation(predictions: pd.DataFrame, actual_returns: pd.DataFrame,
                             lookback: int, risk_free: float, rebal_freq: str) -> Dict[str, pd.Series]:
    """
    Runs a portfolio backtest simulation using model predictions and PyPortfolioOpt.

    Args:
        predictions: DataFrame containing model predictions. Expected structure:
                     Index: DatetimeIndex (can have duplicate dates for multiple symbols)
                     Columns: 'symbol', 'actual' (optional), and prediction columns ending in '_pred'.
        actual_returns: DataFrame containing actual log returns and potentially features.
                        Must contain 'log_return' column and be convertible to a
                        MultiIndex of (datetime, symbol).
        lookback: Lookback window (in days) for calculating the covariance matrix.
        risk_free: Annualized risk-free rate for Sharpe ratio calculation.
        rebal_freq: Rebalancing frequency string (e.g., 'W-FRI', 'M', 'BM'). See pandas offset aliases.

    Returns:
        A dictionary where keys are model names and values are pandas Series
        of simulated portfolio log returns indexed by timestamp. Returns an empty
        dictionary if PyPortfolioOpt is not installed.
    """
    if not HAS_PYPFOpt:
        logging.error("PyPortfolioOpt not found. Portfolio simulation skipped.")
        return {}

    portfolio_returns_log = {}
    model_pred_cols = [col for col in predictions.columns if col.endswith('_pred')]
    if not model_pred_cols:
        logging.warning("No prediction columns found ending in '_pred'. Simulation cannot run.")
        return {}

    # --- Prepare Actual Returns DataFrame ---
    # Ensure actual returns have a (datetime, symbol) MultiIndex
    date_col_name = 'datetime_idx' # Default name, adjust if needed
    if not isinstance(actual_returns.index, pd.MultiIndex):
        potential_date_cols = ['datetime_idx', 'timestamp', 'date']
        found_date_col = None
        for col in potential_date_cols:
            if col in actual_returns.columns:
                found_date_col = col
                date_col_name = col # Store the actual date column name found
                break
        if found_date_col and 'symbol' in actual_returns.columns:
            logging.info(f"Setting MultiIndex on actual_returns using columns: '{found_date_col}', 'symbol'")
            actual_returns = actual_returns.set_index([found_date_col, 'symbol'])
        else:
            raise ValueError("actual_returns must have a MultiIndex or columns like ('datetime_idx'/'timestamp'/'date', 'symbol') to set one.")

    # Ensure first level of MultiIndex is datetime
    if not pd.api.types.is_datetime64_any_dtype(actual_returns.index.levels[0]):
        logging.info("Converting first level of actual_returns index to datetime.")
        actual_returns = actual_returns.set_index([
            pd.to_datetime(actual_returns.index.get_level_values(0)),
            actual_returns.index.get_level_values(1)
        ])
        # Use the detected or default date column name for the index level name
        actual_returns.index.set_names([date_col_name, 'symbol'], inplace=True)

    # Sort index for efficient slicing (crucial!)
    if not actual_returns.index.is_monotonic_increasing:
        logging.info("Sorting actual_returns index.")
        actual_returns = actual_returns.sort_index()

    # --- Pre-Simulation Asset Filtering: Drop assets with >10% missing returns ---
    asset_missing_frac = actual_returns.groupby('symbol')['log_return'].apply(lambda x: x.isna().mean())
    drop_threshold = 0.10
    assets_to_keep = asset_missing_frac[asset_missing_frac <= drop_threshold].index.tolist()
    dropped_assets = asset_missing_frac[asset_missing_frac > drop_threshold].index.tolist()
    if dropped_assets:
        logging.warning(f"Dropping {len(dropped_assets)} assets with >10% missing returns: {dropped_assets}")
    else:
        logging.info("No assets dropped for missing data pre-filtering.")
    actual_returns = actual_returns[actual_returns.index.get_level_values('symbol').isin(assets_to_keep)]
    logging.info(f"Assets remaining after pre-filter: {len(assets_to_keep)}")

    # --- Clean actual_returns: Drop rows with NaN log_return before simulation ---
    initial_row_count = actual_returns.shape[0]
    actual_returns_clean = actual_returns.dropna(subset=['log_return'])
    rows_dropped = initial_row_count - actual_returns_clean.shape[0]
    if rows_dropped > 0:
        logging.warning(f"Dropped {rows_dropped} rows from actual_returns due to NaN log_return before simulation.")
    else:
        logging.info("No NaN log_return values found in actual_returns before simulation.")
    # For diagnostics: log number of valid assets per date (first 5 dates)
    valid_assets_per_date = actual_returns_clean.groupby(level=0).apply(lambda df: df['log_return'].notna().sum())
    logging.info(f"Valid assets per date (first 5): {valid_assets_per_date.head().to_dict()}")
    actual_returns = actual_returns_clean

    # Get unique dates from the correctly formatted index
    unique_test_dates = actual_returns.index.get_level_values(0).unique()
    if not isinstance(unique_test_dates, pd.DatetimeIndex):
         unique_test_dates = pd.DatetimeIndex(unique_test_dates) # Ensure it's a DatetimeIndex

    if unique_test_dates.empty:
        logging.error("No unique dates found in actual_returns index. Cannot simulate.")
        return {}

    # Determine rebalancing dates
    rebalance_dates = pd.date_range(unique_test_dates.min(), unique_test_dates.max(), freq=rebal_freq)
    logging.info(f"Simulation period: {unique_test_dates.min().date()} to {unique_test_dates.max().date()}.")
    logging.info(f"Using rebalancing frequency '{rebal_freq}', resulting in {len(rebalance_dates)} potential rebalance dates.")

    # --- Simulation Loop ---
    for pred_col in model_pred_cols:
        model_name = pred_col.replace('_pred', '')
        logging.info(f"--- Simulating portfolio for model: {model_name} ---")
        daily_returns = []
        current_weights = None
        initialized = False # Flag to track if initial weights are set

        # Log first 5 predictions for diagnostics
        try:
            first_preds = predictions[["symbol", pred_col]].drop_duplicates("symbol").head(5)
            logging.info(f"First 5 predictions for {model_name}:\n{first_preds}")
        except Exception as e:
            logging.warning(f"Could not log first predictions for {model_name}: {e}")
        # Log first 5 predictions for diagnostics [remove logs]

        # Loop through trading days defined by actual_returns
        for i in range(len(unique_test_dates)):
            today = unique_test_dates[i]

            # Determine the next day IF it exists in our data
            next_day = unique_test_dates[i + 1] if i + 1 < len(unique_test_dates) else None

            rebalance_today = today in rebalance_dates

            if rebalance_today:
                logging.debug(f"Rebalance check for {today.date()}")
                if next_day is None:
                     logging.warning(f"At end of unique_test_dates ({today.date()}), cannot rebalance for a 'next_day'.")
                     continue # Skip rebalancing if there's no next day in the data

                # --- Rebalance Logic ---
                # 1. Get mu (expected returns = predictions for next day)
                try:
                    # Select all prediction rows for next_day. Handles potential duplicate dates in predictions index.
                    preds_df_next_day = predictions.loc[next_day]

                    # Process based on whether .loc returned a DataFrame or Series
                    if isinstance(preds_df_next_day, pd.DataFrame) and 'symbol' in preds_df_next_day.columns:
                        if preds_df_next_day.empty:
                             logging.warning(f"Prediction slice empty for {next_day.date()}. Holding.")
                             mu_series = None # Indicate failure to get predictions
                        else:
                             preds_indexed_by_symbol = preds_df_next_day.set_index('symbol')
                             mu_series = preds_indexed_by_symbol[pred_col]
                    elif isinstance(preds_df_next_day, pd.Series) and 'symbol' in preds_df_next_day.index and pred_col in preds_df_next_day.index:
                         # Handle edge case where .loc might return a Series (e.g., only 1 symbol on that day)
                         symbol_value = preds_df_next_day['symbol']
                         pred_value = preds_df_next_day[pred_col]
                         mu_series = pd.Series([pred_value], index=[symbol_value], name=pred_col)
                         mu_series.index.name = 'symbol'
                    else:
                        # Log error for unexpected structure
                        logging.error(f"Unexpected structure or missing 'symbol'/'{pred_col}' in prediction slice for {next_day.date()}. Type: {type(preds_df_next_day)}. Holding.")
                        mu_series = None # Indicate failure

                except KeyError:
                    logging.warning(f"Prediction data not found for date {next_day.date()}. Holding.")
                    mu_series = None # Indicate failure
                except Exception as e:
                     # Catch other potential errors during prediction access
                     logging.error(f"Error accessing prediction data for {next_day.date()}: {e}. Holding.")
                     mu_series = None # Indicate failure

                # Continue only if mu_series was successfully created
                if mu_series is not None and not mu_series.empty:
                    # 2. Get Sigma (covariance matrix using data up to 'today')
                    try:
                        returns_up_to_today = actual_returns.loc[pd.IndexSlice[:today, :], 'log_return'].unstack(level='symbol')
                    except Exception as e:
                        logging.error(f"Error creating returns_pivot on {today.date()}: {e}. Cannot rebalance.")
                        # Keep current_weights, will be used in next day's calculation
                        continue

                    if len(returns_up_to_today) >= lookback:
                        # Use only symbols present in mu prediction for covariance calculation
                        # Ensure alignment by filtering returns_up_to_today columns
                        common_symbols = returns_up_to_today.columns.intersection(mu_series.index)
                        if common_symbols.empty:
                            logging.warning(f"No common symbols between historical returns and predictions on {today.date()}. Holding.")
                            weights_to_use = current_weights
                        else:
                            returns_subset = returns_up_to_today[common_symbols].iloc[-lookback:]
                            # Instead of returns_subset.dropna(axis=1, how='any'), use forward fill then zero fill
                            returns_subset = returns_subset.fillna(method='ffill').fillna(0)
                            if returns_subset.shape[1] < 2: # Need at least 2 assets for meaningful covariance/optimization
                                 logging.warning(f"Not enough valid assets ({returns_subset.shape[1]}) after filtering NaNs for cov calc on {today.date()}. Holding.")
                                 weights_to_use = current_weights
                            else:
                                try:
                                    # S = risk_models.sample_cov(returns_subset, returns_data=True, frequency=252)
                                    # Use Ledoit-Wolf shrinkage for more stable estimates, especially with many assets
                                    S = risk_models.CovarianceShrinkage(returns_subset, returns_data=True, frequency=252).ledoit_wolf()

                                except Exception as cov_e:
                                    logging.warning(f"Covariance calculation failed on {today.date()}: {cov_e}. Holding.")
                                    weights_to_use = current_weights
                                else:
                                    # 3. Optimize Portfolio
                                    # Align mu and Sigma (use symbols valid in returns_subset)
                                    mu_aligned = mu_series.reindex(returns_subset.columns).dropna() # Ensure mu matches assets with valid returns

                                    if mu_aligned.empty or S.shape[0] != len(mu_aligned):
                                         logging.warning(f"mu_series empty or misaligned with valid covariance assets on {today.date()}. Holding.")
                                         weights_to_use = current_weights
                                    else:
                                        # Re-align S to match the final mu_aligned index
                                        S_aligned = S.reindex(index=mu_aligned.index, columns=mu_aligned.index)

                                        # Check for positive semi-definiteness (more robust than determinant check)
                                        # A small positive value is added for numerical stability if needed.
                                        min_eig = np.min(np.linalg.eigh(S_aligned)[0])
                                        if min_eig < 1e-8:
                                             logging.warning(f"Covariance matrix not positive semi-definite (min eigenvalue={min_eig:.2e}) on {today.date()}. Adding jitter. Holding.")
                                             # S_aligned += np.eye(S_aligned.shape[0]) * 1e-8 # Optional: Add jitter
                                             weights_to_use = current_weights # Safer to just hold weights
                                        else:
                                            # Define daily risk-free rate above optimization logic
                                            daily_rf = risk_free / 252
                                            solvers_to_try = ['OSQP', 'SCS']
                                            for solver in solvers_to_try:
                                                try:
                                                    ef = EfficientFrontier(mu_aligned, S_aligned, solver=solver, verbose=True)
                                                    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                                                    # Optimize for max Sharpe using daily risk-free rate
                                                    weights_dict = ef.max_sharpe(risk_free_rate=daily_rf)
                                                    weights_to_use = pd.Series(ef.clean_weights())
                                                    logging.info(f"Rebalanced weights successfully on {today.date()} for {model_name} using {solver}. Assets: {len(weights_to_use[weights_to_use > 0])}.")
                                                    logging.debug(f"Weights: {weights_to_use[weights_to_use > 0].to_dict()}")
                                                    break  # Success, exit solver loop
                                                except Exception as e:
                                                    logging.warning(f"{solver} solver failed on {today.date()} for {model_name}: {e}.")
                                            else:
                                                logging.warning(f"All solvers failed on {today.date()} for {model_name}. Holding weights.")
                                                weights_to_use = current_weights
                    else: # Not enough data for covariance
                        logging.warning(f"Not enough historical data ({len(returns_up_to_today)} days < {lookback}) for cov calc on {today.date()}. Holding.")
                        weights_to_use = current_weights
                else: # mu_series fetch failed
                    logging.warning(f"Failed to generate mu_series for {next_day.date()}. Holding weights.")
                    weights_to_use = current_weights

                # Update weights for the *next* period's calculation
                current_weights = weights_to_use

            # --- Initial Weight Setting (if first day and rebalance failed) ---
            if not initialized and current_weights is None:
                 logging.info("Setting initial portfolio weights to equal weight.")
                 # Use symbols available on the first day in actual_returns
                 initial_symbols = actual_returns.loc[unique_test_dates[0]].index.get_level_values('symbol').unique()
                 if not initial_symbols.empty:
                     n_assets = len(initial_symbols)
                     current_weights = pd.Series(1/n_assets, index=initial_symbols)
                     logging.info(f"Initialized with {n_assets} assets.")
                 else:
                     logging.error("Cannot set initial weights: No symbols found for the first date.")
                     # Cannot proceed without initial weights if first rebalance fails
                     return {} # Or handle differently
            initialized = True # Mark as initialized

            # --- Calculate Portfolio Return for 'next_day' using 'current_weights' ---
            # This calculation happens *after* potential rebalancing on 'today'
            # The weights decided on 'today' are applied to 'next_day' returns.
            if next_day is None: # Skip if we are at the last day
                 continue

            if current_weights is not None and not current_weights.empty:
                try:
                    # Get actual returns for the next day
                    next_day_rets = actual_returns.loc[pd.IndexSlice[next_day, :], 'log_return']
                    # If only one symbol returned, it might be a Series without symbol index
                    if isinstance(next_day_rets, pd.Series) and not isinstance(next_day_rets.index, pd.MultiIndex):
                         # Try to infer symbol if index is not MultiIndex (e.g., only one symbol exists for the day)
                         symbol = actual_returns.loc[pd.IndexSlice[next_day, :]].index.get_level_values('symbol')
                         if len(symbol) == 1:
                              next_day_rets_series = pd.Series(next_day_rets.iloc[0], index=[symbol[0]])
                         else: # Cannot safely map if multiple rows return unexpected Series
                              raise ValueError("Unexpected Series format for next_day_rets")
                    else:
                         # Should have MultiIndex, select level 'symbol'
                         next_day_rets_series = next_day_rets.droplevel(0) # Drop date level


                    # Align returns and weights (inner join handles missing data)
                    aligned_returns, aligned_weights = next_day_rets_series.align(current_weights, join='inner')

                    if aligned_returns.empty or aligned_weights.empty:
                         logging.warning(f"No common assets between returns and weights for {next_day.date()}. Portfolio return is 0.")
                         port_return = 0.0
                    else:
                         # Normalize weights if they don't sum to 1 after alignment (e.g., asset dropped)
                         aligned_weights /= aligned_weights.sum()
                         port_return = np.dot(aligned_returns, aligned_weights)

                    daily_returns.append({'timestamp': next_day, 'return': port_return})

                except KeyError:
                     logging.warning(f"Actual return data missing for {next_day.date()}. Skipping portfolio return calculation.")
                except Exception as calc_e:
                     logging.error(f"Error calculating portfolio return for {next_day.date()}: {calc_e}")
                     # Optionally append NaN or 0, or just skip the day
                     # daily_returns.append({'timestamp': next_day, 'return': np.nan})

        # --- Store Final Timeseries ---
        if daily_returns:
            portfolio_log = pd.DataFrame(daily_returns).set_index('timestamp')['return']
            # Ensure the Series has the correct model name
            portfolio_log.name = model_name
            portfolio_returns_log[model_name] = portfolio_log
            logging.info(f"--- Finished simulation for model: {model_name}. Portfolio log length: {len(portfolio_log)} ---")
        else:
            logging.warning(f"No daily returns were calculated for model: {model_name}.")
            portfolio_returns_log[model_name] = pd.Series(dtype=float, name=model_name) # Store empty series

    return portfolio_returns_log


def evaluate_portfolio(portfolio_returns: Dict[str, pd.Series], risk_free: float) -> Dict[str, Dict]:
    """Calculates Sharpe, VaR, CVaR, Max Drawdown using quantstats."""
    logging.info("\n--- Portfolio Performance Evaluation ---")
    results = {}
    if not HAS_QUANTSTATS:
        logging.warning("quantstats not found. Skipping advanced metrics.")
        return {"quantstats_missing": {}}

    for model_name, returns_series in portfolio_returns.items():
        if not returns_series.empty and returns_series.notna().any():
            # Quantstats assumes daily returns
            returns_series = returns_series.dropna() # Ensure no NaNs for quantstats
            sharpe = qs.stats.sharpe(returns_series, rf=risk_free, annualize=True)
            VaR = qs.stats.var(returns_series, confidence=0.95) # Typically returns negative value
            CVaR = qs.stats.cvar(returns_series, confidence=0.95) # Typically returns negative value
            max_drawdown = qs.stats.max_drawdown(returns_series)
            logging.info(f"Model: {model_name}")
            logging.info(f"  Annualized Sharpe Ratio: {sharpe:.4f}")
            logging.info(f"  Max Drawdown: {max_drawdown:.4f}")
            logging.info(f"  VaR (95%): {VaR:.4f}")
            logging.info(f"  CVaR (95%): {CVaR:.4f}")
            results[model_name] = {'Sharpe': sharpe, 'Max Drawdown': max_drawdown, 'VaR_95': VaR, 'CVaR_95': CVaR}
        else:
            logging.warning(f"No valid portfolio returns generated for model: {model_name}")
            results[model_name] = {'Sharpe': np.nan, 'Max Drawdown': np.nan, 'VaR_95': np.nan, 'CVaR_95': np.nan}
    return results


# --- Main Execution ---
def main():
    """Runs the baseline training and evaluation pipeline."""
    logging.info("--- Starting Baseline Evaluation Pipeline ---")
    # Load data
    try:
        df_full = pd.read_parquet(PROCESSED_FILE)
    except Exception as e:
        logging.error(f"Failed to load processed data from {PROCESSED_FILE}: {e}")
        return

    # Define task and split data
    X_train, y_train, X_val, y_val, X_test, y_test, test_df = define_task_and_split(
        df_full, FEATURE_COLS, TRAIN_END_DATE, VAL_END_DATE
    )

    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)

    # Tune and Train Models
    # Ridge
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 200.0]}
    final_ridge = tune_and_train_model("Ridge", Ridge(), ridge_params, X_train_scaled, y_train, CV_SPLITS)

    # RandomForest
    rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [5, 10]}
    final_rf = tune_and_train_model("RandomForest", RandomForestRegressor(random_state=42, n_jobs=-1), rf_params, X_train_scaled, y_train, CV_SPLITS, use_random_search=True, n_iter=6)

    # XGBoost
    final_xgb = None
    if xgb:
        xgb_params = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.7, 0.9], 'colsample_bytree': [0.7, 0.9]}
        final_xgb = tune_and_train_model("XGBoost", xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1), xgb_params, X_train_scaled, y_train, CV_SPLITS, use_random_search=True, n_iter=10)
    else:
        logging.info("Skipping XGBoost tuning and training.")

    # Evaluate Predictions on Validation Set
    val_pred_ridge = evaluate_predictions("Ridge", final_ridge, X_val_scaled, y_val)
    val_pred_rf = evaluate_predictions("RandomForest", final_rf, X_val_scaled, y_val) if final_rf else None
    val_pred_xgb = evaluate_predictions("XGBoost", final_xgb, X_val_scaled, y_val) if final_xgb else None
    # Log validation metrics
    val_mse_ridge = mean_squared_error(y_val, val_pred_ridge)
    val_r2_ridge = r2_score(y_val, val_pred_ridge)
    logging.info(f"Ridge - Validation MSE: {val_mse_ridge:.6f}, Validation R2: {val_r2_ridge:.4f}")
    best_model_name = "Ridge"
    best_model = final_ridge
    best_val_mse = val_mse_ridge
    if val_pred_rf is not None:
        val_mse_rf = mean_squared_error(y_val, val_pred_rf)
        val_r2_rf = r2_score(y_val, val_pred_rf)
        logging.info(f"RandomForest - Validation MSE: {val_mse_rf:.6f}, Validation R2: {val_r2_rf:.4f}")
        if val_mse_rf < best_val_mse:
            best_model_name = "RandomForest"
            best_model = final_rf
            best_val_mse = val_mse_rf
    if val_pred_xgb is not None:
        val_mse_xgb = mean_squared_error(y_val, val_pred_xgb)
        val_r2_xgb = r2_score(y_val, val_pred_xgb)
        logging.info(f"XGBoost - Validation MSE: {val_mse_xgb:.6f}, Validation R2: {val_r2_xgb:.4f}")
        if val_mse_xgb < best_val_mse:
            best_model_name = "XGBoost"
            best_model = final_xgb
            best_val_mse = val_mse_xgb
    logging.info(f"Selected best model based on validation MSE: {best_model_name}")

    # Evaluate Predictions (on test set) for best model only
    all_preds = test_df[['symbol']].copy()
    all_preds['actual'] = y_test.values
    pred_best = evaluate_predictions(best_model_name, best_model, X_test_scaled, y_test)
    all_preds[f'{best_model_name.lower()}_pred'] = pred_best.values

    # Save predictions
    preds_path = os.path.join(RESULTS_DIR, 'baseline_predictions.csv')
    all_preds.to_csv(preds_path)
    logging.info(f"Baseline predictions saved to {preds_path}")

    # Prepare actual_returns for portfolio simulation (robust to index/column)
    if 'datetime_idx' not in test_df.columns or 'symbol' not in test_df.columns:
        test_df_reset = test_df.reset_index()
    else:
        test_df_reset = test_df.copy()
    test_df_reset = test_df_reset.set_index(['datetime_idx', 'symbol'])

    # Run Portfolio Simulation for best model only
    portfolio_time_series = run_portfolio_simulation(
        all_preds, # Contains predictions and symbol column
        test_df_reset, # Now has MultiIndex (datetime_idx, symbol)
        lookback=COVARIANCE_LOOKBACK,
        risk_free=RISK_FREE_RATE,
        rebal_freq=REBALANCE_FREQ
    )
    # Evaluate Portfolio Performance
    portfolio_metrics = evaluate_portfolio(portfolio_time_series, RISK_FREE_RATE)

    # Save metrics
    metrics_path = os.path.join(RESULTS_DIR, 'baseline_portfolio_metrics.csv')
    if portfolio_metrics:
         pd.DataFrame.from_dict(portfolio_metrics, orient='index').to_csv(metrics_path)
         logging.info(f"Portfolio metrics saved to {metrics_path}")

    # Save final models
    model_path = os.path.join(RESULTS_DIR, f'baseline_{best_model_name.lower()}_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    logging.info(f"{best_model_name} model saved to {model_path}")

    logging.info("--- Baseline Evaluation Pipeline Finished ---")
    logging.info("----------------------------------------------------------------------------------------------")
    logging.info("----------------------------------------------------------------------------------------------")



if __name__ == "__main__":
    main()
