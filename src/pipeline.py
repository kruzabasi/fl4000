import pandas as pd
import numpy as np
from typing import List
import os
import logging
from data_preprocessing import load_raw_data, check_missing_data, clean_data, calculate_log_returns, check_anomalies
from feature_engineering import add_moving_averages, add_volatility, add_calendar_features, add_lags, add_momentum_indicators, add_volume_indicators 
from config import SYMBOLS, DATA_DIR, PROCESSED_DIR

# Configure logging
logging.basicConfig(
    filename= os.path.join("logs", "pipeline.log"),
    level= logging.INFO,
    format= "%(asctime)s %(levelname)s: %(message)s"
)

RAW_DIR = DATA_DIR
PROCESSED_DIR = PROCESSED_DIR
OUTPUT_FILENAME = 'ftse_processed_features.parquet'


# Feature Engineering Parameters
SMA_WINDOWS = [5, 20, 60]
VOL_WINDOWS = [20]
RSI_WINDOW = 14
LAG_COLS = ['log_return', 'volume', 'close'] # Add more features to lag as needed
LAG_PERIODS = [1, 2, 3, 5, 10] # Example lag periods

# --- Pipeline Functions ---

def run_data_pipeline(symbols: List[str], raw_dir: str, processed_dir: str, output_filename: str) -> None:
    """
    Executes the full data acquisition (loading assumed done), preprocessing,
    and feature engineering pipeline.

    Args:
        symbols (List[str]): List of stock symbols to process.
        raw_dir (str): Directory containing raw data CSVs.
        processed_dir (str): Directory to save the processed data.
        output_filename (str): Filename for the saved processed data.
    """
    try:
        # 1. Load Raw Data
        logging.info("--- Starting Data Pipeline ---")
        df_raw = load_raw_data(raw_dir=raw_dir, symbols=symbols)

        # 2. Initial Checks
        check_missing_data(df_raw)

        # 3. Clean Data
        df_cleaned = clean_data(df_raw)

        # 4. Calculate Returns
        df_returns = calculate_log_returns(df_cleaned, price_col='close')

        # 5. Initial Anomaly Check (Volume)
        check_anomalies(df_returns, return_col=None, volume_col='volume') # Check volume only first

        # --- Prepare for Feature Engineering ---
        logging.info("Preparing DataFrame index for feature engineering.")
        # Ensure timestamp is datetime
        try:
            df_returns['timestamp'] = pd.to_datetime(df_returns['timestamp'])
            # Set timestamp as index (keep the column too if needed later, using drop=False)
            df_returns = df_returns.set_index('timestamp', drop=False)
            df_returns.index.name = 'datetime_idx' # Or None, or any name except 'timestamp'
            logging.info("Successfully set DatetimeIndex.")
        except KeyError:
            logging.error("Failed to set DatetimeIndex: 'timestamp' column not found.")
            # Decide how to handle - raise error or skip features? Raising is safer.
            raise ValueError("Pipeline cannot proceed without a 'timestamp' column to set as index.")
        except Exception as e:
            logging.error(f"Error converting/setting timestamp index: {e}")
            raise # Re-raise the exception

        # 6. Feature Engineering
        logging.info("--- Starting Feature Engineering ---")
        df_features = add_moving_averages(df_returns, windows=SMA_WINDOWS)
        df_features = add_volatility(df_features, windows=VOL_WINDOWS)
        df_features = add_calendar_features(df_features)
        df_features = add_lags(df_features, lag_cols=LAG_COLS, lags=LAG_PERIODS)
        df_features = add_momentum_indicators(df_features, rsi_window=RSI_WINDOW)
        df_features = add_volume_indicators(df_features)
        logging.info("--- Finished Feature Engineering ---")

        # 7. Drop NaNs from Features/Lags
        initial_rows = df_features.shape[0]
        df_final = df_features.dropna()
        rows_dropped = initial_rows - df_final.shape[0]
        if rows_dropped > 0:
            logging.info(f"Dropped {rows_dropped} rows due to NaNs from feature engineering (rolling/lags).")
        else:
            logging.info("No NaNs found after feature engineering.")


        # 8. Final Anomaly Check (including Returns)
        check_anomalies(df_final, return_col='log_return', volume_col='volume')

        # 9. Save Processed Data
        os.makedirs(processed_dir, exist_ok=True)
        output_path = os.path.join(processed_dir, output_filename)
        try:
            df_final.to_parquet(output_path, index=True)
            logging.info(f"Successfully saved processed data to {output_path}")
            logging.info(f"Final DataFrame shape: {df_final.shape}")
        except Exception as e:
            logging.error(f"Failed to save data to Parquet: {e}. Saving to CSV as fallback.")
            try:
                df_final.to_csv(output_path.replace('.parquet', '.csv'), index=True)
                logging.info(f"Successfully saved processed data to {output_path.replace('.parquet', '.csv')}")
            except Exception as ce:
                logging.error(f"Failed to save data to CSV: {ce}")


        logging.info("--- Data Pipeline Finished ---")

    except Exception as e:
        logging.exception(f"Data pipeline failed: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    run_data_pipeline(
        symbols=SYMBOLS,
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        output_filename=OUTPUT_FILENAME
    )