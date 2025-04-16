import pandas as pd
import numpy as np
import os
import glob
import logging
from config import DATA_DIR
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    filename=os.path.join("logs", "preprocess.log"), 
    format="%(asctime)s %(levelname)s: %(message)s"
    )

def load_raw_data(raw_dir: str = DATA_DIR, symbols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Loads individual raw stock data CSVs from a directory and combines them
    into a single DataFrame. Skips files that are malformed or missing required columns.
    Args:
        raw_dir (str): The directory containing the raw CSV files.
        symbols (Optional[List[str]]): A list of specific symbols to load.
                                       If None, attempts to load all .csv files.
    Returns:
        pd.DataFrame: A combined DataFrame with data for all loaded symbols,
                      indexed by timestamp and sorted.
    Raises:
        ValueError: If no valid data files are found.
    """

    all_files = glob.glob(os.path.join(raw_dir, '*.csv'))
    if symbols is not None:
        all_files = [f for f in all_files if os.path.splitext(os.path.basename(f))[0] in symbols]

    dfs = []
    required_cols = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    for file in all_files:
        try:
            df = pd.read_csv(file)
            if not required_cols.issubset(df.columns):
                logging.warning(f"Skipping {file}: missing required columns {required_cols - set(df.columns)}.")
                continue
            # Add symbol column if not present
            if 'symbol' not in df.columns:
                symbol = os.path.splitext(os.path.basename(file))[0]
                df['symbol'] = symbol
            dfs.append(df)
        except Exception as e:
            logging.warning(f"Skipping {file}: {e}")
            continue

    if not dfs:
        logging.error("No valid data files loaded. Returning empty DataFrame.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # Ensure essential columns exist after standardization
    expected_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in expected_cols if col not in combined.columns]
    if missing_cols:
        logging.warning(f"Combined DataFrame missing essential columns: {missing_cols}")
    logging.info("Finished loading raw data.")
    return combined

def check_missing_data(df: pd.DataFrame) -> None:
    """
    Checks and logs information about missing values in the DataFrame,
    grouped by symbol.

    Args:
        df (pd.DataFrame): Input DataFrame with stock data.
    """
    logging.info("Checking for missing data...")
    if 'symbol' not in df.columns:
        logging.error("Missing 'symbol' column for grouping.")
        print(f"Total missing values:\n{df.isnull().sum()}")
        return

    missing_summary = df.groupby('symbol').apply(lambda x: x.isnull().sum())
    missing_per_symbol = missing_summary[missing_summary.sum(axis=1) > 0]

    if not missing_per_symbol.empty:
        logging.warning("Missing Values Summary (per symbol):\n" + missing_per_symbol.to_string())
    else:
        logging.info("No missing values found per symbol.")

    total_missing = df.isnull().sum()
    total_missing_filtered = total_missing[total_missing > 0]
    if not total_missing_filtered.empty:
         logging.warning(f"\nTotal missing values per column:\n{total_missing_filtered.to_string()}")
    else:
         logging.info("No missing values found in total.")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by handling missing values using forward fill,
    then backward fill within each symbol group. Applies filling only
    to columns other than 'symbol' and 'timestamp'.

    Args:
        df (pd.DataFrame): Input DataFrame. Must contain 'symbol' and
                           'timestamp' columns (or a DatetimeIndex that
                           will be converted to 'timestamp').
    Returns:
        pd.DataFrame: Cleaned DataFrame, sorted by timestamp.
    """
    logging.info("Cleaning data (forward fill then backward fill per symbol)...")
    if df.empty:
        logging.warning("Input DataFrame is empty. Returning empty DataFrame.")
        return df

    # --- Input Validation ---
    if 'symbol' not in df.columns:
        logging.error(f"Missing 'symbol' column. Columns present: {df.columns.tolist()}")
        # Decide error handling: raise error or return unmodified/partially processed df?
        # Raising an error is often better to prevent silent failures downstream.
        raise ValueError("Input DataFrame must contain a 'symbol' column.")

    # Ensure 'timestamp' exists, converting index if necessary
    if 'timestamp' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            logging.debug("Converting DatetimeIndex to 'timestamp' column.")
            # Avoid modifying the original DataFrame passed to the function
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        else:
            raise ValueError("Input DataFrame must have a 'timestamp' column or a DatetimeIndex.")
    # --- End Validation ---

    # Make a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # Sort by symbol and timestamp is crucial for correct ffill/bfill within groups
    logging.debug("Sorting by symbol and timestamp.")
    df_processed = df_processed.sort_values(by=['symbol', 'timestamp'])

    # Identify columns to apply filling to (exclude grouping keys)
    cols_to_fill = df_processed.columns.difference(['symbol', 'timestamp'])

    if cols_to_fill.empty:
        logging.warning("No columns found to fill (excluding 'symbol', 'timestamp').")
    else:
        logging.info(f"Applying forward fill then backward fill to columns: {cols_to_fill.tolist()}")

        # Apply the filling function to each group
        df_processed = (
            df_processed.groupby('symbol', group_keys=False)[cols_to_fill]
            .apply(lambda g: g.ffill().bfill())
        )
        # Re-attach the grouping columns and timestamp
        df_processed = df_processed.join(df[['symbol', 'timestamp']])

        # --- Check for remaining NaNs ---
        # This check should focus on the columns that were *supposed* to be filled
        if df_processed[cols_to_fill].isnull().values.any():
            initial_rows = len(df_processed)
            # Drop rows where *any* of the filled columns still have NaN
            # This happens if an entire group had NaNs at the start/end for a column
            df_processed = df_processed.dropna(subset=cols_to_fill)
            final_rows = len(df_processed)
            rows_dropped = initial_rows - final_rows
            if rows_dropped > 0:
                 logging.warning(f"NaNs remained in columns {cols_to_fill.tolist()} after ffill/bfill. Dropped {rows_dropped} rows.")
        else:
            logging.info("NaN filling complete, no remaining NaNs found in processed columns.")

    logging.debug("Sorting final DataFrame by timestamp.")
    df_final = df_processed.sort_values(by='timestamp')

    return df_final

def calculate_log_returns(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    """
    Calculates the logarithmic returns for the specified price column,
    grouped by symbol. Drops rows where return cannot be calculated (first row per symbol).
    Args:
        df (pd.DataFrame): Input DataFrame, must contain 'symbol' column and price_col.
        price_col (str): The name of the column containing the prices to use.
    Returns:
        pd.DataFrame: DataFrame with added 'log_return' column.
    """
    logging.info(f"Calculating log returns based on column: {price_col}")
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")
    if 'symbol' not in df.columns:
        logging.error("Missing 'symbol' column for grouping. Calculating returns globally.")
        df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
    else:
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        df = df.sort_values(by=['symbol', 'timestamp'])
        df['log_return'] = df.groupby('symbol')[price_col].transform(
            lambda x: np.log(x / x.shift(1))
        )
    initial_rows = df.shape[0]
    df = df.dropna(subset=['log_return']) # Drop rows with NaN returns
    rows_dropped = initial_rows - df.shape[0]
    if rows_dropped > 0:
         logging.info(f"Dropped {rows_dropped} rows due to NaN log returns (first observation per symbol).")
    return df

def check_anomalies(df: pd.DataFrame, return_col: Optional[str] = 'log_return', volume_col: str = 'volume', return_threshold: float = 0.5) -> None:
    """
    Performs basic anomaly checks for zero volume and extreme returns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        return_col (Optional[str]): Name of the return column to check. If None, check is skipped.
        volume_col (str): Name of the volume column.
        return_threshold (float): The absolute threshold for extreme returns.
    """
    logging.info("Checking for anomalies...")
    # Check for zero volume
    if volume_col in df.columns:
        zero_volume_days = df[df[volume_col] <= 0]
        if not zero_volume_days.empty:
            logging.warning(f"Found {len(zero_volume_days)} instances with zero or negative volume.")
    else:
        logging.warning(f"Volume column '{volume_col}' not found. Skipping volume check.")

    # Check for extreme returns
    if return_col and return_col in df.columns:
        extreme_returns = df[df[return_col].abs() > return_threshold]
        if not extreme_returns.empty:
            logging.warning(f"Found {len(extreme_returns)} instances with abs({return_col}) > {return_threshold:.2f}")
    elif return_col:
        logging.warning(f"Return column '{return_col}' not found. Skipping extreme return check.")