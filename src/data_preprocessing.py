import pandas as pd
import numpy as np
import os
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
    into a single DataFrame.

    Args:
        raw_dir (str): The directory containing the raw CSV files.
        symbols (Optional[List[str]]): A list of specific symbols to load.
                                         If None, attempts to load all .csv files.

    Returns:
        pd.DataFrame: A combined DataFrame with data for all loaded symbols,
                      indexed by timestamp and sorted.

    Raises:
        ValueError: If no data is loaded.
    """
    logging.info(f"Loading raw data from: {raw_dir}")
    all_dfs = []
    if symbols is None:
        try:
            symbols = [f.split('.')[0] for f in os.listdir(raw_dir) if f.endswith('.csv')]
            logging.info(f"No specific symbols provided, found {len(symbols)} CSV files.")
        except FileNotFoundError:
             logging.error(f"Raw data directory not found: {raw_dir}")
             raise
    else:
         logging.info(f"Loading specific symbols: {symbols}")


    for symbol in symbols:
        # Assuming filenames match symbols (e.g., AZN.LON.csv or just AZN.csv)
        # Try both formats if needed
        potential_filenames = [f"{symbol}.csv", f"{symbol.split('.')[0]}.csv"]
        file_path = None
        for fname in potential_filenames:
            p = os.path.join(raw_dir, fname)
            if os.path.exists(p):
                file_path = p
                break

        if file_path and os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
                # Ensure 'symbol' column exists or add it
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol
                all_dfs.append(df)
                logging.debug(f"Loaded data for {symbol}")
            except Exception as e:
                logging.warning(f"Error loading {symbol} from {file_path}: {e}")
        else:
            logging.warning(f"File not found for symbol: {symbol} in {raw_dir}")

    if not all_dfs:
        raise ValueError("No data loaded. Check raw data directory and symbol list.")

    combined = pd.concat(all_dfs).sort_index()
    logging.info(f"Combined raw data shape: {combined.shape}")

    # Standardize column names (lowercase, replace space with underscore)
    combined.columns = [str(col).lower().replace(' ', '_') for col in combined.columns]

    # Ensure essential columns exist after standardization
    expected_cols = ['symbol', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
    missing_cols = [col for col in expected_cols if col not in combined.columns]
    if missing_cols:
        logging.warning(f"Combined DataFrame missing essential columns: {missing_cols}")

    # Keep only necessary columns if desired (optional, uncomment if needed)
    # cols_to_keep = [col for col in expected_cols if col in combined.columns]
    # combined = combined[cols_to_keep]

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
    then backward fill within each symbol group.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame, sorted by timestamp.
    """
    logging.info("Cleaning data (forward fill then backward fill)...")
    if 'symbol' not in df.columns:
        logging.error("Missing 'symbol' column for cleaning. Applying fill globally.")
        df_filled = df.ffill().bfill()
    else:
        # Ensure data is sorted by symbol then time for correct filling
        df_sorted = df.sort_values(by=['symbol', df.index.name])
        logging.debug("Applying forward fill per symbol...")
        df_filled = df_sorted.groupby('symbol', group_keys=False).ffill()
        logging.debug("Applying backward fill per symbol...")
        df_filled = df_filled.groupby('symbol', group_keys=False).bfill()

    if df_filled.isnull().values.any():
        initial_rows = df.shape[0]
        df_filled = df_filled.dropna()
        final_rows = df.shape[0]
        logging.warning(f"NaNs remained after ffill/bfill. Dropped {initial_rows - final_rows} rows.")
    else:
        logging.info("NaN filling complete, no remaining NaNs found.")

    # Resort by date index after cleaning
    return df_filled.sort_index()

def calculate_log_returns(df: pd.DataFrame, price_col: str = 'adjusted_close') -> pd.DataFrame:
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
        df = df.sort_values(by=['symbol', df.index.name]) # Ensure correct order
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