import pandas as pd
import os
import numpy as np
import logging
import ta
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    filename=os.path.join("logs", "feature_engineering.log"), 
    format="%(asctime)s %(levelname)s: %(message)s"
    )

def add_moving_averages(df: pd.DataFrame, windows: List[int] = [5, 20, 60], price_col: str = 'adjusted_close') -> pd.DataFrame:
    """
    Adds Simple Moving Averages (SMA) for specified windows, grouped by symbol.

    Args:
        df (pd.DataFrame): Input DataFrame.
        windows (List[int]): List of window sizes for SMA calculation.
        price_col (str): The price column to calculate SMA on.

    Returns:
        pd.DataFrame: DataFrame with added SMA columns (e.g., 'sma_5').
    """
    if price_col not in df.columns:
        logging.error(f"Price column '{price_col}' not found for SMA calculation.")
        return df
    if 'symbol' not in df.columns:
        logging.error("Missing 'symbol' column for grouping. Calculating SMA globally.")
        for window in windows:
            df[f'sma_{window}'] = df[price_col].rolling(window=window, min_periods=window).mean()
    else:
        # Ensure timestamp column exists for sorting
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        df = df.sort_values(by=['symbol', 'timestamp'])
        for window in windows:
            df[f'sma_{window}'] = df.groupby('symbol', group_keys=False)[price_col].transform(
                lambda x: x.rolling(window=window, min_periods=window).mean()
            )
    logging.info(f"Added SMAs for windows: {windows}")
    return df

def add_volatility(df: pd.DataFrame, windows: List[int] = [20], return_col: str = 'log_return') -> pd.DataFrame:
    """
    Adds rolling standard deviation of log returns (volatility), grouped by symbol.

    Args:
        df (pd.DataFrame): Input DataFrame.
        windows (List[int]): List of window sizes for volatility calculation.
        return_col (str): The return column to calculate volatility on.

    Returns:
        pd.DataFrame: DataFrame with added volatility columns (e.g., 'volatility_20').
    """
    if return_col not in df.columns:
        logging.error(f"Return column '{return_col}' not found for volatility calculation.")
        return df
    if 'symbol' not in df.columns:
        logging.error("Missing 'symbol' column for grouping. Calculating volatility globally.")
        for window in windows:
             df[f'volatility_{window}'] = df[return_col].rolling(window=window, min_periods=window).std()
    else:
        # Ensure timestamp column exists for sorting
        sort_col = None # Initialize sort_col
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            # Use 'timestamp' if the index is a DatetimeIndex, otherwise use the default name 'index'
            index_name = 'timestamp' if isinstance(df.index, pd.DatetimeIndex) else 'index'
            df = df.reset_index().rename(columns={'index': index_name})
            sort_col = index_name
        elif 'timestamp' in df.columns:
            sort_col = 'timestamp'
        elif isinstance(df.index, pd.MultiIndex):
            # If MultiIndex, assume 'timestamp' is one of the levels or handle appropriately
            # For now, let's assume 'timestamp' exists as a column if not in index
            # This part might need adjustment based on actual MultiIndex structure
            if 'timestamp' in df.columns:
                sort_col = 'timestamp'
            else:
                # Attempt to find a time-based level name if index name is None
                time_level_name = df.index.names[-1] # Heuristic: assume last level is time
                if time_level_name:
                    sort_col = time_level_name
                else: # Fallback if no suitable column/index level found
                    logging.warning("Could not determine time column for sorting in add_volatility. Sorting by symbol only.")
                    df = df.sort_values(by=['symbol'])
                    # sort_col remains None
        else:
            # Fallback if index is not DatetimeIndex and no 'timestamp' column
            logging.warning("Index is not DatetimeIndex and 'timestamp' column not found. Cannot sort by time in add_volatility.")
            # sort_col remains None

        # Sort only if a valid time column was identified
        if sort_col:
            df = df.sort_values(by=['symbol', sort_col])

        # Calculate volatility after sorting (or without time sort if sort_col is None)
        for window in windows:
            df[f'volatility_{window}'] = df.groupby('symbol', group_keys=False)[return_col].transform(
                lambda x: x.rolling(window=window, min_periods=window).std()
            )
    logging.info(f"Added Volatility for windows: {windows}")
    return df

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar-based features derived from the DataFrame's DatetimeIndex.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: DataFrame with added 'day_of_week', 'month', 'quarter'.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        logging.error("DataFrame index is not a DatetimeIndex. Cannot add calendar features.")
        return df
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    logging.info("Added calendar features (day_of_week, month, quarter).")
    return df

def add_lags(df: pd.DataFrame, lag_cols: List[str] = ['log_return'], lags: List[int] = [1, 2, 3, 5]) -> pd.DataFrame:
    """
    Adds lagged versions of specified columns, grouped by symbol.

    Args:
        df (pd.DataFrame): Input DataFrame.
        lag_cols (List[str]): List of column names to create lags for.
        lags (List[int]): List of lag periods (number of steps to shift).

    Returns:
        pd.DataFrame: DataFrame with added lag columns (e.g., 'log_return_lag_1').
    """
    missing_lag_cols = [col for col in lag_cols if col not in df.columns]
    if missing_lag_cols:
        logging.error(f"Missing columns for lag calculation: {missing_lag_cols}")
        return df
    if 'symbol' not in df.columns:
        logging.error("Missing 'symbol' column for grouping. Calculating lags globally.")
        for col in lag_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    else:
        # Ensure timestamp column exists for sorting
        if 'timestamp' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={'index': 'timestamp'})
        df = df.sort_values(by=['symbol', 'timestamp'])
        for col in lag_cols:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df.groupby('symbol', group_keys=False)[col].shift(lag)
    logging.info(f"Added lags {lags} for columns: {lag_cols}")
    return df

def add_momentum_indicators(df: pd.DataFrame, rsi_window: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> pd.DataFrame:
    """
    Adds momentum indicators such as RSI and MACD to the DataFrame using the 'ta' library.
    Args:
        df (pd.DataFrame): Input DataFrame, must contain 'adjusted_close' column.
        rsi_window (int): Window size for RSI calculation.
        macd_fast (int): Fast EMA period for MACD.
        macd_slow (int): Slow EMA period for MACD.
        macd_signal (int): Signal EMA period for MACD.
    Returns:
        pd.DataFrame: DataFrame with new columns: 'rsi', 'macd', 'macd_signal', 'macd_diff'.
    """
    if 'adjusted_close' not in df.columns:
        raise ValueError("'adjusted_close' column required for momentum indicators.")
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['adjusted_close'], window=rsi_window, fillna=True).rsi()
    macd = ta.trend.MACD(close=df['adjusted_close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal, fillna=True)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    logging.info(f"Added RSI (window={rsi_window}) and MACD indicators.")
    return df

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds volume-based indicators such as On-Balance Volume (OBV) to the DataFrame using the 'ta' library.
    Args:
        df (pd.DataFrame): Input DataFrame, must contain 'adjusted_close' and 'volume' columns.
    Returns:
        pd.DataFrame: DataFrame with new column: 'obv'.
    """
    if 'adjusted_close' not in df.columns or 'volume' not in df.columns:
        raise ValueError("'adjusted_close' and 'volume' columns required for OBV indicator.")
    df = df.copy()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['adjusted_close'], volume=df['volume'], fillna=True).on_balance_volume()
    logging.info("Added OBV indicator.")
    return df
