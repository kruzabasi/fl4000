# Snippets for use within a Jupyter Notebook (e.g., notebooks/eda.ipynb)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.stattools import adfuller # Ensure statsmodels is imported
import os
import sys

# Add src directory to path to import config (adjust relative path if needed)
src_path = os.path.abspath(os.path.join('../', 'src'))
if src_path not in sys.path:
    sys.path.append(src_path)

try:
    import config # Import config variables
except ImportError:
    print("Could not import config.py. Ensure it's accessible and define paths manually.")
    # Define paths manually as fallback
    PROCESSED_DIR = './data/processed' # Adjusted relative path
    OUTPUT_FILENAME = 'ftse_processed_features.parquet'
    # Define a default symbol if config.SYMBOLS is not available
    DEFAULT_SYMBOL = 'AZN.LON' # Example fallback
else:
    PROCESSED_DIR = config.PROCESSED_DIR
    OUTPUT_FILENAME = 'ftse_processed_features.parquet' # Assuming this filename is used
    DEFAULT_SYMBOL = config.SYMBOLS[0] if config.SYMBOLS else 'AZN.LON'


# --- Load Processed Data ---
processed_file = os.path.join(PROCESSED_DIR, OUTPUT_FILENAME)
print(f"Attempting to load processed data from: {processed_file}")

if os.path.exists(processed_file):
    try:
        df_features = pd.read_parquet(processed_file)
        print("Loaded processed data. Shape:", df_features.shape)
        print("Columns:", df_features.columns)
        # Display first few rows and info
        print("\nDataFrame Head:")
        print(df_features.head())
        print("\nDataFrame Info:")
        df_features.info()
    except Exception as e:
        print(f"Error loading parquet file: {e}. Trying CSV as fallback.")
        try:
            csv_file = processed_file.replace('.parquet', '.csv')
            if os.path.exists(csv_file):
                 df_features = pd.read_csv(csv_file, index_col=0, parse_dates=True)
                 print("Loaded processed data from CSV. Shape:", df_features.shape)
            else:
                 print("CSV file also not found.")
                 df_features = pd.DataFrame() # Empty df
        except Exception as ce:
             print(f"Error loading CSV file: {ce}")
             df_features = pd.DataFrame() # Empty df

else:
    print(f"Error: Processed data file not found at {processed_file}. Run the pipeline first.")
    df_features = pd.DataFrame() # Empty df for subsequent cells not to fail immediately

# --- Visualize Time Series (Example Symbol) ---
symbol_to_plot = DEFAULT_SYMBOL
if not df_features.empty and 'symbol' in df_features.columns and symbol_to_plot in df_features['symbol'].unique():
    df_symbol = df_features[df_features['symbol'] == symbol_to_plot]

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    df_symbol['adjusted_close'].plot(title=f'{symbol_to_plot} Adjusted Close')
    plt.ylabel("Price")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    df_symbol['log_return'].plot(title=f'{symbol_to_plot} Log Returns')
    plt.ylabel("Log Return")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print(f"Cannot plot time series for {symbol_to_plot}. Data missing or empty.")

# --- Distribution Analysis (Log Returns - All Stocks) ---
if not df_features.empty and 'log_return' in df_features.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df_features['log_return'].dropna(), kde=True, bins=100)
    plt.title('Distribution of Log Returns (All Stocks)')
    plt.xlabel('Log Return')
    plt.grid(axis='y')
    plt.show()
else:
    print("Cannot plot log return distribution.")

# --- Distribution Analysis (Example Features) ---
features_to_plot = ['volatility_20', 'sma_20', 'rsi_14'] # Add features as needed
if not df_features.empty:
    for feature in features_to_plot:
        if feature in df_features.columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(df_features[feature].dropna(), kde=True, bins=100)
            plt.title(f'Distribution of {feature} (All Stocks)')
            plt.xlabel(feature)
            plt.grid(axis='y')
            plt.show()
        else:
            print(f"Feature '{feature}' not found in DataFrame.")
else:
    print("DataFrame empty, cannot plot feature distributions.")


# --- Correlation Analysis ---
if not df_features.empty:
    numeric_df = df_features.select_dtypes(include=np.number)
    # Optionally drop highly collinear columns if needed before modeling
    # Exclude lags of the same base variable if desired, or keep for analysis
    correlation_matrix = numeric_df.corr()

    plt.figure(figsize=(18, 15)) # Adjust size as needed
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
     print("Cannot calculate correlation matrix.")

# --- Stationarity Test (Example: Log Returns for one symbol) ---
if not df_features.empty and 'log_return' in df_features.columns and symbol_to_plot in df_features['symbol'].unique():
    sample_returns = df_features[df_features['symbol'] == symbol_to_plot]['log_return'].dropna()
    if not sample_returns.empty:
        print(f"\n--- ADF Test for Stationarity ({symbol_to_plot} Log Returns) ---")
        result = adfuller(sample_returns)
        print(f'ADF Statistic: {result[0]:.4f}')
        print(f'p-value: {result[1]:.4f}')
        print('Critical Values:')
        for key, value in result[4].items():
            print(f'\t{key}: {value:.4f}')
        if result[1] <= 0.05:
            print("Result: Reject the null hypothesis (Likely Stationary)")
        else:
            print("Result: Fail to reject the null hypothesis (Likely Non-Stationary)")
    else:
        print(f"No log returns found for {symbol_to_plot} to perform ADF test.")
else:
     print("Cannot perform ADF test on log returns.")