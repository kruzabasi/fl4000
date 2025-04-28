import pandas as pd
import numpy as np
import os

def main():
    # Canonized feature and target list
    canon_features = [
        'adjusted_close', 'close', 'dividend_amount', 'high', 'low', 'open', 'split_coefficient', 'volume',
        'sma_5', 'sma_20', 'sma_60', 'volatility_20', 'day_of_week', 'month', 'quarter',
        'log_return_lag_1', 'log_return_lag_2', 'log_return_lag_3', 'log_return_lag_5', 'log_return_lag_10',
        'volume_lag_1', 'volume_lag_2', 'volume_lag_3', 'volume_lag_5', 'volume_lag_10',
        'adjusted_close_lag_1', 'adjusted_close_lag_2', 'adjusted_close_lag_3', 'adjusted_close_lag_5', 'adjusted_close_lag_10',
        'rsi', 'macd', 'macd_signal', 'macd_diff', 'obv'
    ]
    target_col = 'log_return'
    id_cols = ['symbol', 'timestamp']

    input_path = 'data/processed/ftse_processed_features.parquet'
    output_path = 'data/processed/central_validation_set.parquet'
    output_csv = 'data/processed/central_validation_set.csv'

    df = pd.read_parquet(input_path)
    # Time-based split: last 10% by timestamp per symbol
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['symbol', 'timestamp'])
    val_indices = []
    for symbol, group in df.groupby('symbol'):
        n_val = max(1, int(0.1 * len(group)))
        val_indices.extend(group.index[-n_val:])
    val_df = df.loc[val_indices].copy()
    # Only keep canon features + target + id
    keep_cols = canon_features + [target_col] + id_cols
    keep_cols = [c for c in keep_cols if c in val_df.columns]
    val_df = val_df[keep_cols]
    val_df = val_df.dropna()
    val_df.to_parquet(output_path)
    val_df.to_csv(output_csv, index=False)
    print(f"Validation set saved to {output_path} and {output_csv}. Shape: {val_df.shape}")

if __name__ == "__main__":
    main()
