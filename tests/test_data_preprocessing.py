import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Adjust import path based on your project structure
from data_preprocessing import clean_data, calculate_log_returns

def test_clean_data_fills_nan():
    """Test that clean_data correctly fills NaNs using ffill then bfill per group."""
    data = {
        'adjusted_close': [100, np.nan, 102, 103, 200, 201, np.nan, 203],
        'volume': [1000, 1100, 1200, 1300, 2000, 2100, 2200, 2300],
        'symbol': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B']
    }
    # Use distinct dates to make timestamp sorting unambiguous
    idx = pd.to_datetime(['2023-01-01', '2023-01-03', '2023-01-05', '2023-01-07', # Symbol A
                          '2023-01-02', '2023-01-04', '2023-01-06', '2023-01-08']) # Symbol B
    test_df = pd.DataFrame(data, index=idx)
    # Add timestamp column from index *before* calling clean_data
    test_df['timestamp'] = test_df.index

    # Expected data *after* grouped ffill/bfill (order is symbol A then B due to sorting)
    # NaNs filled: A's NaN close becomes 100 (ffill). B's NaN close becomes 201 (ffill).
    expected_intermediate_data = {
        # Symbol A rows (sorted by time)
        'adjusted_close':  [100.0, 100.0, 102.0, 103.0,
        # Symbol B rows (sorted by time)
                   200.0, 201.0, 201.0, 203.0],
        'volume': [1000, 1100, 1200, 1300,
                   2000, 2100, 2200, 2300],
        'symbol': ['A', 'A', 'A', 'A',
                   'B', 'B', 'B', 'B'],
        'timestamp': pd.to_datetime([
                     '2023-01-01', '2023-01-03', '2023-01-05', '2023-01-07', # A
                     '2023-01-02', '2023-01-04', '2023-01-06', '2023-01-08'  # B
                     ])
    }
    # Create DF in the order it would be after grouped filling (sorted by symbol, then timestamp)
    expected_intermediate_df = pd.DataFrame(expected_intermediate_data)
    # Set the index to match the original index order for sorting later
    expected_intermediate_df.index = pd.to_datetime([
                                     '2023-01-01', '2023-01-03', '2023-01-05', '2023-01-07', # A
                                     '2023-01-02', '2023-01-04', '2023-01-06', '2023-01-08'  # B
                                     ])


    # The final expected result is sorted by 'timestamp'
    expected_df = expected_intermediate_df.sort_values(by='timestamp')

    # Run the function
    cleaned_df = clean_data(test_df) # No need for .copy() inside test if clean_data does it

    # Debug prints (optional)
    # print("\nCleaned DataFrame:")
    # print(cleaned_df)
    # print("\nExpected DataFrame:")
    # print(expected_df)
    # print("\nCleaned Dtypes:")
    # print(cleaned_df.dtypes)
    # print("\nExpected Dtypes:")
    # print(expected_df.dtypes)


    # Assert equality - check_dtype is important as ffill/bfill turn int columns with NaNs to float
    assert_frame_equal(cleaned_df, expected_df, check_dtype=True)


def test_calculate_log_returns_handles_groups():
    """Test log return calculation handles groups and drops first NaN."""
    data = {
        'adjusted_close': [100, 101, 102, 200, 198],
        'symbol': ['A', 'A', 'A', 'B', 'B']
    }
    idx = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03',
                          '2023-01-01', '2023-01-02'])
    test_df = pd.DataFrame(data, index=idx)
    test_df['timestamp'] = test_df.index

    # Expected results after calculation and dropping initial NaNs per group
    expected_returns = [np.log(101/100), np.log(102/101), np.log(198/200)]
    expected_idx = pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-02'])
    expected_df = pd.DataFrame({
         'adjusted_close': [101, 102, 198],
         'symbol': ['A', 'A', 'B'],
         'log_return': expected_returns,
         'timestamp': expected_idx
    }, index=expected_idx).sort_index()

    result_df = calculate_log_returns(test_df, price_col='adjusted_close')

    # Sort both for comparison as order might change slightly due to groupby
    # assert_frame_equal(result_df.sort_index(), expected_df.sort_index(), check_dtype=False)

    result_df = result_df[expected_df.columns]
    assert_frame_equal(result_df.sort_index(), expected_df.sort_index(), check_dtype=False)

    assert 'log_return' in result_df.columns
    assert result_df.notna().all().all() # Ensure no NaNs left after dropna