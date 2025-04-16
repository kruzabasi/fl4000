import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Adjust import path based on your project structure
from feature_engineering import add_moving_averages, add_lags, add_calendar_features

@pytest.fixture
def sample_timeseries_df():
    """Provides a sample DataFrame for testing."""
    data = {
        'close': [100, 101, 102, 103, 104, 200, 198, 196, 194, 192],
        'log_return': np.log([np.nan, 101/100, 102/101, 103/102, 104/103, np.nan, 198/200, 196/198, 194/196, 192/194]),
        'symbol': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
    }
    idx = pd.to_datetime([
        '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05',
        '2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'
    ])
    return pd.DataFrame(data, index=idx)

def test_add_moving_averages_calculates_correctly(sample_timeseries_df):
    """Test SMA calculation, handling groups and min_periods."""
    window = 3
    result_df = add_moving_averages(sample_timeseries_df.copy(), windows=[window])

    # Manually calculate expected SMA for symbol 'A'
    expected_sma_A = [np.nan, np.nan, (100+101+102)/3, (101+102+103)/3, (102+103+104)/3]
    # Manually calculate expected SMA for symbol 'B'
    expected_sma_B = [np.nan, np.nan, (200+198+196)/3, (198+196+194)/3, (196+194+192)/3]

    sma_col = f'sma_{window}'
    assert sma_col in result_df.columns
    # Check NaN values at the beginning due to min_periods
    assert result_df[result_df['symbol'] == 'A'][sma_col].iloc[:window-1].isna().all()
    assert result_df[result_df['symbol'] == 'B'][sma_col].iloc[:window-1].isna().all()
    # Check calculated values
    np.testing.assert_array_almost_equal(
        result_df[result_df['symbol'] == 'A'][sma_col].iloc[window-1:].values,
        expected_sma_A[window-1:],
        decimal=5
    )
    np.testing.assert_array_almost_equal(
        result_df[result_df['symbol'] == 'B'][sma_col].iloc[window-1:].values,
        expected_sma_B[window-1:],
        decimal=5
    )

def test_add_lags_shifts_correctly(sample_timeseries_df):
    """Test lagging handles groups correctly."""
    lag = 1
    col = 'log_return'
    result_df = add_lags(sample_timeseries_df.copy(), lag_cols=[col], lags=[lag])

    lag_col_name = f'{col}_lag_{lag}'
    assert lag_col_name in result_df.columns

    # Check first element of each group is NaN
    assert pd.isna(result_df[result_df['symbol'] == 'A'][lag_col_name].iloc[0])
    assert pd.isna(result_df[result_df['symbol'] == 'B'][lag_col_name].iloc[0])

    # Check shifted values within group 'A'
    np.testing.assert_array_almost_equal(
        result_df[result_df['symbol'] == 'A'][lag_col_name].iloc[1:].values,
        sample_timeseries_df[sample_timeseries_df['symbol'] == 'A'][col].iloc[:-1].values,
        decimal=5
    )
     # Check shifted values within group 'B'
    np.testing.assert_array_almost_equal(
        result_df[result_df['symbol'] == 'B'][lag_col_name].iloc[1:].values,
        sample_timeseries_df[sample_timeseries_df['symbol'] == 'B'][col].iloc[:-1].values,
        decimal=5
    )

def test_add_calendar_features(sample_timeseries_df):
    """Test calendar features are added correctly."""
    result_df = add_calendar_features(sample_timeseries_df.copy())
    assert 'day_of_week' in result_df.columns
    assert 'month' in result_df.columns
    assert 'quarter' in result_df.columns
    # Check a known value, e.g., 2023-01-01 was a Sunday (dayofweek=6)
    assert result_df.loc['2023-01-01', 'day_of_week'].iloc[0] == 6 # Assuming first row is 2023-01-01
    assert result_df.loc['2023-01-01', 'month'].iloc[0] == 1
    assert result_df.loc['2023-01-01', 'quarter'].iloc[0] == 1