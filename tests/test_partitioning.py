# FILE: tests/test_partitioning.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
import os
import sys

# --- Add src directory to Python path ---
# This assumes your tests directory is parallel to your src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)

# Adjust import path based on your project structure
from partition_data import add_sector_info, partition_data_non_iid

# Mock config.GCIS for testing
MOCK_GCIS = {
    'SYM1': 'SectorA',
    'SYM2': 'SectorA',
    'SYM3': 'SectorB',
    'SYM4': 'SectorB',
    'SYM5': 'SectorC',
}

@pytest.fixture
def sample_processed_df():
    """Creates a small sample DataFrame similar to processed data."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    data = {
        'symbol': ['SYM1']*10 + ['SYM2']*10 + ['SYM3']*10 + ['SYM4']*10 + ['SYM5']*10,
        'log_return': np.random.randn(50) * 0.01,
        'feature1': np.random.rand(50),
    }
    # Create multiindex DataFrame
    symbols = data['symbol']
    index = pd.MultiIndex.from_product([dates, list(MOCK_GCIS.keys())], names=['timestamp', 'symbol_level'])
    # This part is tricky - need to match the structure post-loading
    # Let's assume df structure before adding sector: indexed by timestamp, symbol column exists
    index_flat = pd.date_range(start='2023-01-01', periods=50, freq='D') # Simplification
    df = pd.DataFrame({
         'symbol': data['symbol'],
         'log_return': data['log_return'],
         'feature1': data['feature1']
    }, index=index_flat)
    df.index.name = 'timestamp'
    return df

@patch('partition_data.SYMBOL_TO_SECTOR_MAP', MOCK_GCIS) # Patch the map used inside the module
def test_partition_distributions(sample_processed_df):
    """Tests if data is distributed and skew seems present."""
    num_clients = 5
    alpha = 0.1 # High skew
    np.random.seed(42) # for reproducibility of dirichlet and assignment

    # Add sector info using the patched map
    df_sectored = add_sector_info(sample_processed_df, MOCK_GCIS)

    client_datasets = partition_data_non_iid(df_sectored, num_clients, alpha)

    assert len(client_datasets) == num_clients
    total_rows = sum(len(df) for df in client_datasets.values())
    assert total_rows == len(df_sectored) # Check all rows are distributed

    # Check skew (qualitative): Clients should ideally have data dominated by few sectors
    high_skew_observed = False
    for client_id, client_df in client_datasets.items():
        if not client_df.empty:
            sector_dist = client_df['gics_sector'].value_counts(normalize=True)
            # Check if one sector dominates significantly (e.g., > 70% with alpha=0.1)
            if sector_dist.max() > 0.7:
                high_skew_observed = True
            print(f"Client {client_id} Sector Dist: {sector_dist.round(2).to_dict()}")
        else:
             print(f"Client {client_id} has no data.")

    # With alpha=0.1 and 5 clients/3 sectors, we expect high skew
    assert high_skew_observed, "Expected high sector concentration (skew) not observed"


def test_add_sector_info_adds_column_and_maps_correctly(sample_processed_df):
    """Test add_sector_info adds 'gics_sector' and maps symbols to sectors."""
    df = sample_processed_df.copy()
    df_out = add_sector_info(df, MOCK_GCIS)
    assert 'gics_sector' in df_out.columns
    # All mapped symbols should have correct sector
    for sym, sector in MOCK_GCIS.items():
        assert (df_out[df_out['symbol'] == sym]['gics_sector'] == sector).all()
    # No NaNs in gics_sector
    assert df_out['gics_sector'].isna().sum() == 0


def test_add_sector_info_drops_unmapped_symbols(sample_processed_df):
    """Test add_sector_info drops rows with symbols not in the mapping."""
    df = sample_processed_df.copy()
    # Add a symbol not in the map
    df.loc[df.index[0], 'symbol'] = 'UNMAPPED'
    df_out = add_sector_info(df, MOCK_GCIS)
    assert 'UNMAPPED' not in df_out['symbol'].values
    # Should have fewer rows than input
    assert len(df_out) <= len(df)


def test_partition_data_non_iid_errors_on_missing_sector():
    """Test partition_data_non_iid raises ValueError if 'gics_sector' missing."""
    df = pd.DataFrame({'symbol': ['SYM1'], 'feature1': [1.0]})
    with pytest.raises(ValueError):
        partition_data_non_iid(df, num_clients=2, alpha=0.5)


def test_partition_data_non_iid_balanced_distribution():
    """Test partition_data_non_iid with high alpha yields more balanced splits."""
    # Small DataFrame with 3 sectors, 3 clients
    df = pd.DataFrame({
        'symbol': ['SYM1']*10 + ['SYM2']*10 + ['SYM3']*10,
        'gics_sector': ['SectorA']*10 + ['SectorB']*10 + ['SectorC']*10,
        'feature1': np.random.rand(30)
    })
    num_clients = 3
    alpha = 10.0 # High alpha for balanced
    np.random.seed(123)
    client_datasets = partition_data_non_iid(df, num_clients, alpha)
    # All clients should have data from all sectors
    for client_df in client_datasets.values():
        assert set(client_df['gics_sector']) == {'SectorA', 'SectorB', 'SectorC'}
    # Distribution should be roughly even (allow min > 0.05)
    for client_df in client_datasets.values():
        sector_counts = client_df['gics_sector'].value_counts(normalize=True)
        assert sector_counts.min() > 0.05


def test_partition_data_non_iid_total_rows_preserved(sample_processed_df):
    """Test that partitioning preserves total number of rows."""
    df = add_sector_info(sample_processed_df, MOCK_GCIS)
    num_clients = 4
    alpha = 0.5
    np.random.seed(42)
    client_datasets = partition_data_non_iid(df, num_clients, alpha)
    total_rows = sum(len(df) for df in client_datasets.values())
    assert total_rows == len(df)