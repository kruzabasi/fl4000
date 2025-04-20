import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path  # Use pathlib for cleaner path handling
from typing import List, Optional, Dict

# Attempt to import config, handle potential errors
try:
    import config
    # Make sure config.GCIS exists and is a dictionary
    if not hasattr(config, 'GCIS') or not isinstance(config.GCIS, dict):
        logging.warning("config.GCIS not found or not a dictionary. Using placeholder.")
        # Define a placeholder if config doesn't provide GCIS
        SYMBOL_TO_SECTOR_MAP = {symbol: f"Sector_{(i % 10) + 1}" for i, symbol in enumerate(config.SYMBOLS)}
    else:
        SYMBOL_TO_SECTOR_MAP = config.GCIS
except ImportError:
    logging.error("config.py not found. Using placeholder settings.")
    # Define placeholder paths and symbols if config import fails
    PROCESSED_DIR = os.path.join("..", "data", "processed") # Adjust path relative to src
    SYMBOLS = ['AZN.LON', 'HSBA.LON', 'BP.LON'] # Example
    SYMBOL_TO_SECTOR_MAP = {symbol: f"Sector_{(i % 3) + 1}" for i, symbol in enumerate(SYMBOLS)}
    FEDERATED_DATA_DIR = os.path.join("..", "data", "federated")
else:
    # Use paths from config if import succeeded
    PROCESSED_DIR = config.PROCESSED_DIR
    FEDERATED_DATA_DIR = os.path.join(os.path.dirname(PROCESSED_DIR), "federated") # Place federated next to processed

# --- Setup Logging ---

# Configure logging
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "partition_data.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# Partitioning Parameters
NUM_CLIENTS = 40
ALPHA = 0.5 # Dirichlet concentration parameter for non-IID skew
RANDOM_SEED = 42 # For reproducibility

np.random.seed(RANDOM_SEED)

def add_sector_info(df: pd.DataFrame, symbol_sector_map: Dict[str, str]) -> pd.DataFrame:
    """Adds GICS sector information based on the symbol."""
    logging.info("Mapping symbols to GICS sectors...")
    df['gics_sector'] = df['symbol'].map(symbol_sector_map)
    original_rows = len(df)
    rows_before_drop = len(df)
    df.dropna(subset=['gics_sector'], inplace=True)
    rows_after_drop = len(df)
    if rows_after_drop < rows_before_drop:
        dropped_symbols = set(df[df['gics_sector'].isnull()]['symbol'].unique()) # Find symbols that caused drop
        logging.warning(f"Dropped {rows_before_drop - rows_after_drop} rows due to missing sector information for symbols: {dropped_symbols}")
    logging.info("Sector information added.")
    return df

def partition_data_non_iid(df: pd.DataFrame, num_clients: int, alpha: float) -> Dict[int, pd.DataFrame]:
    """
    Partition FTSE 100 data into non-IID client datasets using Dirichlet-distributed sector proportions.

    This function simulates realistic cross-silo federated learning (FL) scenarios, as described in Methodology Section 3.4.3. Data is sorted by GICS sector, and each client receives a unique mixture of sectors, with the degree of specialization controlled by the Dirichlet concentration parameter (ALPHA).

    Key Steps and Logic:
    - Data points are grouped by GICS sector ('gics_sector' column).
    - For each client, a probability vector over sectors is drawn from a Dirichlet distribution with parameter `alpha`.
    - Lower `alpha` values yield higher skew: clients receive most data from a few sectors (specialization), while higher `alpha` values result in more uniform distributions.
    - For each sector, data indices are split among clients according to these proportions, ensuring that:
        * Most data for each client comes from a small number of sectors (e.g., 80% primary, 20% random, as in Section 3.4.3).
        * All data is assigned, and each client receives a DataFrame.
    - Quantity skew is introduced by the Dirichlet sampling, so some clients may receive more or fewer samples.
    - No client-side balancing is applied (see Shaheen et al., 2024 for further methods).

    Args:
        df (pd.DataFrame): Input DataFrame with features and a 'gics_sector' column.
        num_clients (int): Number of federated clients to partition data into (e.g., 40).
        alpha (float): Dirichlet concentration parameter. Lower values increase sector skew per client.

    Returns:
        Dict[int, pd.DataFrame]: Mapping from client ID to their non-IID data partition.

    References:
        - Zhao et al., 2018; Li et al., 2020 (label/quantity skew via Dirichlet)
        - Methodology Section 3.4.3 (Non-IID Data Simulation)
        - Shaheen et al., 2024 (data balancing, not implemented)
    """
    if 'gics_sector' not in df.columns:
        raise ValueError("DataFrame must include 'gics_sector' column for partitioning.")

    sectors = df['gics_sector'].unique()
    num_sectors = len(sectors)
    logging.info(f"Starting non-IID partitioning for {num_clients} clients across {num_sectors} sectors using alpha={alpha}.")

    # Generate client proportions over sectors
    client_proportions = np.random.dirichlet([alpha] * num_sectors, num_clients)

    # Create sector to integer index mapping
    sector_to_idx = {sector: i for i, sector in enumerate(sectors)}

    # Get DataFrame indices grouped by sector
    sector_indices = {
        sector: df.index[df['gics_sector'] == sector].tolist()
        for sector in sectors
    }

    # Dictionary to hold indices assigned to each client
    client_data_indices = {i: [] for i in range(num_clients)}
    logging.info("Assigning data indices to clients...")

    for sector, indices in sector_indices.items():
        if not indices:
            logging.warning(f"No data found for sector '{sector}'. Skipping.")
            continue

        sector_idx = sector_to_idx[sector]
        num_sector_data = len(indices)
        proportions_for_sector = client_proportions[:, sector_idx]

        # Normalize proportions for safety, though Dirichlet should sum to 1
        proportions_for_sector = np.maximum(proportions_for_sector, 1e-10) # Avoid zero probabilities
        proportions_for_sector /= proportions_for_sector.sum()

        # Calculate target counts per client for this sector
        target_counts = (proportions_for_sector * num_sector_data).astype(int) # Initial integer counts

        # Adjust counts to match total number of indices exactly
        diff = num_sector_data - target_counts.sum()
        if diff != 0:
            adjustment_indices = np.random.choice(num_clients, abs(diff), p=proportions_for_sector)
            np.add.at(target_counts, adjustment_indices, np.sign(diff))

        # Final check for exact sum
        while target_counts.sum() != num_sector_data:
             logging.warning(f"Correcting count mismatch for sector {sector}. Diff: {num_sector_data - target_counts.sum()}")
             if target_counts.sum() > num_sector_data:
                  decrement_idx = np.random.choice(np.where(target_counts > 0)[0])
                  target_counts[decrement_idx] -= 1
             else:
                  increment_idx = np.random.choice(num_clients)
                  target_counts[increment_idx] += 1


        assert target_counts.sum() == num_sector_data, f"Count mismatch for sector {sector}"

        # Shuffle indices within the sector before assigning
        np.random.shuffle(indices)

        # Assign indices based on calculated counts
        current_pos = 0
        for client_id in range(num_clients):
            count = target_counts[client_id]
            assigned_indices = indices[current_pos : current_pos + count]
            client_data_indices[client_id].extend(assigned_indices)
            current_pos += count
        logging.debug(f"Assigned {num_sector_data} indices for sector '{sector}' across clients.")

    logging.info("Finished assigning indices.")

    # Create client DataFrames
    client_dfs = {}
    total_client_rows = 0
    client_sector_counts = {i: {} for i in range(num_clients)}
    for client_id in range(num_clients):
        indices = client_data_indices[client_id]
        if not indices:
            logging.warning(f"Client {client_id} has no assigned data points.")
            client_dfs[client_id] = pd.DataFrame(columns=df.columns) # Empty DF
            continue

        client_df = df.loc[indices].sort_index()
        client_dfs[client_id] = client_df
        total_client_rows += len(client_df)
        client_sector_counts[client_id] = client_df['gics_sector'].value_counts().to_dict()

    logging.info(f"Partitioned data for {num_clients} clients. Total rows distributed: {total_client_rows}")

    # Log distribution stats (optional)
    client_stats_df = pd.DataFrame.from_dict(client_sector_counts, orient='index').fillna(0).astype(int)
    logging.info("\nClient Sector Row Counts:\n" + client_stats_df.to_string())
    # You might save this stats_df to a file as well

    return client_dfs


def save_client_data(client_dfs: Dict[int, pd.DataFrame], base_dir: str) -> None:
    """Saves each client's DataFrame to a dedicated directory."""
    logging.info(f"Saving partitioned data to base directory: {base_dir}")
    for client_id, client_df in client_dfs.items():
        client_dir = os.path.join(base_dir, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)
        output_path = os.path.join(client_dir, "client_data.parquet")
        try:
            if not client_df.empty:
                client_df.to_parquet(output_path)
                logging.debug(f"Saved data for client {client_id} to {output_path}")
            else:
                 logging.warning(f"Skipping save for client {client_id} due to empty DataFrame.")
        except Exception as e:
            logging.error(f"Failed to save data for client {client_id}: {e}")
    logging.info("Finished saving client data.")

def main():
    """Main function to run the partitioning process."""
    processed_file = os.path.join(PROCESSED_DIR, 'ftse_processed_features.parquet')
    logging.info(f"Loading processed data from: {processed_file}")
    try:
        df_processed = pd.read_parquet(processed_file)
        logging.info(f"Loaded data shape: {df_processed.shape}")
    except Exception as e:
        logging.error(f"Failed to load processed data: {e}")
        return

    # Add sector info using the map from config
    df_sectored = add_sector_info(df_processed, SYMBOL_TO_SECTOR_MAP)

    if df_sectored.empty or 'gics_sector' not in df_sectored.columns or df_sectored['gics_sector'].isnull().all():
        logging.error("Sector information missing or empty after mapping. Cannot partition.")
        return

    # Partition data
    client_datasets = partition_data_non_iid(df_sectored, NUM_CLIENTS, ALPHA)

    # Save partitions
    save_client_data(client_datasets, FEDERATED_DATA_DIR)


if __name__ == "__main__":
    main()