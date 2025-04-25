# Data Pipeline Documentation

## 1. Data Source
- **Provider:** Alpha Vantage API
- **Universe:** FTSE 100 constituent stocks (sample)
- **Symbols:** e.g., SHEL.LON, BP.LON, HSBA.LON, etc. (see `config.py` for sample list)
- **Date Range:** Full available history per symbol (via `TIME_SERIES_DAILY_ADJUSTED`, `OUTPUT_SIZE=full`)
- **Format:** Daily OHLCV (Open, High, Low, Close, Volume) data, CSV

## 2. Data Acquisition
- **Script:** `src/data_acquisition.py`
- **Key Steps:**
    1. **API Request:** For each symbol, download daily price data using Alpha Vantage API.
    2. **Rate Limit Handling:** Detects and retries on rate limit errors using exponential backoff.
    3. **Logging:** All actions and errors are logged to `logs/ingest.log`.
    4. **Validation:** After download, checks each CSV for completeness and absence of rate limit messages.
    5. **Output:** Raw CSV files saved to `data/raw/` (one per symbol).

## 3. Preprocessing
- **Script:** `src/data_preprocessing.py`
- **Key Steps:**
    1. **Loading:** Reads all valid CSVs from `data/raw/` into a combined DataFrame.
    2. **Column Standardization:** Ensures columns are lowercase and consistent.
    3. **Missing Value Handling:**
        - Forward and backward fill (within each symbol group) for all price/volume columns.
        - Drops rows with unresolved NaNs after filling.
    4. **Symbol Column:** Ensures every row has a `symbol` identifier.
    5. **Sorting:** Data is sorted by `symbol` and `timestamp`.

## 4. Return Calculation
- **Method:** Logarithmic returns
- **Column:** `log_return`
- **Function:** `calculate_log_returns` in `src/data_preprocessing.py`
- **Formula:**
    ```python
    log_return = np.log(price / price.shift(1))
    ```
    - Calculated per symbol using the `close` price column.

## 5. Feature Engineering
- **Script:** `src/feature_engineering.py`
- **Key Steps:** Adds technical indicators and derived features to the dataset.

### Feature List
| Feature Name                      | Description / Calculation                                     | Source Function                |
|-----------------------------------|---------------------------------------------------------------|-------------------------------|
| `close`                           | Adjusted closing price (from raw data)                        | Acquisition/Preprocessing      |
| `log_return`                      | Logarithmic return (see Section 4)                            | `calculate_log_returns` (in `data_preprocessing.py`) |
| `sma_5`, `sma_20`, `sma_60`       | Simple Moving Average (5, 20, 60 days)                        | `add_moving_averages`          |
| `volatility_20`                   | Rolling standard deviation of returns (20 days)               | `add_volatility`               |
| `rsi_14`                          | Relative Strength Index (14 days)                             | `add_momentum_indicators`      |
| `macd`, `macd_signal`, `macd_diff`| MACD (12-26 EMA), signal line (9 EMA of MACD), difference     | `add_momentum_indicators`      |
| `obv`                             | On-Balance Volume                                             | `add_volume_indicators`        |
| `volume`                          | Daily trading volume                                          | Acquisition/Preprocessing      |
| `day_of_week`, `month`, `quarter` | Calendar features derived from timestamp                      | `add_calendar_features`        |
| `log_return_lag_X`, etc.          | Lagged versions of specified features (e.g., 1, 2, 3, 5 days) | `add_lags`                     |

*Note: See `src/feature_engineering.py` and `src/pipeline.py` for specific parameters and the full list of engineered features.*

## 6. Final Dataset Structure
- **Output File:** `data/processed/ftse_processed_features.parquet` (or `.csv` fallback)
- **Columns:**
    - `timestamp` (datetime64): Date of observation
    - `symbol` (string): Stock symbol
    - `open`, `high`, `low`, `close` (float): Price columns
    - `volume` (int/float): Daily trading volume
    - `log_return` (float): Daily log return
    - Technical indicators (float): e.g., `sma_20`, `volatility_20`, `rsi_14`, `macd`, `obv`, `log_return_lag_1`, etc.
- **Index:** `timestamp` is set as the DatetimeIndex of the DataFrame.
- **Data Types:**
    - `timestamp` (column): datetime64[ns]
    - `symbol`: object (string)
    - All numeric features: float64 (except `volume`, which may be int64 or float64)

---

**References:**
- See `src/pipeline.py` for orchestration of all steps.
- See `src/feature_engineering.py` for detailed feature calculations.
- See `src/data_preprocessing.py` for cleaning and return calculation logic.
- See `src/data_acquisition.py` for download and validation logic.
