import requests
import time
import os
import logging
from pathlib import Path  # Use pathlib for cleaner path handling
from typing import List, Tuple, Optional, Dict, Any  # Add type hints

# --- Configuration (Assuming these are imported from config.py) ---
from config import (
    ALPHA_VANTAGE_API_KEY, BASE_URL, FUNCTION, OUTPUT_SIZE,
    DATATYPE, SYMBOLS, SLEEP_DURATION, DATA_DIR
)

# --- Constants ---
MAX_RETRIES = 5
MAX_BACKOFF_SECONDS = 60
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "ingest.log"
# Ensure DATA_DIR is a Path object and points to the 'raw' subdirectory
RAW_DATA_DIR = (Path(DATA_DIR) / "raw"
                if not Path(DATA_DIR).name == "raw"
                else Path(DATA_DIR))

# --- Setup Logging ---
LOG_DIR.mkdir(exist_ok=True)  # Create log directory if it doesn't exist
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# --- Helper Functions ---

def log_and_print(message: str, level: int = logging.INFO):
    """Logs a message and prints it to the console."""
    if level == logging.ERROR:
        logging.error(message)
        print(f"ERROR: {message}")
    elif level == logging.WARNING:
        logging.warning(message)
        print(f"WARNING: {message}")
    else:
        logging.info(message)
        print(message)

def exponential_backoff(attempt: int) -> float:
    """Calculate exponential backoff time."""
    wait_time = (2 ** attempt) + (time.time() % 1) # Add jitter
    return min(wait_time, MAX_BACKOFF_SECONDS)

def is_rate_limit_content(content: str) -> bool:
    """
    Check if the file content indicates an Alpha Vantage rate limit or
    common informational message returned instead of data.
    """
    content_lower = content.lower() # Check case-insensitively

    # Check 1: Previous "Information" style message (often JSON)
    is_information_style = ("information" in content_lower and
                            "api key" in content_lower and
                            "rate limit" in content_lower)

    # Check 2: "Thank you" style message (often plain text or simple JSON value)
    is_thank_you_style = "thank you for using alpha vantage" in content_lower

    # Check 3: "Note" style message (JSON format as reported by user)
    # Check specifically for the JSON key "note": and keywords within the value
    is_note_style = ('"note":' in content_lower and # Look for the JSON key "note":
                     # "api key" in content_lower and
                     "rate limit" in content_lower and
                     "requests per day" in content_lower) # Be more specific

    # Check 4: "Premium endpoint" message (sometimes returned for free tier keys)
    is_premium_style = "premium endpoint" in content_lower

    # Check 5: Simple rate limit message (sometimes just plain text)
    is_simple_rate_limit = "our standard api call frequency is" in content_lower

    # Return True if *any* known rate limit or info pattern is found
    return (is_information_style or
            is_thank_you_style or
            is_note_style or
            is_premium_style or
            is_simple_rate_limit)

def get_api_params(symbol: str) -> Dict[str, Any]:
    """Construct the parameters dictionary for the API request."""
    return {
        "function": FUNCTION,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": OUTPUT_SIZE,
        "datatype": DATATYPE
    }

# --- Core Functions ---

def download_data_single(symbol: str, session: requests.Session) -> Optional[str]:
    """
    Downloads data for a single symbol using a requests Session.

    Args:
        symbol: The stock symbol.
        session: The requests.Session object to use.

    Returns:
        "rate_limit" if a rate limit message is detected in the response.
        "success" if the download is successful.
        "failed" if download fails after retries (non-rate-limit error).
        None if an unexpected error occurs preventing file write.
    """
    params = get_api_params(symbol)
    file_path = RAW_DATA_DIR / f"{symbol}.csv"

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(BASE_URL, params=params)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            response_text = response.text

            # Check for rate limit *before* writing
            if is_rate_limit_content(response_text):
                log_and_print(f"Rate limit message detected for {symbol}. Response snippet: '{response_text[:100].strip().replace(os.linesep,' ')}'", level=logging.WARNING)
                # Don't write the rate limit message to the file
                return "rate_limit"

            # Write successful data
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(response_text)

            log_and_print(f"Successfully downloaded data for {symbol} to {file_path}")
            return "success"

        except requests.exceptions.HTTPError as http_err:
            log_and_print(f"HTTP error for {symbol} (Attempt {attempt + 1}/{MAX_RETRIES}): {http_err}", level=logging.ERROR)
            # Specific check for common rate-limit related status codes (though AV often uses 200)
            if response.status_code in [429, 503]:
                 log_and_print(f"Status code {response.status_code} suggests potential rate limit or server issue for {symbol}.", level=logging.WARNING)
                 # Treat as potentially recoverable, proceed to backoff

        except requests.exceptions.RequestException as req_err:
            # Catch broader connection errors, timeouts, etc.
            log_and_print(f"Request error for {symbol} (Attempt {attempt + 1}/{MAX_RETRIES}): {req_err}", level=logging.ERROR)

        except IOError as io_err:
             log_and_print(f"File write error for {symbol} to {file_path}: {io_err}", level=logging.ERROR)
             return None # File system error, likely not recoverable by retrying download

        except Exception as e:
            # Catch any other unexpected errors during download/processing
            log_and_print(f"Unexpected error downloading data for {symbol} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}", level=logging.ERROR)

        # If loop continues, it means an error occurred and we should retry
        if attempt < MAX_RETRIES - 1:
            sleep_time = exponential_backoff(attempt)
            log_and_print(f"Retrying {symbol} in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        else:
             log_and_print(f"Max retries reached for {symbol}. Download failed.", level=logging.ERROR)
             return "failed" # Failed after all retries

    return "failed" # Should technically be unreachable if loop completes, but acts as safeguard

def download_data_batch(symbols: List[str]) -> Tuple[List[str], List[str]]:
    """
    Download data for a list of symbols, stopping if rate limit is hit.

    Args:
        symbols: A list of stock symbols to download.

    Returns:
        A tuple containing two lists: (completed_symbols, unsuccessful_symbols)
    """
    completed = []
    unsuccessful = list(symbols) # Start assuming all are unsuccessful
    rate_limit_hit = False

    # Use a session object for potential connection reuse
    with requests.Session() as session:
        for symbol in symbols:
            log_and_print(f"--- Attempting download for {symbol} ---")
            result = download_data_single(symbol, session)

            if result == "success":
                completed.append(symbol)
                if symbol in unsuccessful: unsuccessful.remove(symbol)
            elif result == "rate_limit":
                log_and_print(f"Stopping batch download due to rate limit hit on {symbol}.", level=logging.ERROR)
                rate_limit_hit = True
                # Don't remove the current symbol from unsuccessful
                break # Stop processing more symbols in this batch
            else: # result == "failed" or None
                # Symbol already in unsuccessful, do nothing extra
                 log_and_print(f"Download failed for {symbol} after retries.", level=logging.WARNING)

            # Optional: Add a small fixed sleep even on success to respect API limits
            time.sleep(SLEEP_DURATION)

    total_attempted = len(completed) + (len(unsuccessful) if rate_limit_hit else len(symbols))
    log_and_print(f"Batch download summary: Attempted: {total_attempted}, Succeeded: {len(completed)}, Failed/Skipped: {len(unsuccessful)}")

    return completed, unsuccessful


def redownload_failed_files() -> Tuple[List[str], List[str]]:
    """
    Check files in the raw data directory. If a file contains a rate limit
    message, attempt to redownload it. Stop further attempts if a rate limit
    is hit during redownload.

    Returns:
        A tuple containing two lists:
        (symbols_successfully_redownloaded, symbols_failed_or_skipped_redownload)
    """
    redownload_completed = []
    redownload_failed = []
    files_to_check = list(RAW_DATA_DIR.glob("*.csv")) # Get all CSV files
    symbols_to_retry = []

    log_and_print(f"--- Starting check for files needing redownload in {RAW_DATA_DIR} ---")

    # First pass: Identify files containing the rate limit message
    for file_path in files_to_check:
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read()
                if is_rate_limit_content(content):
                    symbol = file_path.stem # Get filename without extension
                    symbols_to_retry.append(symbol)
                    log_and_print(f"Identified {symbol} ({file_path.name}) for redownload attempt.")
        except Exception as e:
            log_and_print(f"Error reading {file_path}: {e}", level=logging.ERROR)
            redownload_failed.append(file_path.stem) # Count as failed if unreadable

    if not symbols_to_retry:
        log_and_print("No files found with rate limit messages requiring redownload.")
        return [], []

    log_and_print(f"--- Attempting redownload for {len(symbols_to_retry)} identified symbols ---")

    rate_limit_hit = False
    with requests.Session() as session:
        for symbol in symbols_to_retry:
            if rate_limit_hit:
                log_and_print(f"Skipping redownload attempt for {symbol} due to earlier rate limit hit.")
                redownload_failed.append(symbol)
                continue

            log_and_print(f"Attempting redownload for {symbol}...")
            result = download_data_single(symbol, session)

            if result == "success":
                redownload_completed.append(symbol)
            elif result == "rate_limit":
                log_and_print(f"Rate limit hit during redownload of {symbol}. Stopping further redownload attempts.", level=logging.ERROR)
                redownload_failed.append(symbol)
                rate_limit_hit = True # Stop processing more symbols
            else: # result == "failed" or None
                 log_and_print(f"Redownload failed for {symbol} after retries.", level=logging.WARNING)
                 redownload_failed.append(symbol)

            # Optional: Add sleep between redownload attempts
            time.sleep(SLEEP_DURATION)

    log_and_print(f"Redownload process summary: Attempted: {len(symbols_to_retry)}, Succeeded: {len(redownload_completed)}, Failed/Skipped: {len(redownload_failed)}")

    return redownload_completed, redownload_failed

def final_validation() -> Tuple[List[str], List[str]]:
    """
    Performs a final check on all CSV files in the raw data directory
    to ensure none contain the rate limit message.

    Returns:
        A tuple containing two lists: (list_of_valid_symbols, list_of_invalid_symbols)
    """
    log_and_print(f"--- Performing final validation in {RAW_DATA_DIR} ---")
    valid_symbols = []
    invalid_symbols = []
    all_csv_files = list(RAW_DATA_DIR.glob("*.csv"))

    if not all_csv_files:
        log_and_print("No CSV files found in directory for validation.", level=logging.WARNING)
        return [], []

    for file_path in all_csv_files:
        symbol = file_path.stem
        try:
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read(500) # Read only the start for efficiency
                if is_rate_limit_content(content):
                    invalid_symbols.append(symbol)
                    log_and_print(f"Validation FAILED for {symbol} ({file_path.name}) - Contains rate limit message.", level=logging.WARNING)
                else:
                    valid_symbols.append(symbol)
                    # Optional: Log success at DEBUG level if needed
                    # logging.debug(f"Validation PASSED for {symbol} ({file_path.name})")
        except Exception as e:
            log_and_print(f"Validation ERROR reading {file_path.name}: {e}", level=logging.ERROR)
            invalid_symbols.append(symbol) # Treat unreadable files as invalid

    log_and_print(f"Final validation complete. Valid files: {len(valid_symbols)}, Invalid files: {len(invalid_symbols)}")
    if invalid_symbols:
        log_and_print(f"WARNING: The following symbols correspond to invalid files: {invalid_symbols}", level=logging.WARNING)

    return valid_symbols, invalid_symbols


def main():
    """Main execution function."""
    log_and_print("=== Starting Data Acquisition Process ===")

    # Optional: Initial download for symbols defined in config (if needed)
    log_and_print("--- Running Initial Batch Download ---")
    initial_completed, initial_unsuccessful = download_data_batch(SYMBOLS)
    log_and_print(f"Initial download result - Completed: {len(initial_completed)}, Unsuccessful: {len(initial_unsuccessful)}")

    # --- Attempt to redownload any files previously failed due to rate limits ---
    redownload_success, redownload_fail = redownload_failed_files()

    # --- Perform final validation and report ---
    final_valid, final_invalid = final_validation()

    log_and_print("=== Data Acquisition Process Finished ===")
    log_and_print(f"\nFinal Status Summary:\n  Total Valid Symbols: {len(final_valid)}\n  Total Invalid Symbols: {len(final_invalid)}")
    if final_invalid:
         log_and_print(f"  Invalid Symbols: {final_invalid}", level=logging.WARNING)

if __name__ == "__main__":
    # Ensure the target directory exists before starting
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    main()