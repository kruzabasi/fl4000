import requests
import time
import os
import logging
from config import ALPHA_VANTAGE_API_KEY, BASE_URL, FUNCTION, OUTPUT_SIZE, DATATYPE, SYMBOLS, SLEEP_DURATION, DATA_DIR

# Set up logging
logging.basicConfig(
    filename=os.path.join("logs", "ingest.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def exponential_backoff(attempt):
    """Calculate exponential backoff time."""
    return min(2 ** attempt, 60)  # Capped at 60 seconds

def download_data_single(symbol):
    params = {
        "function": FUNCTION,
        "symbol": symbol,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": OUTPUT_SIZE,
        "datatype": DATATYPE
    }
    attempt = 0
    while attempt < 5:  # Retry up to 5 times
        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()

            # Handle Alpha Vantage rate limit messages
            if ("Thank you for using Alpha Vantage" in response.text or
                'Information' in response.text and 'API key' in response.text and 'rate limit' in response.text):
                logging.warning(f"Rate limit reached or API key limit for {symbol}. Response: {response.text}")
                print(f"Rate limit or error response for {symbol}: {response.text.strip()}")
                return "rate_limit"

            file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
            with open(file_path, "w") as f:
                f.write(response.text)

            logging.info(f"Downloaded data for {symbol} to {file_path}")
            print(f"Downloaded data for {symbol} to {file_path}")
            return True

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error for {symbol}: {http_err}")
            print(f"HTTP error for {symbol}: {http_err}")
        except Exception as e:
            logging.error(f"Error downloading data for {symbol}: {e}")
            print(f"Error downloading data for {symbol}: {e}")

        attempt += 1
        sleep_time = exponential_backoff(attempt)
        print(f"Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)

    return False


def download_data(symbols):
    """
    Download data for a list of symbols. Stops if rate limit is hit. Returns (completed, unsuccessful).
    """
    completed = []
    unsuccessful = []
    attempted = 0
    for symbol in symbols:
        result = download_data_single(symbol)
        attempted += 1
        if result == True:
            completed.append(symbol)
        elif result == "rate_limit":
            unsuccessful.append(symbol)
            logging.error(f"Rate limit hit during download of {symbol}. Stopping further attempts.")
            print(f"Rate limit hit during download of {symbol}. Stopping further attempts.")
            # All remaining symbols are unsuccessful
            unsuccessful.extend([s for s in symbols if s not in completed and s not in unsuccessful and s != symbol])
            break
        else:
            unsuccessful.append(symbol)
    logging.info(f"Download attempted for {attempted} symbols.")
    logging.info(f"Successfully downloaded: {completed}")
    logging.info(f"Unsuccessful (rate limit or error): {unsuccessful}")
    print(f"Download attempted for {attempted} symbols.")
    print(f"Successfully downloaded: {completed}")
    print(f"Unsuccessful (rate limit or error): {unsuccessful}")
    return completed, unsuccessful


def redownload_failed_files():
    """
    Check all files in data/raw/. If a file contains the API rate limit message, attempt to redownload it.
    Stops all attempts if rate limit is hit again. Logs summary of successes and failures.
    """
    raw_dir = os.path.join(DATA_DIR, "raw") if not DATA_DIR.endswith("raw") else DATA_DIR
    completed = []
    unsuccessful = []
    attempted = 0
    rate_limit_hit = False
    for filename in os.listdir(raw_dir):
        if rate_limit_hit:
            # Once rate limit is hit, skip further attempts but still record unattempted files as unsuccessful
            if filename.endswith(".csv"):
                unsuccessful.append(filename.replace(".csv", ""))
            continue
        if filename.endswith(".csv"):
            file_path = os.path.join(raw_dir, filename)
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    # Check for JSON-style API limit message
                    if (
                        'Information' in content and 'API key' in content and 'rate limit' in content
                    ):
                        symbol = filename.replace(".csv", "")
                        logging.info(f"Attempting to redownload {symbol} due to previous API rate limit.")
                        print(f"Attempting to redownload {symbol} due to previous API rate limit.")
                        result = download_data_single(symbol)
                        attempted += 1
                        if result == True:
                            completed.append(symbol)
                        elif result == "rate_limit":
                            unsuccessful.append(symbol)
                            logging.error(f"Rate limit hit during redownload of {symbol}. Stopping further attempts.")
                            print(f"Rate limit hit during redownload of {symbol}. Stopping further attempts.")
                            rate_limit_hit = True
                        else:
                            unsuccessful.append(symbol)
                    else:
                        # If file is not a rate-limited file, consider it already complete
                        completed.append(filename.replace(".csv", ""))
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
                unsuccessful.append(filename.replace(".csv", ""))

    logging.info(f"Redownload attempted for {attempted} files.")
    logging.info(f"Successfully redownloaded or already complete: {completed}")
    logging.info(f"Unsuccessful (rate limit or error): {unsuccessful}")
    print(f"Redownload attempted for {attempted} files.")
    print(f"Successfully redownloaded or already complete: {completed}")
    print(f"Unsuccessful (rate limit or error): {unsuccessful}")

    return completed, unsuccessful


def main():
    completed, unsuccessful = download_data(SYMBOLS)
    if len(unsuccessful) > 0:
        print(f"Some files could not be downloaded due to error or rate limit.")
    print(f"Data acquisition complete.\nSummary:\n  Successful: {len(completed)}\n  Unsuccessful: {len(unsuccessful)}")

if __name__ == "__main__":
    main()
