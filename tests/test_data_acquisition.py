import pytest
import requests
import time
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import sys
import os

# --- Add src directory to Python path ---
# This assumes your tests directory is parallel to your src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, src_path)

# --- Import directly from the module ---
from data_acquisition import (
    exponential_backoff, get_api_params, download_data_single,
    log_and_print, redownload_failed_files, final_validation,
    is_rate_limit_content, download_data_batch,  # Added other used functions
    RAW_DATA_DIR, MAX_RETRIES, MAX_BACKOFF_SECONDS # Added constants
)

# --- Constants for Testing ---
TEST_SYMBOL_GOOD = "GOOD"
TEST_SYMBOL_BAD = "BAD"
TEST_SYMBOL_RATE_LIMIT = "RATE"
TEST_SYMBOL_IO_ERROR = "IOERR"
TEST_SYMBOL_RETRY = "RETRY"

FAKE_CSV_DATA = "timestamp,open,high,low,close,volume\n2024-01-01,100,101,99,100.5,10000\n"
RATE_LIMIT_NOTE_JSON = """
{
    "Note": "We have detected your API key and our standard API rate limit is 25 requests per day."
}
"""
RATE_LIMIT_INFO_JSON = """
{
    "Information": "API key rate limit exceeded."
}
"""
RATE_LIMIT_THANK_YOU = "Thank you for using Alpha Vantage!"

# --- Fixtures ---

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, mocker):
    """
    Fixture to set up a clean test environment for each test.
    - Creates a temporary raw data directory.
    - Patches the RAW_DATA_DIR constant in the data_acquisition module.
    - Mocks time.sleep to avoid actual pauses.
    """
    test_raw_dir = tmp_path / "raw"
    test_raw_dir.mkdir()

    # Patch the module's RAW_DATA_DIR constant (target remains the same)
    mocker.patch('data_acquisition.RAW_DATA_DIR', test_raw_dir)
    mocker.patch('time.sleep', return_value=None)
    mocker.patch('logging.basicConfig', return_value=None)
    # Patch the log_and_print function *within the data_acquisition module*
    mocker.patch('data_acquisition.log_and_print', return_value=None)

    return test_raw_dir

# --- Tests for Utility Functions ---

@pytest.mark.parametrize(
    "attempt, max_backoff, expected_min, expected_max_plus_epsilon",
    [
        (0, 60, 1.0, 1.999),
        (1, 60, 2.0, 2.999),
        (5, 60, 32.0, 32.999),
        (6, 60, 60.0, 60.0),
        (10, 60, 60.0, 60.0),
        (3, 10, 8.0, 8.999),
        (4, 10, 10.0, 10.0)
    ]
)
def test_exponential_backoff(attempt, max_backoff, expected_min, expected_max_plus_epsilon, mocker):
    """Test exponential backoff calculation with capping and jitter."""
    # Patch MAX_BACKOFF_SECONDS *within the data_acquisition module*
    mocker.patch('data_acquisition.MAX_BACKOFF_SECONDS', max_backoff)
    mocker.patch('time.time', return_value=123456.789)

    # Call the imported function directly
    wait_time = exponential_backoff(attempt)

    assert wait_time >= expected_min
    assert wait_time <= expected_max_plus_epsilon
    assert wait_time <= max_backoff

@pytest.mark.parametrize(
    "content, expected",
    [
        (RATE_LIMIT_NOTE_JSON, True),
        (RATE_LIMIT_INFO_JSON, True),
        (RATE_LIMIT_THANK_YOU, True),
        ('{"Information": "API key rate limit exceeded."}', True),
        ('{"Note": "standard API rate limit is 25 requests per day..."}', True),
        ('Our standard API call frequency is limited', True),
        ('some random text about api key and limits but not the pattern', False),
        (FAKE_CSV_DATA, False),
        ('just some normal csv,data\n1,2', False),
        ('', False),
        ('{"data": "valid"}', False),
        ('premium endpoint access required', True)
    ]
)
def test_is_rate_limit_content(content, expected):
    """Test the rate limit content detection logic."""
    # Call the imported function directly
    assert is_rate_limit_content(content) == expected

def test_get_api_params(mocker):
    """Test the API parameter dictionary creation."""
    # Mock config values used by the function *within the data_acquisition module*
    mocker.patch('data_acquisition.FUNCTION', 'TEST_FUNC')
    mocker.patch('data_acquisition.ALPHA_VANTAGE_API_KEY', 'TEST_KEY')
    mocker.patch('data_acquisition.OUTPUT_SIZE', 'TEST_SIZE')
    mocker.patch('data_acquisition.DATATYPE', 'TEST_TYPE')

    expected = {
        "function": 'TEST_FUNC',
        "symbol": 'TEST_SYM',
        "apikey": 'TEST_KEY',
        "outputsize": 'TEST_SIZE',
        "datatype": 'TEST_TYPE'
    }
    # Call the imported function directly
    assert get_api_params('TEST_SYM') == expected


# --- Tests for Core Download Logic ---

@pytest.fixture
def mock_session(mocker):
    """Fixture to create a mocked requests.Session."""
    mock = mocker.MagicMock(spec=requests.Session)
    mock.__enter__.return_value = mock
    mock.__exit__.return_value = None
    return mock

@pytest.fixture
def mock_response(mocker):
    """Fixture to create a mock requests.Response object."""
    response = mocker.MagicMock(spec=requests.Response)
    response.raise_for_status.return_value = None
    response.status_code = 200
    response.text = FAKE_CSV_DATA
    return response

def test_download_data_single_success(setup_test_environment, mock_session, mock_response, mocker):
    """Test successful download and file write."""
    test_raw_dir = setup_test_environment
    mock_session.get.return_value = mock_response
    mock_response.text = FAKE_CSV_DATA

    # Call the imported function directly
    result = download_data_single(TEST_SYMBOL_GOOD, mock_session)
    expected_file = test_raw_dir / f"{TEST_SYMBOL_GOOD}.csv"

    assert result == "success"
    mock_session.get.assert_called_once()
    assert expected_file.exists()
    assert expected_file.read_text(encoding='utf-8') == FAKE_CSV_DATA

def test_download_data_single_rate_limit(setup_test_environment, mock_session, mock_response):
    """Test handling of rate limit response."""
    test_raw_dir = setup_test_environment
    mock_response.text = RATE_LIMIT_NOTE_JSON
    mock_session.get.return_value = mock_response

    # Call the imported function directly
    result = download_data_single(TEST_SYMBOL_RATE_LIMIT, mock_session)
    expected_file = test_raw_dir / f"{TEST_SYMBOL_RATE_LIMIT}.csv"

    assert result == "rate_limit"
    mock_session.get.assert_called_once()
    assert not expected_file.exists()

def test_download_data_single_http_error_retry_fail(setup_test_environment, mock_session, mock_response, mocker):
    """Test retries on HTTP error eventually failing."""
    test_raw_dir = setup_test_environment
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Test HTTP Error")
    mock_session.get.return_value = mock_response
    mock_sleep = mocker.patch('time.sleep')

    # Call the imported function directly
    # Use the imported constant directly
    result = download_data_single(TEST_SYMBOL_BAD, mock_session)
    expected_file = test_raw_dir / f"{TEST_SYMBOL_BAD}.csv"

    assert result == "failed"
    assert mock_session.get.call_count == MAX_RETRIES
    assert mock_sleep.call_count == MAX_RETRIES - 1
    assert not expected_file.exists()

def test_download_data_single_request_error_retry_fail(setup_test_environment, mock_session, mocker):
    """Test retries on general RequestException eventually failing."""
    test_raw_dir = setup_test_environment
    mock_session.get.side_effect = requests.exceptions.RequestException("Test Connection Error")
    mock_sleep = mocker.patch('time.sleep')

    # Call the imported function directly
    # Use the imported constant directly
    result = download_data_single(TEST_SYMBOL_BAD, mock_session)
    expected_file = test_raw_dir / f"{TEST_SYMBOL_BAD}.csv"

    assert result == "failed"
    assert mock_session.get.call_count == MAX_RETRIES
    assert mock_sleep.call_count == MAX_RETRIES - 1
    assert not expected_file.exists()

def test_download_data_single_io_error(setup_test_environment, mock_session, mock_response, mocker):
    """Test handling of file writing error."""
    test_raw_dir = setup_test_environment
    mock_session.get.return_value = mock_response
    mock_response.text = FAKE_CSV_DATA
    mocker.patch('builtins.open', side_effect=IOError("Test file write error"))

    # Call the imported function directly
    result = download_data_single(TEST_SYMBOL_IO_ERROR, mock_session)

    assert result is None
    mock_session.get.assert_called_once()

def test_download_data_single_retry_success(setup_test_environment, mock_session, mock_response, mocker):
    """Test successful download after one retry."""
    test_raw_dir = setup_test_environment
    mock_sleep = mocker.patch('time.sleep')
    mock_session.get.side_effect = [
        requests.exceptions.RequestException("Temporary Error"),
        mock_response
    ]
    mock_response.text = FAKE_CSV_DATA

    # Call the imported function directly
    result = download_data_single(TEST_SYMBOL_RETRY, mock_session)
    expected_file = test_raw_dir / f"{TEST_SYMBOL_RETRY}.csv"

    assert result == "success"
    assert mock_session.get.call_count == 2
    mock_sleep.assert_called_once()
    assert expected_file.exists()
    assert expected_file.read_text(encoding='utf-8') == FAKE_CSV_DATA


# --- Tests for Batch/Workflow Logic ---

# Patch still targets the function within its original module
@patch('data_acquisition.download_data_single')
def test_download_data_batch_all_success(mock_download_single, setup_test_environment, mocker):
    """Test batch download where all symbols succeed."""
    mock_download_single.return_value = "success"
    symbols = ["S1", "S2", "S3"]
    mock_sleep = mocker.patch('time.sleep')

    # Call the imported function directly
    completed, unsuccessful = download_data_batch(symbols)

    assert completed == symbols
    assert unsuccessful == []
    assert mock_download_single.call_count == len(symbols)
    assert mock_sleep.call_count == len(symbols)

@patch('data_acquisition.download_data_single')
def test_download_data_batch_some_fail(mock_download_single, setup_test_environment):
    """Test batch download with some non-rate-limit failures."""
    symbols = ["S1", "S2", "S3", "S4"]
    mock_download_single.side_effect = ["success", "failed", "success", "failed"]

    # Call the imported function directly
    completed, unsuccessful = download_data_batch(symbols)

    assert completed == ["S1", "S3"]
    assert unsuccessful == ["S2", "S4"]
    assert mock_download_single.call_count == len(symbols)

@patch('data_acquisition.download_data_single')
def test_download_data_batch_rate_limit_hit(mock_download_single, setup_test_environment):
    """Test batch download stopping after a rate limit is hit."""
    symbols = ["S1", "S2", "S3", "S4", "S5"]
    mock_download_single.side_effect = ["success", "success", "rate_limit", "success", "success"]

    # Call the imported function directly
    completed, unsuccessful = download_data_batch(symbols)

    assert completed == ["S1", "S2"]
    assert unsuccessful == ["S3", "S4", "S5"]
    assert mock_download_single.call_count == 3


# --- Tests for Redownload Logic ---

# Helper to create dummy files
def create_dummy_file(dir_path: Path, symbol: str, content: str):
    file = dir_path / f"{symbol}.csv"
    file.write_text(content, encoding='utf-8')
    return file

@patch('data_acquisition.download_data_single')
def test_redownload_no_files(mock_download_single, setup_test_environment):
    """Test redownload when the target directory is empty."""
    # Call the imported function directly
    completed, failed = redownload_failed_files()
    assert completed == []
    assert failed == []
    mock_download_single.assert_not_called()

@patch('data_acquisition.download_data_single')
def test_redownload_only_good_files(mock_download_single, setup_test_environment):
    """Test redownload when only valid files exist."""
    test_raw_dir = setup_test_environment
    create_dummy_file(test_raw_dir, "GOOD1", FAKE_CSV_DATA)
    create_dummy_file(test_raw_dir, "GOOD2", FAKE_CSV_DATA)

    # Call the imported function directly
    completed, failed = redownload_failed_files()

    assert completed == []
    assert failed == []
    mock_download_single.assert_not_called()

@patch('data_acquisition.download_data_single')
def test_redownload_only_bad_files_all_success(mock_download_single, setup_test_environment):
    """Test redownload where bad files exist and redownload succeeds."""
    test_raw_dir = setup_test_environment
    symbols_to_retry = ["BAD1", "BAD2"]
    create_dummy_file(test_raw_dir, symbols_to_retry[0], RATE_LIMIT_NOTE_JSON)
    create_dummy_file(test_raw_dir, symbols_to_retry[1], RATE_LIMIT_INFO_JSON)
    create_dummy_file(test_raw_dir, "GOOD1", FAKE_CSV_DATA)

    mock_download_single.return_value = "success"

    # Call the imported function directly
    completed, failed = redownload_failed_files()

    assert sorted(completed) == sorted(symbols_to_retry)
    assert failed == []
    assert mock_download_single.call_count == len(symbols_to_retry)
    call_args_list = [c.args[0] for c in mock_download_single.call_args_list]
    assert sorted(call_args_list) == sorted(symbols_to_retry)

@patch('data_acquisition.download_data_single')
def test_redownload_only_bad_files_some_fail(mock_download_single, setup_test_environment):
    """Test redownload where some redownload attempts fail (non-rate-limit)."""
    test_raw_dir = setup_test_environment
    symbols_to_retry = ["BAD1", "BAD2", "BAD3"]
    create_dummy_file(test_raw_dir, "BAD1", RATE_LIMIT_NOTE_JSON)
    create_dummy_file(test_raw_dir, "BAD2", RATE_LIMIT_INFO_JSON)
    create_dummy_file(test_raw_dir, "BAD3", RATE_LIMIT_THANK_YOU)
    expected_results = {"BAD1": "success", "BAD2": "failed", "BAD3": "success"}
    mock_download_single.side_effect = lambda symbol, session: expected_results.get(symbol, "unexpected_call")

    # Call the imported function directly
    completed, failed = redownload_failed_files()

    assert sorted(completed) == sorted([symbols_to_retry[0], symbols_to_retry[2]])
    assert sorted(failed) == sorted(["BAD2"])
    assert mock_download_single.call_count == len(symbols_to_retry)

@patch('data_acquisition.download_data_single')
def test_redownload_rate_limit_hit(mock_download_single, setup_test_environment, mocker): # Added mocker
    """Test redownload stopping when a rate limit is hit during the process."""
    test_raw_dir = setup_test_environment
    symbols_identified = ["BAD1", "BAD2", "BAD3", "BAD4"]
    # Mock glob directly on the patched RAW_DATA_DIR Path object if needed,
    # or rely on tmp_path structure if file creation order is sufficient.
    # For consistency:
    mocker.patch.object(Path, 'glob', return_value=[ # Patch glob on Path class
        test_raw_dir / f"{s}.csv" for s in symbols_identified
    ])

    create_dummy_file(test_raw_dir, symbols_identified[0], RATE_LIMIT_NOTE_JSON)
    create_dummy_file(test_raw_dir, symbols_identified[1], RATE_LIMIT_INFO_JSON)
    create_dummy_file(test_raw_dir, symbols_identified[2], RATE_LIMIT_NOTE_JSON)
    create_dummy_file(test_raw_dir, symbols_identified[3], RATE_LIMIT_THANK_YOU)
    mock_download_single.side_effect = ["success", "rate_limit", "success", "success"]

    # Call the imported function directly
    completed, failed = redownload_failed_files()

    assert completed == [symbols_identified[0]]
    assert sorted(failed) == sorted([symbols_identified[1], symbols_identified[2], symbols_identified[3]])
    assert mock_download_single.call_count == 2
    call_args_list = [c.args[0] for c in mock_download_single.call_args_list]
    assert call_args_list == symbols_identified[:2]


# --- Tests for Final Validation ---

def test_final_validation_no_files(setup_test_environment):
    """Test final validation with an empty directory."""
    # Call the imported function directly
    valid, invalid = final_validation()
    assert valid == []
    assert invalid == []

def test_final_validation_all_good(setup_test_environment):
    """Test final validation when all files are valid."""
    test_raw_dir = setup_test_environment
    symbols = ["V1", "V2", "V3"]
    for s in symbols:
        create_dummy_file(test_raw_dir, s, FAKE_CSV_DATA)

    # Call the imported function directly
    valid, invalid = final_validation()

    assert sorted(valid) == sorted(symbols)
    assert invalid == []

def test_final_validation_all_bad(setup_test_environment):
    """Test final validation when all files contain rate limit messages."""
    test_raw_dir = setup_test_environment
    symbols = ["B1", "B2"]
    create_dummy_file(test_raw_dir, symbols[0], RATE_LIMIT_INFO_JSON)
    create_dummy_file(test_raw_dir, symbols[1], RATE_LIMIT_NOTE_JSON)

    # Call the imported function directly
    valid, invalid = final_validation()

    assert valid == []
    assert sorted(invalid) == sorted(symbols)

def test_final_validation_mixed(setup_test_environment):
    """Test final validation with a mix of good and bad files."""
    test_raw_dir = setup_test_environment
    good_symbols = ["G1", "G2"]
    bad_symbols = ["B1"]
    create_dummy_file(test_raw_dir, good_symbols[0], FAKE_CSV_DATA)
    create_dummy_file(test_raw_dir, bad_symbols[0], RATE_LIMIT_NOTE_JSON)
    create_dummy_file(test_raw_dir, good_symbols[1], FAKE_CSV_DATA)

    # Call the imported function directly
    valid, invalid = final_validation()

    assert sorted(valid) == sorted(good_symbols)
    assert sorted(invalid) == sorted(bad_symbols)

def test_final_validation_read_error(setup_test_environment, mocker):
    """Test final validation handling a file read error."""
    test_raw_dir = setup_test_environment
    symbols = ["GOOD", "READ_ERR", "GOOD2"]
    create_dummy_file(test_raw_dir, symbols[0], FAKE_CSV_DATA)
    error_file = create_dummy_file(test_raw_dir, symbols[1], "cannot read this")
    create_dummy_file(test_raw_dir, symbols[2], FAKE_CSV_DATA)

    original_open = open
    def mock_open_wrapper(*args, **kwargs):
        # Use Path object comparison for robustness
        if Path(args[0]) == error_file:
            raise IOError("Permission denied simulation")
        kwargs.setdefault('encoding', 'utf-8')
        return original_open(*args, **kwargs)

    mocker.patch('builtins.open', side_effect=mock_open_wrapper)

    # Call the imported function directly
    valid, invalid = final_validation()

    assert sorted(valid) == sorted([symbols[0], symbols[2]])
    assert invalid == [symbols[1]]