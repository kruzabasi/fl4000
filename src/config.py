import os
from gcis import get_gcis
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
FUNCTION = "TIME_SERIES_DAILY"
OUTPUT_SIZE = "full" # Get all available data
DATATYPE = "csv" # for easier parsing with pandas

#  Tiingo API configuration
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo"
TIINGO_FUNCTION = "daily"
TIINGO_DATATYPE = "csv"  # for easier parsing with pandas
TIINGO_START_DATE = "2005-01-04"
TIINGO_END_DATE = "2025-04-18"

# List of FTSE 100 sample symbols
SYMBOLS = ["SHEL.LON", "BP.LON", "HSBA.LON", "BARC.LON", "LLOY.LON", "LGEN.LON", "PRU.LON", "AV.LON", "STAN.LON", "LSEG.LON", "SDR.LON", "ADM.LON", "GSK.LON", "AZN.LON", "SN.LON", "ULVR.LON", "DGE.LON", "BATS.LON", "TSCO.LON", "IMB.LON", "RKT.LON", "RIO.LON", "AAL.LON", "ANTO.LON", "MNDI.LON", "VOD.LON", "BT-A.LON", "NG.LON", "SSE.LON", "SVT.LON", "UU.LON", "REL.LON", "BA.LON", "EXPN.LON", "AHT.LON", "SMIN.LON", "RR.LON", "ITRK.LON", "SGE.LON", "AUTO.LON", "LAND.LON", "SGRO.LON", "CPG.LON", "NXT.LON", "WTB.LON", "IHG.LON", "KGF.LON", "JD.LON"]

# Sleep duration in seconds (5 requests/min limit)
SLEEP_DURATION = 12

# Data storage path
DATA_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
NORMALIZED_DIR = os.path.join("data", "normalized")
RAW_T_DATA_DIR = os.path.join("data", "raw_t")

# Federated Learning Simulation Configuration
# Update these paths as needed for your environment
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEDERATED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "federated")  # Absolute path
NUM_CLIENTS = 40  # Update to match your number of clients
RANDOM_SEED = 42
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "results")  # Absolute path
LEARNING_RATE = 0.0001  # eta (further reduced for stability)
C_CLIP = 5.0  # L2 norm clipping threshold for client updates (tune as needed)

# Ensure the data directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RAW_T_DATA_DIR, exist_ok=True)

# GCIs
GCIS = get_gcis()