import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
FUNCTION = "TIME_SERIES_DAILY"
OUTPUT_SIZE = "full" # Get all available data
DATATYPE = "csv" # for easier parsing with pandas

# List of FTSE 100 sample symbols
SYMBOLS = ["SHEL.LON", "BP.LON", "HSBA.LON", "BARC.LON", "LLOY.LON", "LGEN.LON", "PRU.LON", "AV.LON", "STAN.LON", "LSEG.LON", "SDR.LON", "ADM.LON", "GSK.LON", "AZN.LON", "SN.LON", "ULVR.LON", "DGE.LON", "BATS.LON", "TSCO.LON", "IMB.LON", "RKT.LON", "RIO.LON", "AAL.LON", "ANTO.LON", "MNDI.LON", "VOD.LON", "BT-A.LON", "NG.LON", "SSE.LON", "SVT.LON", "UU.LON", "REL.LON", "BA.LON", "EXPN.LON", "AHT.LON", "SMIN.LON", "RR.LON", "ITRK.LON", "SGE.LON", "AUTO.LON", "LAND.LON", "SGRO.LON", "CPG.LON", "NXT.LON", "WTB.LON", "IHG.LON", "KGF.LON", "JD.LON"]

# Sleep duration in seconds (5 requests/min limit)
SLEEP_DURATION = 12

# Data storage path
DATA_DIR = os.path.join("data", "raw")

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)