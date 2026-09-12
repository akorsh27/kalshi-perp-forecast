"""
Centralized configuration for the Kalshi Perps Forecast project.
Loads settings from environment variables (.env file) and exposes them
as module-level constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "kalshi_perps.db")))

# --- Kalshi API ---
KALSHI_ENV = os.getenv("KALSHI_ENV", "demo")
KALSHI_API_KEY_ID = os.getenv("KALSHI_API_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = Path(
    os.getenv("KALSHI_PRIVATE_KEY_PATH", str(PROJECT_ROOT / "keys" / "kalshi_private.key"))
)

_BASE_URLS = {
    "demo": "https://external-api.demo.kalshi.co/trade-api/v2",
    "production": "https://external-api.kalshi.com/trade-api/v2",
}
KALSHI_BASE_URL = _BASE_URLS[KALSHI_ENV]

# --- Market Defaults ---
DEFAULT_TICKER = "KXBTCPERP1"

# Candlestick intervals (minutes)
INTERVAL_1MIN = 1
INTERVAL_1HR = 60
INTERVAL_1DAY = 1440

# --- Rate Limiting ---
# Kalshi default: 10 tokens/request, token-bucket with tiers
MAX_RETRIES = 5
RETRY_BACKOFF_FACTOR = 2  # exponential backoff base (seconds)
REQUEST_TIMEOUT = 30  # seconds

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
