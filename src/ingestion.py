"""
Kalshi Perpetual Futures API client.

Handles RSA-PSS authentication, rate-limit-aware retries, and provides
methods for pulling candlestick, trade, and funding-rate data.
"""

import base64
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from src.config import (
    KALSHI_BASE_URL,
    KALSHI_API_KEY_ID,
    KALSHI_PRIVATE_KEY_PATH,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    REQUEST_TIMEOUT,
    DEFAULT_TICKER,
    INTERVAL_1HR,
    LOG_LEVEL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class KalshiAPIError(Exception):
    """Raised when the Kalshi API returns a non-2xx response."""

    def __init__(self, status_code: int, message: str, details: str = ""):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(f"HTTP {status_code}: {message} — {details}")


class RateLimitError(KalshiAPIError):
    """Raised specifically on 429 responses to trigger retry logic."""
    pass


class KalshiClient:
    """
    Authenticated REST client for Kalshi's Perpetual Futures (Margin) API.

    Usage:
        client = KalshiClient()
        markets = client.get_markets()
        candles = client.get_candlesticks("BTC-USD", start_ts, end_ts)
    """

    def __init__(
        self,
        base_url: str = KALSHI_BASE_URL,
        api_key_id: str = KALSHI_API_KEY_ID,
        private_key_path: str = str(KALSHI_PRIVATE_KEY_PATH),
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key_id = api_key_id
        self._private_key = self._load_private_key(private_key_path)
        self.session = requests.Session()

        logger.info("KalshiClient initialized (env: %s)", self.base_url)

    @staticmethod
    def _load_private_key(path: str):
        """Load RSA private key from PEM file."""
        try:
            with open(path, "rb") as f:
                private_key = serialization.load_pem_private_key(f.read(), password=None)
            logger.debug("Private key loaded from %s", path)
            return private_key
        except FileNotFoundError:
            logger.warning(
                "Private key not found at %s. "
                "Unauthenticated requests only (public endpoints).",
                path,
            )
            return None
        except Exception as e:
            logger.error("Failed to load private key: %s", e)
            return None

    def _sign_request(self, method: str, path: str) -> dict:
        """
        Generate Kalshi authentication headers using RSA-PSS.

        Signs: timestamp_ms + METHOD + path (without query params).
        """
        if self._private_key is None:
            return {}

        timestamp_ms = str(int(time.time() * 1000))

        # Path to sign: strip query params, include /trade-api/v2 prefix
        sign_path = path.split("?")[0]
        message = f"{timestamp_ms}{method.upper()}{sign_path}"

        signature = self._private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=hashes.SHA256.digest_size,  # 32 bytes
            ),
            hashes.SHA256(),
        )

        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
        }

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_BACKOFF_FACTOR, min=1, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _request(self, method: str, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make an authenticated request to the Kalshi API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API path relative to base URL (e.g., "/margin/markets")
            params: Query parameters

        Returns:
            Parsed JSON response body

        Raises:
            RateLimitError: On 429 (triggers automatic retry)
            KalshiAPIError: On other non-2xx responses
        """
        # Build full path for signing (includes /trade-api/v2 prefix)
        # The base_url already contains the prefix, so extract it
        url = f"{self.base_url}{endpoint}"

        # The path we sign must include /trade-api/v2
        sign_path = f"/trade-api/v2{endpoint}"
        headers = self._sign_request(method, sign_path)
        headers["Accept"] = "application/json"

        logger.debug("%s %s params=%s", method, endpoint, params)

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.ConnectionError as e:
            logger.error("Connection failed: %s", e)
            raise
        except requests.exceptions.Timeout as e:
            logger.error("Request timed out: %s", e)
            raise

        if response.status_code == 429:
            body = response.json() if response.content else {}
            raise RateLimitError(
                429,
                body.get("message", "Rate limit exceeded"),
                body.get("details", ""),
            )

        if response.status_code >= 400:
            body = response.json() if response.content else {}
            raise KalshiAPIError(
                response.status_code,
                body.get("message", response.reason),
                body.get("details", ""),
            )

        return response.json()

    # ─── Public Market Data Endpoints ────────────────────────────────────

    def get_markets(self, status: Optional[str] = None) -> list[dict]:
        """List available margin markets."""
        params = {}
        if status:
            params["status"] = status
        data = self._request("GET", "/margin/markets", params=params or None)
        return data.get("markets", [])

    def get_market(self, ticker: str = DEFAULT_TICKER) -> dict:
        """Get a single market with current trading stats."""
        return self._request("GET", f"/margin/markets/{ticker}")

    def get_candlesticks(
        self,
        ticker: str = DEFAULT_TICKER,
        start_ts: int = None,
        end_ts: int = None,
        period_interval: int = INTERVAL_1HR,
    ) -> list[dict]:
        """
        Fetch OHLCV candlestick data for a margin market.

        Args:
            ticker: Market ticker (e.g., "BTC-USD")
            start_ts: Start Unix timestamp (seconds). Defaults to 7 days ago.
            end_ts: End Unix timestamp (seconds). Defaults to now.
            period_interval: Candle size in minutes (1, 60, or 1440).

        Returns:
            List of candlestick dicts with fields: end_period_ts, price
            (OHLC + mean + previous), volume, open_interest, etc.
        """
        now = int(time.time())
        if end_ts is None:
            end_ts = now
        if start_ts is None:
            start_ts = now - (7 * 24 * 3600)  # 7 days ago

        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }

        logger.info(
            "Fetching candlesticks: ticker=%s, interval=%dm, range=[%s, %s]",
            ticker,
            period_interval,
            datetime.fromtimestamp(start_ts, tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(end_ts, tz=timezone.utc).isoformat(),
        )

        data = self._request("GET", f"/margin/markets/{ticker}/candlesticks", params=params)
        candles = data.get("candlesticks", [])
        logger.info("Retrieved %d candlesticks", len(candles))
        return candles

    def get_trades(
        self,
        ticker: str = DEFAULT_TICKER,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        limit: int = 1000,
    ) -> list[dict]:
        """
        Fetch public margin trades (paginated — fetches all pages).

        Args:
            ticker: Market ticker
            min_ts: Filter trades after this Unix timestamp
            max_ts: Filter trades before this Unix timestamp
            limit: Results per page (max 1000)

        Returns:
            List of all trade records within the time range.
        """
        all_trades = []
        cursor = None

        while True:
            params = {"ticker": ticker, "limit": limit}
            if min_ts:
                params["min_ts"] = min_ts
            if max_ts:
                params["max_ts"] = max_ts
            if cursor:
                params["cursor"] = cursor

            data = self._request("GET", "/margin/trades", params=params)
            trades = data.get("trades", [])
            all_trades.extend(trades)

            cursor = data.get("cursor")
            if not cursor or len(trades) < limit:
                break

            logger.debug("Paginating trades: %d fetched so far", len(all_trades))

        logger.info("Retrieved %d trades total for %s", len(all_trades), ticker)
        return all_trades

    def get_historical_funding_rates(
        self,
        ticker: str = DEFAULT_TICKER,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> list[dict]:
        """
        Fetch historical funding rates for a market.

        Args:
            ticker: Market ticker (leave empty string for all markets)
            start_ts: Start Unix timestamp (seconds)
            end_ts: End Unix timestamp (seconds)

        Returns:
            List of funding rate records.
        """
        params = {}
        if ticker:
            params["ticker"] = ticker
        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts

        data = self._request("GET", "/margin/funding_rates/historical", params=params)
        rates = data.get("funding_rates", [])
        logger.info("Retrieved %d funding rate records for %s", len(rates), ticker or "ALL")
        return rates

    def get_funding_rate_estimate(self, ticker: str = DEFAULT_TICKER) -> dict:
        """Get the current in-progress funding rate estimate."""
        return self._request(
            "GET", "/margin/funding_rates/estimate", params={"ticker": ticker}
        )


# ─── Convenience: CLI-style entry point for quick data pulls ─────────────

if __name__ == "__main__":
    import json

    client = KalshiClient()

    print("=== Available Markets ===")
    markets = client.get_markets()
    for m in markets:
        print(f"  {m.get('ticker', '?')}: {m.get('title', m.get('subtitle', ''))}")

    if markets:
        ticker = markets[0].get("ticker", DEFAULT_TICKER)
        print(f"\n=== Last 24h Candlesticks ({ticker}, 1hr) ===")
        now = int(time.time())
        candles = client.get_candlesticks(
            ticker=ticker,
            start_ts=now - 86400,
            end_ts=now,
            period_interval=INTERVAL_1HR,
        )
        print(json.dumps(candles[:3], indent=2))
        print(f"  ... ({len(candles)} total)")
