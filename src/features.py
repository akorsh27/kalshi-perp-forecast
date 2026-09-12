"""
Feature engineering for 1-hour-ahead realized volatility forecasting.

Takes raw candlestick and funding-rate data from the database and produces
a feature matrix suitable for model training.

Target variable:
    realized_vol_1h — standard deviation of log-returns over the NEXT hour
    (computed from 1-minute candles aggregated to hourly, or from hourly close-to-close returns
    of the following period).

Features:
    - Lagged realized volatility (1h, 4h, 24h lookbacks)
    - Log returns (1h, 4h)
    - Rolling volume & volume change ratios
    - Bid-ask spread (proxy for liquidity)
    - Open interest level & change
    - Funding rate & funding rate delta
    - Hour-of-day & day-of-week cyclical encodings
"""

import logging
from typing import List

import numpy as np
import pandas as pd

from src.config import LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def build_feature_matrix(
    candles_df: pd.DataFrame,
    funding_df: pd.DataFrame = None,
    target_horizon: int = 1,  # hours ahead
) -> pd.DataFrame:
    """
    Build feature matrix from hourly candlestick data.

    Args:
        candles_df: DataFrame from Database.get_candlesticks_df() (hourly interval)
        funding_df: DataFrame from Database.get_funding_rates_df() (optional)
        target_horizon: Number of hours ahead to predict (default 1)

    Returns:
        DataFrame with features and target column 'target_realized_vol',
        indexed by end_period_ts. Rows with NaN targets (future-looking) are dropped.
    """
    df = candles_df.copy()

    if df.empty:
        logger.warning("Empty candlestick DataFrame — cannot build features")
        return pd.DataFrame()

    df = df.sort_values("end_period_ts").reset_index(drop=True)

    # ─── Price Returns ───────────────────────────────────────────────────
    df["log_return_1h"] = np.log(df["price_close"] / df["price_close"].shift(1))
    df["log_return_4h"] = np.log(df["price_close"] / df["price_close"].shift(4))
    df["log_return_24h"] = np.log(df["price_close"] / df["price_close"].shift(24))

    # ─── Realized Volatility (lagged — lookback windows) ─────────────────
    df["realized_vol_1h"] = df["log_return_1h"].rolling(1).std()  # single-period (for target)
    df["realized_vol_4h"] = df["log_return_1h"].rolling(4).std()
    df["realized_vol_12h"] = df["log_return_1h"].rolling(12).std()
    df["realized_vol_24h"] = df["log_return_1h"].rolling(24).std()

    # Volatility ratio (short-term vs long-term — captures vol clustering)
    df["vol_ratio_4h_24h"] = df["realized_vol_4h"] / df["realized_vol_24h"].replace(0, np.nan)

    # ─── Price Range Features ────────────────────────────────────────────
    df["candle_range"] = (df["price_high"] - df["price_low"]) / df["price_close"]
    df["candle_range_ma4"] = df["candle_range"].rolling(4).mean()

    # ─── Volume Features ─────────────────────────────────────────────────
    df["volume_ma4"] = df["volume"].rolling(4).mean()
    df["volume_ma24"] = df["volume"].rolling(24).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma24"].replace(0, np.nan)
    df["volume_change"] = df["volume"].pct_change()

    # Notional volume
    df["notional_vol_ma4"] = df["volume_notional_usd"].rolling(4).mean()

    # ─── Open Interest Features ──────────────────────────────────────────
    df["oi_change"] = df["open_interest"].pct_change()
    df["oi_ma4"] = df["open_interest"].rolling(4).mean()
    df["oi_change_ma4"] = df["oi_change"].rolling(4).mean()

    # ─── Bid-Ask Spread (liquidity proxy) ────────────────────────────────
    df["spread"] = (df["ask_close"] - df["bid_close"]) / df["price_close"]
    df["spread_ma4"] = df["spread"].rolling(4).mean()

    # ─── Time Features (cyclical encoding) ───────────────────────────────
    if "end_period_dt" in df.columns:
        dt = pd.to_datetime(df["end_period_dt"])
    else:
        dt = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)

    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ─── Funding Rate Features ───────────────────────────────────────────
    if funding_df is not None and not funding_df.empty:
        df = _merge_funding_features(df, funding_df)

    # ─── Target: Forward-looking realized volatility ─────────────────────
    # Realized vol of the NEXT target_horizon periods
    df["target_realized_vol"] = (
        df["log_return_1h"]
        .shift(-target_horizon)
        .rolling(target_horizon)
        .std()
        .shift(-target_horizon + 1)
    )

    # For a 1-hour horizon: target = abs(log return of next period)
    # More robust single-period target:
    df["target_realized_vol"] = df["log_return_1h"].shift(-target_horizon).abs()

    # ─── Clean up ────────────────────────────────────────────────────────
    feature_cols = _get_feature_columns()
    target_col = "target_realized_vol"

    # Keep only rows where we have both features and target
    result = df[["end_period_ts", "end_period_dt"] + feature_cols + [target_col]].copy()
    result = result.dropna(subset=[target_col])

    logger.info(
        "Feature matrix built: %d rows, %d features, target coverage %.1f%%",
        len(result),
        len(feature_cols),
        100 * len(result) / max(len(df), 1),
    )

    return result


def _merge_funding_features(df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge funding rate data into the candlestick feature DataFrame.
    Uses asof merge to align funding rates (which may be at different intervals)
    with the hourly candle timestamps.
    """
    funding = funding_df[["funding_time", "funding_rate", "mark_price"]].copy()
    funding = funding.sort_values("funding_time").drop_duplicates(subset=["funding_time"])
    funding = funding.rename(columns={"funding_time": "end_period_ts"})

    df = pd.merge_asof(
        df.sort_values("end_period_ts"),
        funding,
        on="end_period_ts",
        direction="backward",
        suffixes=("", "_funding"),
    )

    df["funding_rate_lag1"] = df["funding_rate"].shift(1)
    df["funding_rate_delta"] = df["funding_rate"] - df["funding_rate_lag1"]

    return df


def _get_feature_columns() -> List[str]:
    """Return the list of feature column names used by the model."""
    return [
        "log_return_1h",
        "log_return_4h",
        "log_return_24h",
        "realized_vol_4h",
        "realized_vol_12h",
        "realized_vol_24h",
        "vol_ratio_4h_24h",
        "candle_range",
        "candle_range_ma4",
        "volume_ratio",
        "volume_change",
        "oi_change",
        "oi_change_ma4",
        "spread",
        "spread_ma4",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        # Funding features (may be NaN if not available)
        "funding_rate",
        "funding_rate_delta",
    ]


def get_feature_names() -> List[str]:
    """Public accessor for feature column names."""
    return _get_feature_columns()


# ─── CLI entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.database import Database

    db = Database()
    candles = db.get_candlesticks_df(interval_minutes=60)
    funding = db.get_funding_rates_df()

    if candles.empty:
        print("No candlestick data in database. Run ingestion first:")
        print("  python -m src.ingestion")
    else:
        features = build_feature_matrix(candles, funding)
        print(f"\nFeature matrix shape: {features.shape}")
        print(f"Date range: {features['end_period_dt'].min()} → {features['end_period_dt'].max()}")
        print(f"\nFeature columns:\n  {chr(10).join(get_feature_names())}")
        print(f"\nTarget stats:\n{features['target_realized_vol'].describe()}")
