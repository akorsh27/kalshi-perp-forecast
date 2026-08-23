"""
SQLite database schema and helpers for storing Kalshi perps time-series data.

Tables:
    candlesticks    — OHLCV price data at various intervals
    trades          — Individual public trade records
    funding_rates   — Historical funding rate snapshots
"""

import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    UniqueConstraint,
    Index,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import DATABASE_PATH, LOG_LEVEL

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

Base = declarative_base()


class Candlestick(Base):
    """OHLCV candlestick data for a margin market at a given interval."""

    __tablename__ = "candlesticks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    interval_minutes = Column(Integer, nullable=False)  # 1, 60, or 1440
    end_period_ts = Column(Integer, nullable=False)  # Unix timestamp
    end_period_dt = Column(DateTime, nullable=False)

    # Price OHLC (trade prices)
    price_open = Column(Float, nullable=True)
    price_high = Column(Float, nullable=True)
    price_low = Column(Float, nullable=True)
    price_close = Column(Float, nullable=True)
    price_mean = Column(Float, nullable=True)  # VWAP
    price_previous = Column(Float, nullable=True)

    # Bid OHLC (best bid quotes)
    bid_open = Column(Float, nullable=True)
    bid_high = Column(Float, nullable=True)
    bid_low = Column(Float, nullable=True)
    bid_close = Column(Float, nullable=True)

    # Ask OHLC (best ask quotes)
    ask_open = Column(Float, nullable=True)
    ask_high = Column(Float, nullable=True)
    ask_low = Column(Float, nullable=True)
    ask_close = Column(Float, nullable=True)

    # Volume & Open Interest
    volume = Column(Float, nullable=True)
    volume_notional_usd = Column(Float, nullable=True)
    open_interest = Column(Float, nullable=True)
    open_interest_notional_usd = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("ticker", "interval_minutes", "end_period_ts", name="uq_candle"),
        Index("ix_candle_lookup", "ticker", "interval_minutes", "end_period_ts"),
    )


class Trade(Base):
    """Individual public trade records."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String, nullable=False, unique=True)
    ticker = Column(String, nullable=False)
    timestamp = Column(Integer, nullable=False)  # Unix ms
    timestamp_dt = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    count = Column(Float, nullable=True)  # contract count
    side = Column(String, nullable=True)  # "buy" or "sell" if available

    __table_args__ = (
        Index("ix_trade_ticker_ts", "ticker", "timestamp"),
    )


class FundingRate(Base):
    """Historical funding rate records."""

    __tablename__ = "funding_rates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String, nullable=False)
    funding_time = Column(Integer, nullable=False)  # Unix timestamp
    funding_time_dt = Column(DateTime, nullable=False)
    funding_rate = Column(Float, nullable=False)
    mark_price = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("ticker", "funding_time", name="uq_funding"),
        Index("ix_funding_lookup", "ticker", "funding_time"),
    )


class Database:
    """
    Manages SQLite connection and provides insert/query helpers.

    Usage:
        db = Database()
        db.insert_candlesticks(candle_records, ticker="BTC-USD", interval=60)
        df = db.get_candlesticks_df(ticker="BTC-USD", interval=60)
    """

    def __init__(self, db_path: str = str(DATABASE_PATH)):
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        logger.info("Database initialized at %s", db_path)

    def insert_candlesticks(
        self, raw_candles: list[dict], ticker: str, interval_minutes: int
    ) -> int:
        """
        Parse and insert candlestick records (upsert — skip duplicates).

        Returns:
            Number of new records inserted.
        """
        session = self.Session()
        inserted = 0

        try:
            for c in raw_candles:
                end_ts = c["end_period_ts"]

                existing = (
                    session.query(Candlestick)
                    .filter_by(
                        ticker=ticker,
                        interval_minutes=interval_minutes,
                        end_period_ts=end_ts,
                    )
                    .first()
                )
                if existing:
                    continue

                price = c.get("price", {})
                bid = c.get("bid", {})
                ask = c.get("ask", {})

                record = Candlestick(
                    ticker=ticker,
                    interval_minutes=interval_minutes,
                    end_period_ts=end_ts,
                    end_period_dt=datetime.fromtimestamp(end_ts, tz=timezone.utc),
                    price_open=_parse_dollar(price.get("open")),
                    price_high=_parse_dollar(price.get("high")),
                    price_low=_parse_dollar(price.get("low")),
                    price_close=_parse_dollar(price.get("close")),
                    price_mean=_parse_dollar(price.get("mean")),
                    price_previous=_parse_dollar(price.get("previous")),
                    bid_open=_parse_dollar(bid.get("open")),
                    bid_high=_parse_dollar(bid.get("high")),
                    bid_low=_parse_dollar(bid.get("low")),
                    bid_close=_parse_dollar(bid.get("close")),
                    ask_open=_parse_dollar(ask.get("open")),
                    ask_high=_parse_dollar(ask.get("high")),
                    ask_low=_parse_dollar(ask.get("low")),
                    ask_close=_parse_dollar(ask.get("close")),
                    volume=_parse_count(c.get("volume")),
                    volume_notional_usd=_parse_dollar(c.get("volume_notional_value_dollars")),
                    open_interest=_parse_count(c.get("open_interest")),
                    open_interest_notional_usd=_parse_dollar(
                        c.get("open_interest_notional_value_dollars")
                    ),
                )
                session.add(record)
                inserted += 1

            session.commit()
            logger.info(
                "Inserted %d new candlesticks (%s, %dm)", inserted, ticker, interval_minutes
            )
        except Exception as e:
            session.rollback()
            logger.error("Failed to insert candlesticks: %s", e)
            raise
        finally:
            session.close()

        return inserted

    def insert_trades(self, raw_trades: list[dict], ticker: str) -> int:
        """Insert trade records, skipping duplicates by trade_id."""
        session = self.Session()
        inserted = 0

        try:
            for t in raw_trades:
                trade_id = t.get("id", t.get("trade_id", ""))
                if not trade_id:
                    continue

                existing = session.query(Trade).filter_by(trade_id=trade_id).first()
                if existing:
                    continue

                ts = t.get("created_time") or t.get("timestamp") or t.get("ts", 0)
                # Handle ISO string timestamps
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    ts_unix = int(dt.timestamp() * 1000)
                else:
                    ts_unix = ts
                    dt = datetime.fromtimestamp(ts_unix / 1000, tz=timezone.utc)

                record = Trade(
                    trade_id=trade_id,
                    ticker=ticker,
                    timestamp=ts_unix,
                    timestamp_dt=dt,
                    price=_parse_dollar(t.get("price", t.get("yes_price", "0"))),
                    count=_parse_count(t.get("count", t.get("contracts", None))),
                    side=t.get("taker_side", t.get("side")),
                )
                session.add(record)
                inserted += 1

            session.commit()
            logger.info("Inserted %d new trades for %s", inserted, ticker)
        except Exception as e:
            session.rollback()
            logger.error("Failed to insert trades: %s", e)
            raise
        finally:
            session.close()

        return inserted

    def insert_funding_rates(self, raw_rates: list[dict], ticker: str) -> int:
        """Insert funding rate records, skipping duplicates."""
        session = self.Session()
        inserted = 0

        try:
            for r in raw_rates:
                ft = r.get("funding_time", "")
                if isinstance(ft, str):
                    dt = datetime.fromisoformat(ft.replace("Z", "+00:00"))
                    ft_unix = int(dt.timestamp())
                else:
                    ft_unix = ft
                    dt = datetime.fromtimestamp(ft_unix, tz=timezone.utc)

                existing = (
                    session.query(FundingRate)
                    .filter_by(ticker=ticker, funding_time=ft_unix)
                    .first()
                )
                if existing:
                    continue

                record = FundingRate(
                    ticker=ticker,
                    funding_time=ft_unix,
                    funding_time_dt=dt,
                    funding_rate=float(r.get("funding_rate", 0)),
                    mark_price=_parse_dollar(r.get("mark_price")),
                )
                session.add(record)
                inserted += 1

            session.commit()
            logger.info("Inserted %d new funding rates for %s", inserted, ticker)
        except Exception as e:
            session.rollback()
            logger.error("Failed to insert funding rates: %s", e)
            raise
        finally:
            session.close()

        return inserted

    # ─── Query Helpers (return DataFrames) ───────────────────────────────

    def get_candlesticks_df(
        self,
        ticker: str = "BTC-USD",
        interval_minutes: int = 60,
        start_dt: datetime = None,
        end_dt: datetime = None,
    ) -> pd.DataFrame:
        """Query candlesticks as a pandas DataFrame, sorted by time."""
        query = f"""
            SELECT * FROM candlesticks
            WHERE ticker = :ticker AND interval_minutes = :interval
            {"AND end_period_dt >= :start" if start_dt else ""}
            {"AND end_period_dt <= :end" if end_dt else ""}
            ORDER BY end_period_ts ASC
        """
        params = {"ticker": ticker, "interval": interval_minutes}
        if start_dt:
            params["start"] = start_dt.isoformat()
        if end_dt:
            params["end"] = end_dt.isoformat()

        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
        return df

    def get_funding_rates_df(self, ticker: str = "BTC-USD") -> pd.DataFrame:
        """Query funding rates as a pandas DataFrame."""
        query = """
            SELECT * FROM funding_rates
            WHERE ticker = :ticker
            ORDER BY funding_time ASC
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"ticker": ticker})
        return df

    def get_trades_df(self, ticker: str = "BTC-USD") -> pd.DataFrame:
        """Query trades as a pandas DataFrame."""
        query = """
            SELECT * FROM trades
            WHERE ticker = :ticker
            ORDER BY timestamp ASC
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"ticker": ticker})
        return df


# ─── Helpers ─────────────────────────────────────────────────────────────


def _parse_dollar(value) -> float | None:
    """Convert FixedPointDollars string (e.g. '97234.56') to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _parse_count(value) -> float | None:
    """Convert FixedPointCount string (e.g. '10.00') to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
