"""
End-to-end pipeline orchestrator.

Runs the full workflow:
    1. Discover markets & pull all available data from Kalshi
    2. Store in SQLite
    3. Engineer features
    4. Train & evaluate models
    5. (Optional) Launch dashboard

Usage:
    python run_pipeline.py              # Full pipeline
    python run_pipeline.py --ingest     # Only data ingestion
    python run_pipeline.py --train      # Only feature engineering + training
    python run_pipeline.py --dashboard  # Only launch dashboard
"""

import argparse
import logging
import sys
import time

from src.config import LOG_LEVEL, DEFAULT_TICKER, INTERVAL_1HR

logger = logging.getLogger("pipeline")
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_ingestion(ticker: str = None) -> str:
    """
    Pull data from Kalshi API and store in database.
    Returns the discovered/used ticker.
    """
    from src.ingestion import KalshiClient
    from src.database import Database

    client = KalshiClient()
    db = Database()

    # Discover ticker dynamically
    if ticker is None:
        logger.info("Discovering available markets...")
        markets = client.get_markets()
        if not markets:
            logger.error("No markets found. Check API credentials and environment.")
            sys.exit(1)

        logger.info("Available markets:")
        for m in markets:
            logger.info("  %s — %s", m.get("ticker"), m.get("title", m.get("subtitle", "")))

        # Find BTC market
        btc_markets = [m for m in markets if "BTC" in m.get("ticker", "").upper()]
        if btc_markets:
            ticker = btc_markets[0]["ticker"]
            logger.info("Using BTC market: %s", ticker)
        else:
            ticker = markets[0]["ticker"]
            logger.info("No BTC market found, using first available: %s", ticker)

    # Pull candlesticks — go as far back as the API allows
    now = int(time.time())
    lookback_days = 90  # Start with 90 days, API will return what's available
    start_ts = now - (lookback_days * 24 * 3600)

    logger.info("Pulling hourly candlesticks (%d days lookback)...", lookback_days)
    candles = client.get_candlesticks(
        ticker=ticker,
        start_ts=start_ts,
        end_ts=now,
        period_interval=INTERVAL_1HR,
    )
    if candles:
        inserted = db.insert_candlesticks(candles, ticker=ticker, interval_minutes=60)
        logger.info("Candlesticks: %d fetched, %d new inserted", len(candles), inserted)
    else:
        logger.warning("No candlestick data returned")

    # Pull funding rates
    logger.info("Pulling historical funding rates...")
    rates = client.get_historical_funding_rates(ticker=ticker, start_ts=start_ts, end_ts=now)
    if rates:
        inserted = db.insert_funding_rates(rates, ticker=ticker)
        logger.info("Funding rates: %d fetched, %d new inserted", len(rates), inserted)
    else:
        logger.warning("No funding rate data returned")

    # Pull recent trades (last 7 days to avoid excessive pagination)
    trade_start = now - (7 * 24 * 3600)
    logger.info("Pulling trades (last 7 days)...")
    trades = client.get_trades(ticker=ticker, min_ts=trade_start, max_ts=now)
    if trades:
        inserted = db.insert_trades(trades, ticker=ticker)
        logger.info("Trades: %d fetched, %d new inserted", len(trades), inserted)
    else:
        logger.warning("No trade data returned")

    return ticker


def run_training(ticker: str = DEFAULT_TICKER):
    """Run feature engineering and model training."""
    from src.database import Database
    from src.features import build_feature_matrix
    from src.model import backtest, train_final_model

    db = Database()
    candles = db.get_candlesticks_df(ticker=ticker, interval_minutes=60)
    funding = db.get_funding_rates_df(ticker=ticker)

    if candles.empty:
        logger.error("No candlestick data found. Run ingestion first.")
        sys.exit(1)

    logger.info("Building feature matrix from %d candle rows...", len(candles))
    features = build_feature_matrix(candles, funding)
    logger.info("Feature matrix: %d rows × %d columns", *features.shape)

    if len(features) < 50:
        logger.warning("Only %d rows — insufficient for backtest. Training on all data.", len(features))
        model = train_final_model(features)
        return

    logger.info("Running time-series backtest...")
    results = backtest(features)

    print("\n" + "=" * 60)
    print("  BACKTEST RESULTS")
    print("=" * 60)
    for name, m in results["metrics"].items():
        print(f"  {name:12s} | MAE: {m['mae']:.6f} | RMSE: {m['rmse']:.6f} | "
              f"Dir.Acc: {m['directional_accuracy']:.1%} | N={m['n_predictions']}")
    print("=" * 60)

    if not results["feature_importance"].empty:
        print("\n  TOP FEATURES (by gain):")
        for _, row in results["feature_importance"].head(10).iterrows():
            print(f"    {row['feature']:25s} {row['importance']:.1f}")

    # Train final model on full data
    logger.info("Training final model on full dataset...")
    train_final_model(features)

    return results


def run_dashboard(ticker: str = DEFAULT_TICKER):
    """Launch the Dash visualization dashboard."""
    from src.database import Database
    from src.features import build_feature_matrix
    from src.model import backtest
    from src.dashboard import build_dash_app

    db = Database()
    candles = db.get_candlesticks_df(ticker=ticker, interval_minutes=60)
    funding = db.get_funding_rates_df(ticker=ticker)

    if candles.empty:
        logger.error("No data. Run: python run_pipeline.py --ingest")
        sys.exit(1)

    features = build_feature_matrix(candles, funding)
    results = backtest(features)

    app = build_dash_app(
        predictions_df=results["predictions"],
        metrics=results["metrics"],
        feature_importance=results["feature_importance"],
    )
    print("\n  Dashboard running at http://127.0.0.1:8050")
    print("  Press Ctrl+C to stop.\n")
    app.run_server(debug=False, port=8050)


def main():
    parser = argparse.ArgumentParser(
        description="Kalshi BTC Perps Volatility Forecast Pipeline"
    )
    parser.add_argument("--ingest", action="store_true", help="Run data ingestion only")
    parser.add_argument("--train", action="store_true", help="Run training/evaluation only")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard only")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Market ticker (auto-discovered if omitted)")
    args = parser.parse_args()

    # If no specific flag, run full pipeline
    run_all = not (args.ingest or args.train or args.dashboard)

    ticker = args.ticker

    if args.ingest or run_all:
        logger.info("=" * 40 + " INGESTION " + "=" * 40)
        ticker = run_ingestion(ticker)

    if args.train or run_all:
        logger.info("=" * 40 + " TRAINING " + "=" * 40)
        if ticker is None:
            ticker = DEFAULT_TICKER
        run_training(ticker)

    if args.dashboard or run_all:
        if ticker is None:
            ticker = DEFAULT_TICKER
        run_dashboard(ticker)


if __name__ == "__main__":
    main()
