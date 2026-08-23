# Kalshi Perpetual Futures — Short-Term Volatility Forecast

**Can short-term price volatility in crypto perpetual futures be predicted using
funding rates, volume signals, and market microstructure features?**

This project builds an end-to-end ML pipeline that ingests data from Kalshi's
CFTC-regulated Perpetual Futures exchange, engineers predictive features, and
trains forecasting models to predict near-term volatility and directional price
movement for BTC perpetual futures.

## Project Structure

```
├── src/
│   ├── config.py          # Centralized settings & environment config
│   ├── ingestion.py       # Kalshi API client (RSA-PSS auth, retries, pagination)
│   ├── database.py        # SQLite schema (SQLAlchemy ORM) & query helpers
│   ├── features.py        # Feature engineering (rolling vol, returns, funding signals)
│   ├── model.py           # Model training, evaluation, backtesting
│   └── dashboard.py       # Plotly Dash visualization app
├── data/                  # Local SQLite DB & cached data (gitignored)
├── notebooks/             # Exploratory analysis & prototyping
├── tests/                 # Unit & integration tests
├── models/                # Serialized model artifacts (gitignored)
├── .env.example           # Template for API credentials
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
cd kalshi-perp-forecast
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure API credentials

1. Create a Kalshi demo account at [demo.kalshi.co](https://demo.kalshi.co)
2. Generate an API key (Account → API Keys)
3. Save the private key file to `./keys/kalshi_private.key`
4. Copy `.env.example` → `.env` and fill in your Key ID:

```bash
cp .env.example .env
# Edit .env with your KALSHI_API_KEY_ID
```

### 3. Pull data

```bash
python -m src.ingestion
```

This will discover available markets and pull the latest candlestick data into
the local SQLite database at `data/kalshi_perps.db`.

### 4. Run the pipeline

```bash
python -m src.features     # Engineer features from raw data
python -m src.model        # Train & evaluate forecasting models
python -m src.dashboard    # Launch the results dashboard
```

## Data Sources

All data comes from **Kalshi's Perpetual Futures REST API** (demo environment):

| Endpoint | Data | Use |
|----------|------|-----|
| `/margin/markets/{ticker}/candlesticks` | OHLCV + bid/ask + OI | Core price features |
| `/margin/trades` | Individual trades | Tick-level volume/activity signals |
| `/margin/funding_rates/historical` | Periodic funding rates | Premium/discount signal |

## Methodology

1. **Baseline**: Naive forecast (last value) & simple moving-average
2. **Gradient Boosting**: LightGBM with engineered features (rolling vol,
   returns, funding rate delta, volume spikes, hour-of-day)
3. **Evaluation**: Time-series cross-validation with expanding window,
   no lookahead bias. Metrics: MAE, RMSE, directional accuracy
4. **Future work**: LSTM / Temporal Fusion Transformer for sequence modeling

## Key Design Decisions

- **No order execution** — this is analysis/forecasting only
- **RSA-PSS authentication** — follows Kalshi's signature-based auth
- **Idempotent ingestion** — duplicate records are skipped on re-runs
- **Time-based train/test split** — prevents lookahead contamination

## License

MIT — for educational/portfolio purposes only. Not financial advice.
