# QuantLab

Backtesting and paper-trading platform for equities and crypto, built as a long-running side project.

Planned pieces:

- market data ingestion (daily equities from Stooq, crypto candles from Coinbase's public API) into TimescaleDB
- a Go gateway that streams live crypto trades into Redis Streams
- a backtesting engine (vectorized + event-driven) with realistic fees and slippage
- ML signals (LightGBM, LSTM) with walk-forward validation
- FastAPI backend with WebSockets, a paper-trading order manager, and a Next.js dashboard

Everything uses free data sources and runs locally.

## Market data

Crypto only for now, from two exchanges' public APIs (no keys):

| Source | Endpoint | History |
| --- | --- | --- |
| Coinbase Exchange | `/products/{id}/candles` | full history, 300 candles per request (paged) |
| Kraken | `/0/public/OHLC` | latest 720 candles, used to cross-check Coinbase |

On 20 days of BTC-USD hourly bars the two exchanges' closes differ by 0.008% on average.

Every source is converted to one canonical bar frame (Polars, UTC timestamps), cleaned (sorted, duplicates and impossible candles dropped, gaps counted but never forward-filled), and upserted into a `bars` table that becomes a TimescaleDB hypertable on Postgres. Re-running an ingest only fetches bars newer than what's stored.

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e "backend[dev]"
docker compose up -d timescaledb
cd backend
python -m quantlab.data.ingest --symbols BTC-USD,ETH-USD --timeframe 1h --days 365
# or without Docker:
python -m quantlab.data.ingest --db sqlite:///quantlab.db --days 60
```

## License

MIT
