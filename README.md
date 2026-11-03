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

## Live trade stream

```
Coinbase websocket ──> Go gateway ──XADD──> Redis Stream "trades" ──XREADGROUP──> Python consumer ──> bars (1m)
   (matches channel)    reconnect w/ backoff     capped ~500k entries     consumer group "bars-1m"      source "coinbase-ws"
```

- **Gateway** (`gateway/`, Go): subscribes to Coinbase's `matches` channel, normalizes each trade (taker side, UTC ms timestamp), and appends it to a capped Redis Stream. Dropped connections are retried with exponential backoff and jitter; a socket with no data for a minute counts as dead.
- **Consumer** (`quantlab.stream`, Python): reads the stream through a consumer group and builds 1-minute OHLCV bars on **event time** (the trade's own timestamp), with a 5 s grace period for out-of-order trades. Duplicate trade ids (Coinbase re-sends the last trade after every reconnect) are counted once.
- **At-least-once**: a message is acked only after the bar it belongs to is written. If the consumer dies, its unacked trades stay pending and are replayed on restart; writing the same bar twice is a no-op.

```bash
docker compose up -d redis timescaledb
cd gateway && go build -o bin/gateway ./cmd/gateway && ./bin/gateway -products BTC-USD,ETH-USD
cd backend && python -m quantlab.stream.consumer
python -m quantlab.stream.check --symbol BTC-USD   # stream bars vs Coinbase's own 1m candles
```

Checked live on BTC-USD and ETH-USD for 10 minutes (about 3,600 trades), with the consumer **killed with `Stop-Process -Force` halfway and restarted 40 s later**: every complete minute came out identical (open, high, low, close and volume) to Coinbase's REST 1m candles, including the minutes around the crash. The only difference was the first minute, which is partial because the gateway started mid-minute.

## Backtesting

Two engines that must agree:

- **Vectorized** (Numba `@njit`): `weights[t]` is decided at the close of bar t from data up to t and earns the move from t to t+1, so a signal can't trade on the bar it was computed from. The held weight drifts with the price between bars (fixed units, like a real account), and fees + slippage are charged on |target - held weight|.
- **Event-driven**: the strategy gets one bar at a time as a read-only view of the history and trades units against cash. It is slow, but structurally can't look ahead, so it's the reference.

On a 3,000-bar random walk with an SMA crossover (long/short, 15 bps costs) the two equity curves match to a relative 1e-14, and a Hypothesis test checks the compiled engine against a plain-Python twin on random prices and weights.

Throughput (`python -m quantlab.backtest.bench`, one core):

| Engine | Bars | Time | Bars/s |
|---|---:|---:|---:|
| Numba vectorized | 1,000,000 | 10 ms | ~100 M |
| Plain Python loop | 1,000,000 | 0.71 s | 1.4 M |
| Event-driven | 50,000 | 0.21 s | 0.24 M |

Numba is ~70x faster than the same loop in Python, which is what makes parameter sweeps and walk-forward runs cheap.

```bash
python -m quantlab.data.ingest --symbols BTC-USD,ETH-USD --timeframe 1h --days 1095
python -m quantlab.backtest.run --symbol BTC-USD --fast 24 --slow 168
```

First honest result, 3 years of hourly BTC-USD (26,270 bars, Nov 2023 - Nov 2026), 10 bps fee + 5 bps slippage:

| | Total return | CAGR | Sharpe | Max drawdown | Trades |
|---|---:|---:|---:|---:|---:|
| SMA 24/168, long or flat | -10.9% | -3.8% | 0.04 | 57.1% | 254 |
| Buy and hold | +121.6% | +30.4% | 0.80 | 53.9% | 1 |

A textbook trend rule loses to holding once costs are paid, which is the bar the strategies and models in the next phases have to clear.

## License

MIT
