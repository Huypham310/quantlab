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

## Strategies and walk-forward evaluation

`quantlab.backtest.strategies` has four rule families, each a function of past closes only (a test shocks future prices and checks no past weight moves):

| Strategy | Rule | Parameter grid |
|---|---|---|
| `sma_cross` | long while the fast SMA is above the slow one | fast 12/24/48, slow 96/168/336 |
| `momentum` | long while the price is above where it was N bars ago | N = 24 ... 720 |
| `donchian` | long on a new N-bar high, out on a new N-bar low | N = 48 ... 720 |
| `zscore_reversion` | long below -k sigma of the rolling mean, short above +k, flat at the mean | window 24/72/168, k 1.5/2/2.5 |

Each can be wrapped in **volatility targeting** (scale the position to a 30% annualized target from the last week of returns, capped at 1x) and is traded with a **no-trade band**: the engine only rebalances when the held weight drifts more than 0.10 from the target. Without the band, the volatility-scaled versions rebalanced a sliver every hour (8,000-15,000 trades in two years); with it they trade 120-580 times.

**Walk-forward**: parameters are chosen on a rolling 1-year window by Sharpe, then traded on the next 3 months, and the out-of-sample pieces are stitched into one curve. "IS" below is the flattering number: the best parameters picked on the whole history and scored on the same bars.

```bash
python -m quantlab.backtest.research --symbol ETH-USD
```

Out of sample Nov 2024 - Nov 2026 (17,510 hourly bars), 15 bps per trade:

| | BTC IS Sharpe | BTC WF Sharpe | BTC WF return | ETH IS Sharpe | ETH WF Sharpe | ETH WF return | ETH WF max DD |
|---|---:|---:|---:|---:|---:|---:|---:|
| sma_cross | 0.09 | -0.33 | -24.8% | 0.50 | -0.71 | -55.7% | 71.8% |
| momentum | -0.10 | -0.92 | -45.1% | 1.00 | **0.66** | **+46.6%** | 47.5% |
| donchian | 0.64 | -0.33 | -26.0% | 1.30 | 0.05 | -12.6% | 61.0% |
| donchian + vol target | 0.50 | -0.39 | -20.1% | 0.50 | **0.53** | **+20.3%** | **32.8%** |
| zscore_reversion | -0.06 | -0.58 | -42.1% | -0.28 | -0.86 | -70.0% | 77.6% |
| buy and hold | | 0.36 | +12.7% | | 0.27 | -7.5% | 69.2% |

What this says:

- **In-sample numbers lie.** ETH Donchian looks like a Sharpe 1.30 strategy with hindsight; traded walk-forward it is 0.05.
- **Nothing beats holding BTC** out of sample after costs; on ETH, momentum and volatility-targeted Donchian do, with roughly half of buy-and-hold's drawdown for the latter.
- **Volatility targeting mostly buys smaller drawdowns**, not higher returns.

These are the baselines the ML signals in the next phase have to beat.

## ML signals

`quantlab.model` predicts the sign of the next 24-hour return from 19 backward-looking features (returns over 1 h - 1 week, realized volatility and its ratio, z-scores against 1-day / 1-week means, position in the 24 h range, volume z-score, hour / weekday as sin-cos):

- **Purged walk-forward:** same rolling 1-year train / 3-month test windows as above. A label covers 24 bars, so the 24 rows before every boundary (train -> validation -> test) are dropped; otherwise the model trains on returns that overlap the test period. A test checks every fold's last training label ends before its validation and test rows.
- **LightGBM**, tuned per fold with Optuna (TPE, 20 trials, early stopping on validation log loss) over a deliberately regularized space (4-32 leaves, at least 50 rows per leaf).
- **LSTM** (PyTorch, 1 layer) over the last 48 bars of standardized features, tuned per fold with Optuna (6 trials: hidden 16/32/64, learning rate, dropout), early stopping on validation loss.
- P(up) is averaged over the last 12 hours and traded long-only (long when > 0.5), through the same Numba backtester and 15 bps costs.

```bash
pip install -e "backend[dev,ml]"
python -m quantlab.model.run --symbol ETH-USD --model lstm --trials 6
```

Out of sample Nov 2024 - Nov 2026 (17,341 bars; starts a week later than the table above because of the feature warm-up, which changes buy-and-hold noticeably for BTC):

| | AUC | Rank IC | Sharpe | Return | Sharpe before costs | Return before costs | Position changes |
|---|---:|---:|---:|---:|---:|---:|---:|
| BTC LightGBM | 0.501 | +0.003 | -0.35 | -34.2% | 0.24 | +3.8% | 304 |
| BTC LSTM | 0.520 | +0.030 | -0.35 | -30.7% | **0.48** | **+23.3%** | 384 |
| BTC buy and hold | | | 0.22 | -0.5% | | | 1 |
| ETH LightGBM | 0.515 | +0.013 | -0.82 | -71.1% | -0.33 | -50.2% | 362 |
| ETH LSTM | 0.538 | +0.061 | -0.24 | -38.2% | **0.46** | **+22.6%** | 457 |
| ETH buy and hold | | | 0.08 | -28.3% | | | 1 |

What this says:

- **LightGBM finds nothing** (AUC 0.50-0.52). Its splits lean on volatility features (1-week realized vol is 20-32% of the gain), which predict how much the price moves, not which way.
- **The LSTM has a small real edge** (AUC 0.52-0.54, positive rank IC on both coins): before costs it beats buy-and-hold on both, with a Sharpe around 0.47.
- **Costs eat all of it.** 380-460 full position changes in two years at 15 bps each is a 57-69% drag, more than the edge. The fix is to trade less (a wider dead zone around 0.5, longer smoothing, sizing by confidence), but choosing those on this out-of-sample period would be fitting to it, so that's left for a separate validation period.

Reports with per-fold parameters and metrics are written to `backend/reports/`.

## API

FastAPI for requests, Celery (Redis broker + result backend) for anything that takes more than a moment:

```bash
uvicorn quantlab.api.app:app --port 8000
celery -A quantlab.worker worker --loglevel info --pool solo     # --pool solo on Windows
```

| Endpoint | What |
|---|---|
| `GET /health` | API + database check |
| `GET /symbols` | stored symbol / timeframe pairs with bar counts and date range (cached 5 s) |
| `GET /bars/{symbol}?timeframe=1h&limit=500&end=...` | newest N bars, oldest first |
| `GET /strategies` | strategy library with parameters and walk-forward grids |
| `POST /backtests` | queue a backtest (strategy + params, vol target, no-trade band, costs) -> `202 {job_id}` |
| `POST /models/train` | queue a walk-forward LightGBM / LSTM training run -> `202 {job_id}` |
| `GET /jobs/{id}` | `PENDING` / `STARTED` / `PROGRESS` (folds done so far) / `SUCCESS` (result) / `FAILURE` (error) |
| `WS /ws/jobs/{id}` | pushes the job status whenever it changes, closes when it's done |
| `WS /ws/trades/{symbol}?since=$` | live trades for one symbol, read straight from the gateway's Redis Stream |

- Requests are validated before anything is queued: unknown strategies, unknown or missing strategy parameters, out-of-range costs or bands are a `422`, not a job that fails later.
- The training job reports progress per walk-forward fold (`PROGRESS` with each fold's AUC / IC), so the dashboard can show it live over the job WebSocket.
- Equity curves come back downsampled to at most 500 points; NaNs (e.g. AUC of a one-class fold) become `null`.
- Tests run the whole flow in-process: Celery in eager mode with an in-memory result backend, SQLite for the bars, fakeredis for the trade stream.

**Load:** `python -m quantlab.api.loadtest` simulates clients opening a chart (80% "last 500 hourly bars", 20% symbol / strategy lists). One uvicorn worker on the laptop, TimescaleDB in Docker:

| Concurrent clients | p50 | p95 | Throughput |
|---:|---:|---:|---:|
| 10 | 46 ms | 91 ms | 204 req/s |
| 48 (4 client processes) | 235 ms | 310 ms | ~200 req/s |

A single request for 500 bars takes ~10 ms (3.4 ms of it is the SQL round trip into Docker), so one Python worker tops out around 200 req/s and extra clients just queue. Two things learned measuring this: a single-process `httpx` load generator saturates at about the same rate, so the client has to be spread over several processes before the numbers mean anything; and the 500-bar pydantic models I suspected cost only 0.9 ms, while caching `/symbols` (a full-table `GROUP BY`) took it from 17 ms to 1.3 ms. Scaling further means more uvicorn workers behind a Linux host (multi-worker uvicorn on Windows stalled requests in my runs).

## License

MIT
