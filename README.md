# QuantLab

Backtesting and paper-trading platform for equities and crypto, built as a long-running side project.

Planned pieces:

- market data ingestion (daily equities from Stooq, crypto candles from Coinbase's public API) into TimescaleDB
- a Go gateway that streams live crypto trades into Redis Streams
- a backtesting engine (vectorized + event-driven) with realistic fees and slippage
- ML signals (LightGBM, LSTM) with walk-forward validation
- FastAPI backend with WebSockets, a paper-trading order manager, and a Next.js dashboard

Everything uses free data sources and runs locally.

## License

MIT
