"""Download bars from an exchange, clean them, and store them.

Usage:
    python -m quantlab.data.ingest --symbols BTC-USD,ETH-USD --timeframe 1h --days 365
    python -m quantlab.data.ingest --db sqlite:///quantlab.db ...   # no Docker needed
"""

import argparse
import logging
from datetime import UTC, datetime, timedelta

from quantlab.config import get_settings
from quantlab.data.bars import TIMEFRAMES, clean_bars
from quantlab.data.sources import SOURCES
from quantlab.data.store import init_db, latest_ts, make_engine, upsert_bars

log = logging.getLogger("quantlab.ingest")


def ingest(engine, source_name: str, symbols: list[str], timeframe: str, days: int) -> dict:
    """Incremental ingest: resumes from the newest stored bar instead of re-downloading."""
    client = SOURCES[source_name]()
    # chỉ lấy tới nến đã đóng gần nhất
    step = TIMEFRAMES[timeframe]
    now = datetime.now(UTC)
    end = datetime.fromtimestamp(int(now.timestamp()) // step * step, UTC)

    summary = {}
    for symbol in symbols:
        last = latest_ts(engine, source_name, symbol, timeframe)
        start = last + timedelta(seconds=step) if last else end - timedelta(days=days)
        if start >= end:
            log.info("%s %s already up to date", source_name, symbol)
            summary[symbol] = 0
            continue
        raw = client.fetch_bars(symbol, timeframe, start, end)
        bars, report = clean_bars(raw, timeframe)
        upsert_bars(engine, bars, source_name, symbol, timeframe)
        log.info(
            "%s %s: %d fetched, %d dup, %d invalid, %d gaps",
            source_name,
            symbol,
            report.rows_in,
            report.duplicates,
            report.invalid,
            report.gaps,
        )
        summary[symbol] = report.rows_out
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--source", choices=sorted(SOURCES), default="coinbase")
    parser.add_argument("--symbols", default="BTC-USD,ETH-USD")
    parser.add_argument("--timeframe", choices=sorted(TIMEFRAMES), default="1h")
    parser.add_argument("--days", type=int, default=365, help="history to fetch on first run")
    parser.add_argument(
        "--db", default=None, help="database url (defaults to QUANTLAB_DATABASE_URL)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # httpx logs every request at INFO, way too noisy with 300-candle pages
    logging.getLogger("httpx").setLevel(logging.WARNING)
    engine = make_engine(args.db or get_settings().database_url)
    if init_db(engine):
        log.info("bars is a TimescaleDB hypertable")
    summary = ingest(engine, args.source, args.symbols.split(","), args.timeframe, args.days)
    log.info("done: %s", summary)


if __name__ == "__main__":
    main()
