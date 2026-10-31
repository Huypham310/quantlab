"""Bar storage in Postgres/TimescaleDB (SQLite works too, for tests and quick local runs)."""

from datetime import UTC, datetime

import polars as pl
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from quantlab.data.bars import BAR_SCHEMA, empty_bars

metadata = MetaData()

bars = Table(
    "bars",
    metadata,
    Column("source", String(16), primary_key=True),
    Column("symbol", String(32), primary_key=True),
    Column("timeframe", String(8), primary_key=True),
    Column("ts", DateTime(timezone=True), primary_key=True),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", Float, nullable=False),
)


def make_engine(url: str) -> Engine:
    return create_engine(url, pool_pre_ping=True)


def init_db(engine: Engine) -> bool:
    """Create tables; on TimescaleDB also turn `bars` into a hypertable. Returns True if it did."""
    metadata.create_all(engine)
    if engine.dialect.name != "postgresql":
        return False
    with engine.begin() as conn:
        has_timescale = conn.execute(
            text("SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb'")
        ).first()
        if not has_timescale:
            return False
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        # chia chunk theo 30 ngày, đủ nhỏ cho dữ liệu 1h mà không quá nhiều chunk
        conn.execute(
            text(
                "SELECT create_hypertable('bars', 'ts', chunk_time_interval => INTERVAL '30 days',"
                " if_not_exists => TRUE, migrate_data => TRUE)"
            )
        )
    return True


def upsert_bars(engine: Engine, df: pl.DataFrame, source: str, symbol: str, timeframe: str) -> int:
    """Insert bars, skipping ones already stored, so re-running an ingest is safe."""
    if df.is_empty():
        return 0
    rows = (
        df.with_columns(
            pl.lit(source).alias("source"),
            pl.lit(symbol).alias("symbol"),
            pl.lit(timeframe).alias("timeframe"),
        )
        .select(["source", "symbol", "timeframe", *BAR_SCHEMA])
        .to_dicts()
    )
    insert = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    stmt = insert(bars).on_conflict_do_nothing(
        index_elements=["source", "symbol", "timeframe", "ts"]
    )
    with engine.begin() as conn:
        # chunk để không vượt giới hạn số tham số của sqlite
        for i in range(0, len(rows), 5000):
            conn.execute(stmt, rows[i : i + 5000])
    return len(rows)


def load_bars(
    engine: Engine,
    source: str,
    symbol: str,
    timeframe: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pl.DataFrame:
    """Read bars back as a canonical frame, oldest first."""
    query = select(
        bars.c.ts, bars.c.open, bars.c.high, bars.c.low, bars.c.close, bars.c.volume
    ).where(bars.c.source == source, bars.c.symbol == symbol, bars.c.timeframe == timeframe)
    if start is not None:
        query = query.where(bars.c.ts >= start)
    if end is not None:
        query = query.where(bars.c.ts < end)
    with engine.connect() as conn:
        rows = conn.execute(query.order_by(bars.c.ts)).all()
    if not rows:
        return empty_bars()
    df = pl.DataFrame([tuple(r) for r in rows], schema=list(BAR_SCHEMA), orient="row")
    # sqlite trả datetime không có timezone, gắn lại UTC
    ts = df["ts"]
    if ts.dtype == pl.Datetime and ts.dtype.time_zone is None:  # type: ignore[union-attr]
        df = df.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    return df.cast(BAR_SCHEMA)


def latest_ts(engine: Engine, source: str, symbol: str, timeframe: str) -> datetime | None:
    """Newest stored bar time, so incremental ingests only fetch what's missing."""
    with engine.connect() as conn:
        value = conn.execute(
            select(bars.c.ts)
            .where(bars.c.source == source, bars.c.symbol == symbol, bars.c.timeframe == timeframe)
            .order_by(bars.c.ts.desc())
            .limit(1)
        ).scalar()
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=UTC)
