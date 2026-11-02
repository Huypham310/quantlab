from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quantlab.data.bars import BAR_SCHEMA
from quantlab.data.store import init_db, make_engine

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def make_bars(
    n: int = 5, start: datetime = T0, step_hours: int = 1, price: float = 100.0
) -> pl.DataFrame:
    """Simple valid hourly bars with a slowly rising price."""
    rows = []
    for i in range(n):
        p = price + i
        rows.append((start + timedelta(hours=i * step_hours), p, p + 1, p - 1, p + 0.5, 10.0 + i))
    return pl.DataFrame(rows, schema=list(BAR_SCHEMA), orient="row").cast(BAR_SCHEMA)


@pytest.fixture
def engine(tmp_path):
    eng = make_engine(f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    init_db(eng)
    yield eng
    eng.dispose()
