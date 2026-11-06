"""Celery worker for backtests and model training.

celery -A quantlab.worker worker --loglevel info --pool solo      # --pool solo on Windows
"""

from __future__ import annotations

from datetime import datetime

from celery import Celery

from quantlab.api import jobs
from quantlab.config import get_settings
from quantlab.data.store import load_bars, make_engine

settings = get_settings()
celery_app = Celery("quantlab", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_track_started=True,
    result_expires=24 * 3600,
    # một job train có thể chạy vài phút: không cho worker ôm sẵn nhiều task
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


def _bars(req: dict):
    engine = make_engine(req.get("db") or settings.database_url)
    try:
        start = datetime.fromisoformat(req["start"]) if req.get("start") else None
        end = datetime.fromisoformat(req["end"]) if req.get("end") else None
        return load_bars(
            engine, req.get("source", "coinbase"), req["symbol"], req.get("timeframe", "1h"),
            start, end,
        )  # fmt: skip
    finally:
        engine.dispose()


@celery_app.task(name="quantlab.backtest", bind=True)
def run_backtest(self, req: dict) -> dict:
    return jobs.backtest_job(_bars(req), req)


@celery_app.task(name="quantlab.train", bind=True)
def train_model(self, req: dict) -> dict:
    done: list[dict] = []

    def progress(fold: dict) -> None:
        done.append({"fold": fold["fold"], "auc": fold.get("auc"), "ic": fold.get("ic")})
        # PROGRESS is a custom state; the API's job endpoint and websocket report it as-is
        self.update_state(state="PROGRESS", meta={"folds_done": len(done), "folds": done})

    return jobs.train_job(_bars(req), req, progress)
