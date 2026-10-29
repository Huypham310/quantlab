from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Backend settings, read from QUANTLAB_* env vars or the repo-level .env."""

    model_config = SettingsConfigDict(env_prefix="QUANTLAB_", env_file="../.env", extra="ignore")

    database_url: str = "postgresql+psycopg://quantlab:quantlab@localhost:5432/quantlab"
    redis_url: str = "redis://localhost:6379/0"
    # tick stream mà gateway Go ghi vào
    trades_stream: str = "trades"


@lru_cache
def get_settings() -> Settings:
    return Settings()
