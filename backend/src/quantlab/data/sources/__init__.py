"""Exchange clients. Each one returns bars in the canonical schema from quantlab.data.bars."""

from quantlab.data.sources.coinbase import CoinbaseClient
from quantlab.data.sources.kraken import KrakenClient

SOURCES = {"coinbase": CoinbaseClient, "kraken": KrakenClient}

__all__ = ["SOURCES", "CoinbaseClient", "KrakenClient"]
