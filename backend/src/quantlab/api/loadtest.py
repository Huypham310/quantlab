"""Small load test for the read endpoints of a running API.

    python -m quantlab.api.loadtest --url http://127.0.0.1:8000 --clients 50 --requests 40

Each simulated client loops over a mix that looks like the dashboard opening a chart: the last
500 hourly bars of a symbol, plus the symbol and strategy lists. Reports latency percentiles
over every request and the overall throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time

import httpx
import numpy as np

PATHS = [
    ("/bars/BTC-USD?limit=500", 0.4),
    ("/bars/ETH-USD?limit=500", 0.4),
    ("/symbols", 0.1),
    ("/strategies", 0.1),
]


async def client_loop(
    http: httpx.AsyncClient, n: int, seed: int, out: list[float], errors: list, mix: list
):
    rng = random.Random(seed)
    paths, weights = zip(*mix, strict=True)
    for _ in range(n):
        path = rng.choices(paths, weights)[0]
        t = time.perf_counter()
        r = await http.get(path)
        out.append(time.perf_counter() - t)
        if r.status_code != 200:
            errors.append((path, r.status_code))


async def run(
    url: str,
    clients: int,
    requests: int,
    path: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """`transport` lets tests point the load test at an in-process app (httpx.ASGITransport)."""
    mix = [(path, 1.0)] if path else PATHS
    latencies: list[float] = []
    errors: list = []
    limits = httpx.Limits(max_connections=clients, max_keepalive_connections=clients)
    async with httpx.AsyncClient(
        base_url=url, limits=limits, timeout=30, transport=transport
    ) as http:
        await http.get("/health")  # warm up the connection pool and the DB pool
        start = time.perf_counter()
        await asyncio.gather(
            *(client_loop(http, requests, i, latencies, errors, mix) for i in range(clients))
        )
        wall = time.perf_counter() - start
    ms = np.array(latencies) * 1000
    return {
        "clients": clients,
        "requests": len(latencies),
        "errors": len(errors),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "requests_per_s": len(latencies) / wall,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--clients", type=int, default=50)
    p.add_argument("--requests", type=int, default=40, help="per client")
    p.add_argument("--path", help="hit only this path instead of the dashboard mix")
    args = p.parse_args(argv)
    r = asyncio.run(run(args.url, args.clients, args.requests, args.path))
    print(
        f"{r['clients']} clients, {r['requests']} requests, {r['errors']} errors: "
        f"p50 {r['p50_ms']:.1f} ms, p95 {r['p95_ms']:.1f} ms, p99 {r['p99_ms']:.1f} ms, "
        f"{r['requests_per_s']:.0f} req/s"
    )


if __name__ == "__main__":
    main()
