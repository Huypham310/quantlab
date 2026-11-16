"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import CandleChart from "@/components/CandleChart";
import { getBars, listSymbols, watchTrades, type TradeTick } from "@/lib/api";

export default function LivePage() {
  const symbols = useQuery({ queryKey: ["symbols"], queryFn: listSymbols });
  const options = useMemo(() => {
    const rows = symbols.data ?? [];
    // prefer 1h crypto pairs for the default chart
    return rows.filter((s) => s.timeframe === "1h");
  }, [symbols.data]);
  const [symbol, setSymbol] = useState("BTC-USD");
  const [last, setLast] = useState<TradeTick | null>(null);

  useEffect(() => {
    if (options.length && !options.some((s) => s.symbol === symbol)) {
      setSymbol(options[0].symbol);
    }
  }, [options, symbol]);

  const bars = useQuery({
    queryKey: ["bars", symbol, "1h"],
    queryFn: () => getBars(symbol, { timeframe: "1h", limit: 500 }),
    enabled: Boolean(symbol),
  });

  useEffect(() => {
    setLast(null);
    return watchTrades(symbol, (t) => setLast(t));
  }, [symbol]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-display text-3xl text-zinc-900">Live</h1>
          <p className="mt-1 text-sm text-zinc-500">
            Hourly candles from TimescaleDB, with live prints from the Redis trade stream.
          </p>
        </div>
        <label className="text-sm text-zinc-600">
          Symbol
          <select
            className="ml-2 rounded border border-zinc-300 bg-white px-2 py-1.5"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
          >
            {(options.length ? options : [{ symbol: "BTC-USD", timeframe: "1h" }]).map((s) => (
              <option key={`${s.symbol}-${s.timeframe}`} value={s.symbol}>
                {s.symbol}
              </option>
            ))}
          </select>
        </label>
      </div>

      {last && (
        <p className="text-sm tabular-nums text-zinc-600">
          last print{" "}
          <span className="font-medium text-zinc-900">{last.price.toFixed(2)}</span> · {last.side}{" "}
          {last.size} · {new Date(last.ts_ms).toLocaleTimeString()}
        </p>
      )}

      {bars.isLoading && <p className="text-sm text-zinc-500">Loading bars…</p>}
      {bars.isError && (
        <p className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {(bars.error as Error).message}
        </p>
      )}
      {bars.data && bars.data.length > 0 && <CandleChart bars={bars.data} />}
      {bars.data && bars.data.length === 0 && (
        <p className="text-sm text-zinc-500">No bars for this symbol yet. Run an ingest first.</p>
      )}
    </div>
  );
}
