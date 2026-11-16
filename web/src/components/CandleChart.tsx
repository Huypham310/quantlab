"use client";

import {
  CandlestickSeries,
  ColorType,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type Time,
} from "lightweight-charts";
import { useEffect, useRef } from "react";
import type { Bar } from "@/lib/api";

function toCandle(b: Bar): CandlestickData<Time> {
  // lightweight-charts wants unix seconds for UTCTimestamp
  const t = Math.floor(new Date(b.ts).getTime() / 1000) as Time;
  return { time: t, open: b.open, high: b.high, low: b.low, close: b.close };
}

export default function CandleChart({ bars }: { bars: Bar[] }) {
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Candlestick"> | null>(null);

  useEffect(() => {
    if (!host.current) return;
    const c = createChart(host.current, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: "#fafaf9" },
        textColor: "#52525b",
        fontFamily: "IBM Plex Sans, ui-sans-serif, system-ui",
      },
      grid: {
        vertLines: { color: "#e4e4e7" },
        horzLines: { color: "#e4e4e7" },
      },
      rightPriceScale: { borderColor: "#d4d4d8" },
      timeScale: { borderColor: "#d4d4d8", timeVisible: true },
    });
    const s = c.addSeries(CandlestickSeries, {
      upColor: "#0f766e",
      downColor: "#b91c1c",
      borderUpColor: "#0f766e",
      borderDownColor: "#b91c1c",
      wickUpColor: "#0f766e",
      wickDownColor: "#b91c1c",
    });
    chart.current = c;
    series.current = s;
    return () => {
      c.remove();
      chart.current = null;
      series.current = null;
    };
  }, []);

  useEffect(() => {
    if (!series.current) return;
    series.current.setData(bars.map(toCandle));
    chart.current?.timeScale().fitContent();
  }, [bars]);

  return <div ref={host} className="h-[420px] w-full rounded-lg border border-zinc-200" />;
}
