"use client";

import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { getJob, listStrategies, startBacktest, watchJob, type JobStatus } from "@/lib/api";

export default function BacktestPage() {
  const strategies = useQuery({ queryKey: ["strategies"], queryFn: listStrategies });
  const [symbol, setSymbol] = useState("BTC-USD");
  const [strategy, setStrategy] = useState("sma_cross");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!jobId) return;
    return watchJob(jobId, setStatus);
  }, [jobId]);

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      const created = await startBacktest({
        symbol,
        strategy,
        timeframe: "1h",
        source: "coinbase",
        fee_bps: 10,
        slippage_bps: 5,
        cash: 10_000,
      });
      setJobId(created.job_id);
      setStatus(await getJob(created.job_id));
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed");
    } finally {
      setBusy(false);
    }
  }

  const metrics = (status?.result?.metrics ?? null) as Record<string, number> | null;
  const equity = (status?.result?.equity ?? null) as { ts: string; value: number }[] | null;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl text-zinc-900">Backtest</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Kick off a Celery job and watch progress over the job WebSocket.
        </p>
      </div>

      <form
        onSubmit={onSubmit}
        className="flex flex-wrap items-end gap-3 rounded-lg border border-zinc-200 bg-white p-4"
      >
        <label className="text-sm">
          Symbol
          <input
            className="mt-1 block rounded border border-zinc-300 px-2 py-1.5"
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
          />
        </label>
        <label className="text-sm">
          Strategy
          <select
            className="mt-1 block rounded border border-zinc-300 px-2 py-1.5"
            value={strategy}
            onChange={(e) => setStrategy(e.target.value)}
          >
            {(strategies.data ?? [{ name: "sma_cross" }, { name: "buy_and_hold" }]).map((s) => (
              <option key={s.name} value={s.name}>
                {s.name}
              </option>
            ))}
          </select>
        </label>
        <button
          type="submit"
          disabled={busy}
          className="rounded bg-teal-800 px-4 py-2 text-sm font-medium text-white hover:bg-teal-900 disabled:opacity-60"
        >
          Run
        </button>
      </form>

      {error && (
        <p className="rounded border border-red-200 bg-red-50 p-3 text-sm text-red-700">{error}</p>
      )}

      {status && (
        <div className="space-y-4 rounded-lg border border-zinc-200 bg-white p-4">
          <p className="text-sm text-zinc-600">
            job <span className="font-mono text-xs">{status.job_id}</span> ·{" "}
            <span className="font-medium text-zinc-900">{status.status}</span>
          </p>
          {status.error && <p className="text-sm text-red-700">{status.error}</p>}
          {metrics && (
            <dl className="grid gap-3 sm:grid-cols-4">
              {["sharpe", "max_drawdown", "total_return", "trades"].map((k) => (
                <div key={k}>
                  <dt className="text-xs uppercase tracking-wide text-zinc-500">{k}</dt>
                  <dd className="mt-1 font-medium tabular-nums">
                    {typeof metrics[k] === "number" ? metrics[k].toFixed?.(4) ?? metrics[k] : "—"}
                  </dd>
                </div>
              ))}
            </dl>
          )}
          {equity && equity.length > 1 && (
            <p className="text-sm text-zinc-500">
              equity curve: {equity.length} points,{" "}
              {equity[0].value.toFixed(0)} → {equity[equity.length - 1].value.toFixed(0)}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
