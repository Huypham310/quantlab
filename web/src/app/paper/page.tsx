"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  cancelPaperOrder,
  getPaperAccount,
  listPaperOrders,
  submitPaperOrder,
} from "@/lib/api";

export default function PaperPage() {
  const qc = useQueryClient();
  const account = useQuery({
    queryKey: ["paper-account"],
    queryFn: getPaperAccount,
    refetchInterval: 2_000,
  });
  const orders = useQuery({
    queryKey: ["paper-orders"],
    queryFn: () => listPaperOrders(),
    refetchInterval: 2_000,
  });

  const [symbol, setSymbol] = useState("BTC-USD");
  const [side, setSide] = useState<"buy" | "sell">("buy");
  const [qty, setQty] = useState("0.01");
  const [message, setMessage] = useState<string | null>(null);

  const submit = useMutation({
    mutationFn: () =>
      submitPaperOrder({ symbol, side, qty: Number(qty), type: "market" }),
    onSuccess: (o) => {
      setMessage(
        o.status === "rejected"
          ? `rejected: ${o.reason}`
          : `${o.status} ${o.side} ${o.filled_qty || o.qty} @ ${o.avg_price || "—"}`,
      );
      void qc.invalidateQueries({ queryKey: ["paper-account"] });
      void qc.invalidateQueries({ queryKey: ["paper-orders"] });
    },
    onError: (e: Error) => setMessage(e.message),
  });

  const cancel = useMutation({
    mutationFn: (id: number) => cancelPaperOrder(id),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["paper-orders"] });
      void qc.invalidateQueries({ queryKey: ["paper-account"] });
    },
  });

  const snap = account.data;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="font-display text-3xl text-zinc-900">Paper</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Risk-checked paper orders against the live last prices. Kill switch shows up when a
          daily-loss limit trips.
        </p>
      </div>

      {snap && (
        <div className="grid gap-4 sm:grid-cols-4">
          {[
            ["Equity", snap.equity.toFixed(2)],
            ["Cash", snap.cash.toFixed(2)],
            ["Open orders", String(snap.open_orders)],
            ["Breaches", String(snap.breaches)],
          ].map(([k, v]) => (
            <div key={k} className="rounded-lg border border-zinc-200 bg-white px-4 py-3">
              <p className="text-xs uppercase tracking-wide text-zinc-500">{k}</p>
              <p className="mt-1 font-medium tabular-nums">{v}</p>
            </div>
          ))}
        </div>
      )}
      {snap?.killed && (
        <p className="rounded border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900">
          Kill switch is on — new orders will be rejected until the broker is reset.
        </p>
      )}

      <form
        className="flex flex-wrap items-end gap-3 rounded-lg border border-zinc-200 bg-white p-4"
        onSubmit={(e) => {
          e.preventDefault();
          submit.mutate();
        }}
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
          Side
          <select
            className="mt-1 block rounded border border-zinc-300 px-2 py-1.5"
            value={side}
            onChange={(e) => setSide(e.target.value as "buy" | "sell")}
          >
            <option value="buy">buy</option>
            <option value="sell">sell</option>
          </select>
        </label>
        <label className="text-sm">
          Qty
          <input
            className="mt-1 block w-28 rounded border border-zinc-300 px-2 py-1.5 tabular-nums"
            value={qty}
            onChange={(e) => setQty(e.target.value)}
          />
        </label>
        <button
          type="submit"
          disabled={submit.isPending}
          className="rounded bg-teal-800 px-4 py-2 text-sm font-medium text-white hover:bg-teal-900 disabled:opacity-60"
        >
          Submit market
        </button>
        {message && <p className="w-full text-sm text-zinc-600">{message}</p>}
      </form>

      <section>
        <h2 className="text-sm font-medium text-zinc-700">Orders</h2>
        <div className="mt-2 overflow-x-auto rounded-lg border border-zinc-200 bg-white">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-zinc-200 text-xs uppercase text-zinc-500">
              <tr>
                <th className="px-3 py-2">Id</th>
                <th className="px-3 py-2">Symbol</th>
                <th className="px-3 py-2">Side</th>
                <th className="px-3 py-2">Qty</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2" />
              </tr>
            </thead>
            <tbody>
              {(orders.data ?? []).map((o) => (
                <tr key={o.id} className="border-b border-zinc-100">
                  <td className="px-3 py-2 tabular-nums">{o.id}</td>
                  <td className="px-3 py-2">{o.symbol}</td>
                  <td className="px-3 py-2">{o.side}</td>
                  <td className="px-3 py-2 tabular-nums">{o.qty}</td>
                  <td className="px-3 py-2">
                    {o.status}
                    {o.reason ? ` (${o.reason})` : ""}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {o.status === "open" && (
                      <button
                        type="button"
                        className="text-xs text-red-700 hover:underline"
                        onClick={() => cancel.mutate(o.id)}
                      >
                        cancel
                      </button>
                    )}
                  </td>
                </tr>
              ))}
              {(orders.data ?? []).length === 0 && (
                <tr>
                  <td colSpan={6} className="px-3 py-6 text-center text-zinc-400">
                    No orders yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
