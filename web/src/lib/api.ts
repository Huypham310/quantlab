// Typed client for the QuantLab FastAPI service. Shapes mirror backend schemas.py.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function wsUrl(path: string): string {
  const base = API_URL.replace(/^http/, "ws");
  return `${base}${path}`;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(0, `Can't reach the API at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail =
      typeof body?.detail === "string"
        ? body.detail
        : Array.isArray(body?.detail)
          ? body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ")
          : res.statusText;
    throw new ApiError(res.status, detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export interface Bar {
  ts: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface SymbolInfo {
  source: string;
  symbol: string;
  timeframe: string;
  bars: number;
  first: string;
  last: string;
}

export interface StrategyInfo {
  name: string;
  params: string[];
  grid: Record<string, unknown[]>;
}

export interface JobCreated {
  job_id: string;
  kind: "backtest" | "train";
}

export interface JobStatus {
  job_id: string;
  status: string;
  progress?: Record<string, unknown> | null;
  result?: Record<string, unknown> | null;
  error?: string | null;
}

export interface PaperOrder {
  id: number;
  symbol: string;
  side: string;
  qty: number;
  type: string;
  limit_price: number | null;
  created_ms: number;
  status: string;
  filled_qty: number;
  avg_price: number;
  reason: string;
  marketable: boolean;
}

export interface PaperAccount {
  cash: number;
  equity: number;
  positions: Record<string, number>;
  last: Record<string, number>;
  open_orders: number;
  fees_paid: number;
  killed: boolean;
  breaches: number;
}

export interface TradeTick {
  exchange: string;
  symbol: string;
  trade_id: string;
  price: number;
  size: number;
  side: string;
  ts_ms: number;
}

export const getHealth = () => request<{ status: string; database: boolean }>("/health");
export const listSymbols = () => request<SymbolInfo[]>("/symbols");
export const listStrategies = () => request<StrategyInfo[]>("/strategies");

export function getBars(
  symbol: string,
  opts: { timeframe?: string; source?: string; limit?: number } = {},
) {
  const q = new URLSearchParams({
    timeframe: opts.timeframe ?? "1h",
    source: opts.source ?? "coinbase",
    limit: String(opts.limit ?? 500),
  });
  return request<Bar[]>(`/bars/${encodeURIComponent(symbol)}?${q}`);
}

export function startBacktest(body: Record<string, unknown>) {
  return request<JobCreated>("/backtests", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const getJob = (id: string) => request<JobStatus>(`/jobs/${id}`);

export function submitPaperOrder(body: {
  symbol: string;
  side: "buy" | "sell";
  qty: number;
  type?: "market" | "limit";
  limit_price?: number;
}) {
  return request<PaperOrder>("/paper/orders", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export const listPaperOrders = (status?: string) =>
  request<PaperOrder[]>(`/paper/orders${status ? `?status=${status}` : ""}`);

export const cancelPaperOrder = (id: number) =>
  request<void>(`/paper/orders/${id}`, { method: "DELETE" });

export const getPaperAccount = () => request<PaperAccount>("/paper/account");

export function watchJob(
  jobId: string,
  onUpdate: (s: JobStatus) => void,
  onError?: (e: Event) => void,
): () => void {
  const ws = new WebSocket(wsUrl(`/ws/jobs/${jobId}`));
  ws.onmessage = (ev) => onUpdate(JSON.parse(ev.data) as JobStatus);
  if (onError) ws.onerror = onError;
  return () => ws.close();
}

export function watchTrades(
  symbol: string,
  onTrade: (t: TradeTick) => void,
  onError?: (e: Event) => void,
): () => void {
  const ws = new WebSocket(wsUrl(`/ws/trades/${encodeURIComponent(symbol)}`));
  ws.onmessage = (ev) => onTrade(JSON.parse(ev.data) as TradeTick);
  if (onError) ws.onerror = onError;
  return () => ws.close();
}
